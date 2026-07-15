resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.name}-api"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "cost_guard" {
  name              = "/aws/lambda/${local.name}-cost-guard"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "api_access" {
  name              = "/aws/apigateway/${local.name}"
  retention_in_days = 14
}

resource "aws_lambda_function" "api" {
  function_name    = "${local.name}-api"
  role             = aws_iam_role.api.arn
  runtime          = "python3.11"
  handler          = "dqa.aws.api_handler.handler"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  architectures    = ["x86_64"]
  memory_size      = 256
  timeout          = 15

  environment {
    variables = {
      DQA_TABLE_NAME         = aws_dynamodb_table.state.name
      DQA_DATA_BUCKET        = aws_s3_bucket.data.id
      DQA_BATCH_QUEUE_ARN    = aws_batch_job_queue.audit.arn
      DQA_JOB_DEFINITION_ARN = aws_batch_job_definition.audit.arn
    }
  }

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy_attachment.api_logs]
}

resource "aws_lambda_function" "cost_guard" {
  function_name    = "${local.name}-cost-guard"
  role             = aws_iam_role.cost_guard.arn
  runtime          = "python3.11"
  handler          = "dqa.aws.cost_guard.handler"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  architectures    = ["x86_64"]
  memory_size      = 128
  timeout          = 10

  environment {
    variables = { DQA_TABLE_NAME = aws_dynamodb_table.state.name }
  }

  depends_on = [aws_cloudwatch_log_group.cost_guard, aws_iam_role_policy_attachment.cost_guard_logs]
}

resource "aws_cognito_user_pool" "users" {
  name                     = "${local.name}-users"
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]
  deletion_protection      = var.environment == "prod" ? "ACTIVE" : "INACTIVE"

  admin_create_user_config { allow_admin_create_user_only = true }
  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_numbers                  = true
    require_symbols                  = true
    require_uppercase                = true
    temporary_password_validity_days = 7
  }
}

resource "aws_cognito_resource_server" "api" {
  identifier   = "dqa"
  name         = "DQA API"
  user_pool_id = aws_cognito_user_pool.users.id
  scope {
    scope_name        = "jobs"
    scope_description = "Submit and manage owned DQA jobs"
  }
}

resource "aws_cognito_user_pool_client" "ui" {
  name                                 = "${local.name}-ui"
  user_pool_id                         = aws_cognito_user_pool.users.id
  generate_secret                      = false
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "dqa/jobs"]
  callback_urls                        = ["https://${aws_cloudfront_distribution.ui.domain_name}/callback"]
  logout_urls                          = ["https://${aws_cloudfront_distribution.ui.domain_name}/"]
  supported_identity_providers         = ["COGNITO"]
  prevent_user_existence_errors        = "ENABLED"
  access_token_validity                = 1
  id_token_validity                    = 1
  refresh_token_validity               = 7
  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }
  depends_on = [aws_cognito_resource_server.api]
}

resource "aws_cognito_user_pool_domain" "users" {
  domain       = "${local.name}-${data.aws_caller_identity.current.account_id}"
  user_pool_id = aws_cognito_user_pool.users.id
}

resource "aws_apigatewayv2_api" "api" {
  name          = local.name
  protocol_type = "HTTP"
  cors_configuration {
    allow_headers = ["authorization", "content-type", "idempotency-key"]
    allow_methods = ["GET", "POST", "DELETE", "OPTIONS"]
    allow_origins = ["https://${aws_cloudfront_distribution.ui.domain_name}"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_authorizer" "cognito" {
  api_id           = aws_apigatewayv2_api.api.id
  authorizer_type  = "JWT"
  identity_sources = ["$request.header.Authorization"]
  name             = "cognito"
  jwt_configuration {
    audience = [aws_cognito_user_pool_client.ui.id]
    issuer   = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.users.id}"
  }
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
  timeout_milliseconds   = 15000
}

locals {
  protected_routes = toset(["POST /jobs", "GET /jobs/{job_id}", "DELETE /jobs/{job_id}", "POST /uploads"])
}

resource "aws_apigatewayv2_route" "protected" {
  for_each             = local.protected_routes
  api_id               = aws_apigatewayv2_api.api.id
  route_key            = each.value
  target               = "integrations/${aws_apigatewayv2_integration.api.id}"
  authorization_type   = "JWT"
  authorizer_id        = aws_apigatewayv2_authorizer.cognito.id
  authorization_scopes = ["dqa/jobs"]
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 4
    throttling_rate_limit  = 2
  }
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_access.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      routeKey         = "$context.routeKey"
      status           = "$context.status"
      responseLength   = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
    })
  }
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
