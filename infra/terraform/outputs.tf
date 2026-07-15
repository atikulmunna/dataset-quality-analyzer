output "api_url" {
  value = aws_apigatewayv2_api.api.api_endpoint
}

output "ui_url" {
  value = "https://${aws_cloudfront_distribution.ui.domain_name}"
}

output "cognito_hosted_ui_domain" {
  value = "https://${aws_cognito_user_pool_domain.users.domain}.auth.${var.aws_region}.amazoncognito.com"
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.users.id
}

output "cognito_client_id" {
  value = aws_cognito_user_pool_client.ui.id
}

output "worker_repository_url" {
  value = aws_ecr_repository.worker.repository_url
}

output "data_bucket" {
  value = aws_s3_bucket.data.id
}

output "admission_enabled" {
  value = var.admission_enabled
}

