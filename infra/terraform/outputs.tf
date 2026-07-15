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

output "state_table_name" {
  value = aws_dynamodb_table.state.name
}

output "batch_queue_arn" {
  value = aws_batch_job_queue.audit.arn
}

output "worker_log_group" {
  value = aws_cloudwatch_log_group.worker.name
}

output "monitor_log_group" {
  value = aws_cloudwatch_log_group.monitor.name
}

output "monitor_function_name" {
  value = aws_lambda_function.monitor.function_name
}

output "operations_alert_topic" {
  value = aws_sns_topic.operations.arn
}

output "operations_dashboard" {
  value = var.enable_operations_dashboard ? aws_cloudwatch_dashboard.operations[0].dashboard_name : null
}
