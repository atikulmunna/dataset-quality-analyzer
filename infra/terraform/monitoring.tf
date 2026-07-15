locals {
  metric_namespace = "DQA"
  metric_dimensions = {
    Environment = var.environment
  }
  alarm_actions = [aws_sns_topic.operations.arn]
}

resource "aws_sns_topic" "operations" {
  name = "${local.name}-operations"
}

resource "aws_sns_topic_subscription" "operations_email" {
  topic_arn = aws_sns_topic.operations.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_log_group" "monitor" {
  name              = "/aws/lambda/${local.name}-monitor"
  retention_in_days = var.environment == "prod" ? 14 : 7
}

resource "aws_iam_role" "monitor" {
  name               = "${local.name}-monitor"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "monitor_logs" {
  role       = aws_iam_role.monitor.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "monitor" {
  name = "${local.name}-monitor"
  role = aws_iam_role.monitor.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "CountOnlyThisQueue"
        Effect   = "Allow"
        Action   = "batch:ListJobs"
        Resource = "*"
      },
      {
        Sid      = "PublishOnlyDqaMetrics"
        Effect   = "Allow"
        Action   = "cloudwatch:PutMetricData"
        Resource = "*"
        Condition = {
          StringEquals = { "cloudwatch:namespace" = local.metric_namespace }
        }
      }
    ]
  })
}

resource "aws_lambda_function" "monitor" {
  function_name    = "${local.name}-monitor"
  role             = aws_iam_role.monitor.arn
  runtime          = "python3.11"
  handler          = "dqa.aws.monitoring.handler"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  architectures    = ["x86_64"]
  memory_size      = 128
  timeout          = 15

  environment {
    variables = {
      DQA_BATCH_QUEUE_ARN  = aws_batch_job_queue.audit.arn
      DQA_METRIC_NAMESPACE = local.metric_namespace
      DQA_ENVIRONMENT      = var.environment
    }
  }

  depends_on = [aws_cloudwatch_log_group.monitor, aws_iam_role_policy_attachment.monitor_logs]
}

resource "aws_cloudwatch_event_rule" "batch_state" {
  name        = "${local.name}-batch-state"
  description = "Emit bounded operational metrics when DQA Batch jobs change state."
  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail = {
      jobQueue = [aws_batch_job_queue.audit.arn]
    }
  })
}

resource "aws_cloudwatch_event_target" "batch_state_monitor" {
  rule = aws_cloudwatch_event_rule.batch_state.name
  arn  = aws_lambda_function.monitor.arn
}

resource "aws_lambda_permission" "eventbridge_monitor" {
  statement_id  = "AllowBatchEvents"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.monitor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.batch_state.arn
}

resource "aws_cloudwatch_metric_alarm" "job_failure" {
  alarm_name          = "${local.name}-job-failure"
  alarm_description   = "At least one terminal AWS Batch audit failure occurred. Search worker logs by job_id."
  namespace           = local.metric_namespace
  metric_name         = "JobFailures"
  dimensions          = local.metric_dimensions
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "queue_backlog" {
  alarm_name          = "${local.name}-queue-backlog"
  alarm_description   = "At least two audits remained queued for 15 minutes."
  namespace           = local.metric_namespace
  metric_name         = "QueueDepth"
  dimensions          = local.metric_dimensions
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 3
  threshold           = 2
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "long_audit" {
  alarm_name          = "${local.name}-long-audit"
  alarm_description   = "An audit used at least 90 minutes of runtime; investigate before the two-hour timeout."
  namespace           = local.metric_namespace
  metric_name         = "AuditRuntimeSeconds"
  dimensions          = local.metric_dimensions
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5400
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "data_storage" {
  alarm_name        = "${local.name}-data-storage"
  alarm_description = "The transient data/artifact bucket exceeded 20 GiB. Verify lifecycle processing."
  namespace         = "AWS/S3"
  metric_name       = "BucketSizeBytes"
  dimensions = {
    BucketName  = aws_s3_bucket.data.id
    StorageType = "StandardStorage"
  }
  statistic           = "Average"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 20 * 1024 * 1024 * 1024
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "api_server_errors" {
  alarm_name          = "${local.name}-api-5xx"
  alarm_description   = "The HTTP API returned a server-side error."
  namespace           = "AWS/ApiGateway"
  metric_name         = "5xx"
  dimensions          = { ApiId = aws_apigatewayv2_api.api.id }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "monitor_errors" {
  alarm_name          = "${local.name}-monitor-errors"
  alarm_description   = "The event-driven monitoring Lambda failed to process a Batch transition."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.monitor.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_dashboard" "operations" {
  count          = var.enable_operations_dashboard ? 1 : 0
  dashboard_name = "${local.name}-operations"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Audit queue, failures, and runtime"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            [local.metric_namespace, "QueueDepth", "Environment", var.environment],
            [".", "JobFailures", ".", "."],
            [".", "AuditRuntimeSeconds", ".", ".", { yAxis = "right" }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "API errors and transient storage"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            ["AWS/ApiGateway", "5xx", "ApiId", aws_apigatewayv2_api.api.id],
            ["AWS/S3", "BucketSizeBytes", "BucketName", aws_s3_bucket.data.id, "StorageType", "StandardStorage", { yAxis = "right" }]
          ]
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 6
        width  = 24
        height = 6
        properties = {
          title  = "Recent worker events by job ID"
          region = var.aws_region
          view   = "table"
          query  = "SOURCE '${aws_cloudwatch_log_group.worker.name}' | fields @timestamp, event, job_id, attempt, duration_seconds, error_code | filter ispresent(job_id) | sort @timestamp desc | limit 50"
        }
      }
    ]
  })
}
