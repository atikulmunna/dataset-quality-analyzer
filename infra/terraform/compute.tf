resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/batch/${local.name}-worker"
  retention_in_days = var.environment == "prod" ? 14 : 7
}

resource "aws_batch_compute_environment" "audit" {
  name         = "${local.name}-fargate"
  type         = "MANAGED"
  state        = "ENABLED"
  service_role = aws_iam_role.batch_service.arn

  compute_resources {
    max_vcpus          = 2
    security_group_ids = [aws_security_group.worker.id]
    subnets            = aws_subnet.worker[*].id
    type               = "FARGATE"
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

resource "aws_batch_job_queue" "audit" {
  name     = "${local.name}-audit"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.audit.arn
  }
}

resource "aws_batch_job_definition" "audit" {
  name                  = "${local.name}-audit"
  type                  = "container"
  platform_capabilities = ["FARGATE"]
  propagate_tags        = true
  parameters            = { job_id = "missing" }

  retry_strategy { attempts = 3 }
  timeout { attempt_duration_seconds = 7200 }

  container_properties = jsonencode({
    image                        = "${aws_ecr_repository.worker.repository_url}:${var.worker_image_tag}"
    command                      = ["hosted", "--job-id", "Ref::job_id"]
    jobRoleArn                   = aws_iam_role.worker_task.arn
    executionRoleArn             = aws_iam_role.worker_execution.arn
    readonlyRootFilesystem       = true
    privileged                   = false
    networkConfiguration         = { assignPublicIp = "ENABLED" }
    fargatePlatformConfiguration = { platformVersion = "1.4.0" }
    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "4096" }
    ]
    environment = [
      { name = "DQA_TABLE_NAME", value = aws_dynamodb_table.state.name },
      { name = "DQA_DATA_BUCKET", value = aws_s3_bucket.data.id },
      { name = "DQA_WORKSPACE", value = "/workspace" },
      { name = "DQA_CONFIG_ROOT", value = "/opt/dqa/configs" },
      { name = "TMPDIR", value = "/workspace" }
    ]
    volumes = [{ name = "workspace" }]
    mountPoints = [{
      sourceVolume  = "workspace"
      containerPath = "/workspace"
      readOnly      = false
    }]
    linuxParameters = { initProcessEnabled = true }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.worker.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "audit"
      }
    }
  })
}
