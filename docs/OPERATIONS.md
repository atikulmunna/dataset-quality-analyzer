# AWS operations runbook

This runbook covers the hosted alpha. It assumes `us-east-1`, an environment workspace matching `dev` or `prod`, and admission closed unless a controlled launch explicitly opens it.

## Monitoring and cost

AWS Batch state-change events invoke the monitor Lambda only when a job changes state. It publishes exactly three custom metrics per deployed environment under `DQA`: `QueueDepth`, `JobFailures`, and `AuditRuntimeSeconds`. AWS service metrics provide API 5xx errors, monitor-Lambda errors, and daily S3 bucket size. EventBridge performs no scheduled polling and the monitoring Lambda has no provisioned concurrency.

Batch events are delivered at least once, so failure counts and runtime samples are operational signals rather than billing-grade totals. Alarm thresholds are deliberately insensitive to duplicate delivery.

Six standard alarms notify the environment's operations SNS topic:

| Alarm | Response |
| --- | --- |
| `job-failure` | Diagnose the job ID, classify the error code, and keep admission closed if failures repeat. |
| `queue-backlog` | Check Batch compute-environment validity, image availability, quota, and admission state. |
| `long-audit` | Inspect dataset size and phase logs; the hard Batch timeout remains two hours. |
| `data-storage` | Verify lifecycle configuration and incomplete uploads before deleting only confirmed stale objects. |
| `api-5xx` | Search API access/Lambda logs by request ID and check downstream DynamoDB/Batch errors. |
| `monitor-errors` | Search monitor logs and replay the original Batch event only when safe. |

Confirm the SNS email subscription after each environment's first deployment. A pending subscription cannot send alerts.

CloudWatch's automatic AWS service dashboards remain free. The combined custom DQA dashboard is implemented but `enable_operations_dashboard` defaults to `false`: enable it only after checking that the shared AWS account still has a free custom-dashboard slot. The dashboard can otherwise add roughly USD 3/month, while the three custom metrics cost at most roughly USD 0.90/environment/month before the account-level free allowance. Keep the project below its USD 3 normal target by deploying only the environment currently in use and disabling the custom dashboard if the free allowance is shared by another project.

## Diagnose by job ID

Read deployment values without exposing Terraform state:

```powershell
Push-Location infra/terraform
$table = terraform output -raw state_table_name
$queue = terraform output -raw batch_queue_arn
$workerLogs = terraform output -raw worker_log_group
$monitorLogs = terraform output -raw monitor_log_group
Pop-Location

./scripts/diagnose_aws_job.ps1 `
  -JobId <job-id> `
  -TableName $table `
  -QueueArn $queue `
  -WorkerLogGroup $workerLogs `
  -MonitorLogGroup $monitorLogs
```

The command returns the consistent DynamoDB lifecycle record, retained Batch attempts, and structured worker/monitor events. Worker events contain `job_id`, attempt, duration, and a bounded error code, but never the owner token, archive contents, credentials, or dataset paths.

## Safe failure-alarm test

After deployment, invoke the monitor with a synthetic terminal Batch failure. This starts no Fargate job and uploads no data:

```powershell
./scripts/simulate_worker_failure.ps1 `
  -MonitorFunctionName dqa-dev-monitor `
  -QueueArn $queue
```

Within five minutes, `dqa-dev-job-failure` must enter `ALARM`, the confirmed SNS recipient must receive a notification, and the synthetic job ID must be searchable in `/aws/lambda/dqa-dev-monitor`. A later period with no failure data returns the alarm to `OK` without operator mutation.

## Incident sequence

1. Close admission first if cost, isolation, or repeated failure is suspected.
2. Capture the job ID, alarm name, UTC time, environment, error code, and Batch attempt ID.
3. Run the diagnostic command. Do not download customer data during routine triage.
4. Check the Batch compute environment and ECR image only if the worker never claimed the job.
5. Retry only through the defined lifecycle; never edit a terminal job record into `queued`.
6. Record cause, affected jobs, cost impact, and corrective action before reopening admission.

Close admission explicitly:

```powershell
aws dynamodb update-item `
  --table-name $table `
  --key '{"pk":{"S":"CONFIG#admission"}}' `
  --update-expression 'SET enabled = :closed' `
  --expression-attribute-values '{":closed":{"BOOL":false}}'
```

## Retention, deletion, and recovery

- Incomplete uploads expire after one day; uploaded source archives expire within two days.
- Successful artifacts expire within eight days. Job metadata expires through DynamoDB TTL after 30 days.
- Worker and monitor logs retain seven days in dev; production operational/API logs retain fourteen days.
- ECR keeps only the newest three immutable worker images.
- Terraform state is encrypted, versioned, and retains five recent noncurrent versions for 30 days.

The alpha intentionally does not enable DynamoDB point-in-time recovery: job metadata is short-lived coordination state, not a customer backup, and PITR would add recurring cost to each environment. Uploaded datasets and generated artifacts are also not backups; users must retain their originals. Terraform state versioning is the recovery mechanism for infrastructure metadata.

Before destroying an environment, close admission, wait for or cancel active jobs, export only incident records that must be retained, then run a reviewed Terraform destroy. Production Cognito deletion protection and non-empty buckets must be disabled deliberately; never bypass them as part of routine cleanup.
