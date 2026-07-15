# DQA AWS Terraform

This root module creates the accepted scale-to-zero alpha architecture in `us-east-1`. Dev and prod use different Terraform workspaces, resource names, Cognito pools, buckets, tables, queues, and IAM roles. The module refuses a plan when the selected workspace does not match `var.environment`.

The deployment starts fail-closed: `admission_enabled = false`. Publish an immutable worker image before deliberately enabling job admission. The seed item deliberately ignores later Terraform changes so a routine apply cannot reopen a switch closed by the cost guard; enabling it is an explicit DynamoDB `UpdateItem` operation. At USD 40 of annual project-tagged actual cost, AWS Budgets publishes to the cost-guard Lambda, which closes the switch. Budget data is delayed, so this complements—rather than replaces—the single-worker Batch ceiling.

Build and validate without changing AWS resources:

```powershell
python scripts/build_lambda.py
Push-Location infra/terraform
terraform init -backend=false
terraform workspace new dev
terraform validate
terraform plan -var-file="environments/dev.tfvars"
Pop-Location
```

Live deployments use the versioned S3 backend and GitHub OIDC bootstrap documented in [`../../docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md). The backend uses native S3 lock files rather than an extra DynamoDB lock table. Do not commit a local backend configuration or AWS credentials.

Review the saved plan before `apply`. The budget email subscription may require confirmation. Use the generated CloudFront/API domains during alpha; this module intentionally creates no NAT Gateway, load balancer, custom domain, Route 53 zone, database server, always-running compute, or customer-managed KMS key.

Before destruction, disable admission and allow active Batch jobs to finish. Production Cognito deletion protection and non-empty buckets intentionally prevent casual destruction; change those controls explicitly only for a planned teardown.

The account currently cannot reserve Lambda concurrency without violating AWS's required unreserved-concurrency floor. The stack therefore uses API Gateway stage throttling (burst 4, sustained 2 requests/second), authenticated routes, application owner limits, and Batch's two-vCPU ceiling instead of provisioned or reserved Lambda capacity. Revisit this only after an account quota increase; reservations do not reduce idle cost.

The 2026-07-15 dev rehearsal deployed all 64 resources, returned `200` from `/health`, returned `401` for an unauthenticated job submission, reported a `VALID` two-vCPU Fargate environment, showed admission disabled, reached a zero-change follow-up plan, and destroyed all 64 resources. Terraform state and direct AWS checks were empty afterward. No worker image was published and no audit job ran.
