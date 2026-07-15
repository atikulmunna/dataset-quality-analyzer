# AWS deployment

Deployment is intentionally manual. No push, pull request, schedule, or idle service can start a release. The workflow tests the selected revision, exercises and scans the worker image, deploys dev, and runs live health, authentication, and fail-closed-admission smoke checks. Production is a separate opt-in and its protected GitHub environment requires approval. A failed dev job is a dependency failure, so production is never offered for approval.

## One-time bootstrap

The persistent bootstrap contains only an encrypted/versioned S3 Terraform-state bucket, a GitHub OIDC provider, and a deployment role. It creates no compute and stores no long-lived AWS key in GitHub.

```powershell
./scripts/bootstrap_aws.ps1
```

The script safely reuses an account-level GitHub OIDC provider when one exists; otherwise, the CloudFormation stack creates it.

Configure repository environments named `development` and `production`. Require `atikulmunna` to approve production deployments. Add these repository or environment variables:

- `AWS_REGION=us-east-1`
- `AWS_DEPLOY_ROLE_ARN` from the bootstrap `DeployRoleArn` output
- `TF_STATE_BUCKET` from the bootstrap `StateBucketName` output
- `ALERT_EMAIL=atikul.munna@northsouth.edu`

The OIDC trust policy accepts tokens only from those two environments in `atikulmunna/dataset-quality-analyzer`. The Terraform backend uses native S3 lock files and keeps dev/prod state in separate workspace paths. State versions older than the newest five expire after 30 days.

## Release and rollback

Run **deploy-aws** from GitHub Actions. Supply a branch, tag, or commit SHA from `main`'s history; commits from unmerged branches are rejected. Leave **promote_production** off for a dev-only release; turn it on to request production approval only after dev passes.

To roll back, rerun the workflow with the last known-good 40-character commit SHA. ECR tags are commit SHAs and immutable. If that tag already exists, the workflow reuses it; otherwise, it publishes the exact image artifact that passed the scan. Terraform then restores the Lambda package, infrastructure configuration, and Batch job-definition image reference from that revision.

Every deployment explicitly leaves job admission disabled. Task 18's controlled launch procedure must open admission only after the private-alpha checklist and cost checks pass.
