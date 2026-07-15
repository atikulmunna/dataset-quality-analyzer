# ADR-001: AWS Hosted-Alpha Architecture

Status: accepted for implementation  
Date: 2026-07-12  
Region: `us-east-1`  
Cost ceiling: USD 50 total project spend, including promotional credits

## Account evidence

Read-only AWS checks on 2026-07-12 confirmed:

- Account: `892217934322`
- Principal: IAM user `munna`
- Account plan: `PAID`, `ACTIVE`
- Remaining promotional credits: USD 119.04
- July month-to-date unblended cost: effectively USD 0.00
- Default region: `us-east-1`
- Existing budget: USD 100/month with one alert at 80%; this is unsafe for the project envelope and must be replaced during infrastructure work.
- AWS could not produce a cost forecast because the account has insufficient history.

Credit expiration is not returned by the queried account-plan API. The Billing console remains authoritative for the expiration date.

## Decision

Use a single-region, scale-to-zero control plane and on-demand audit workers:

```text
CloudFront
  └── private S3 static UI

Browser
  ├── Cognito user pool (invite-ready users)
  ├── API Gateway HTTP API
  │     └── Lambda control-plane handlers
  │           ├── DynamoDB jobs, leases, idempotency, rate counters
  │           ├── S3 presigned upload POSTs
  │           └── AWS Batch SubmitJob
  └── direct dataset upload to private S3

AWS Batch queue
  └── ECS Fargate task (2 vCPU, 4 GiB, 20 GiB ephemeral)
        ├── validate/extract source from S3
        ├── run DQA
        └── publish immutable attempt artifacts to S3

Batch events → EventBridge → Lambda → conditional DynamoDB transition
Logs/metrics → CloudWatch
Container images → ECR
```

The API Lambda does not run audits and does not join a VPC. Fargate tasks run only while jobs exist, in public subnets with no inbound security-group rules. A temporary public IPv4 address permits ECR/S3 access without a continuously billed NAT Gateway or multiple interface endpoints. Hosted Roboflow ingestion remains disabled, so workers do not handle user API credentials.

Use default S3 encryption (SSE-S3) and DynamoDB service encryption for alpha. Do not introduce a customer-managed KMS key until a concrete compliance requirement justifies its fixed and request costs.

## Initial quotas

- Global Batch worker concurrency: 1
- Per owner: 1 running and 1 queued job
- Maximum audit duration across retries: 2 hours
- Worker attempts: 3
- Dataset upload: 2 GiB compressed, 10 GiB expanded, 25,000 images
- Near-duplicate analysis: at most 5,000 images
- Source retention: 24 hours after completion
- Successful artifacts: 7 days
- Failed/cancelled artifacts: 48 hours
- Job metadata: 30 days

## Cost worksheet

The main variable cost is Fargate. AWS's current `us-east-1` Linux/x86 example prices are USD 0.000011244 per vCPU-second and USD 0.000001235 per GB-second. At 2 vCPU and 4 GiB:

- CPU: `2 × 0.000011244 × 3600` = USD 0.08096/hour
- Memory: `4 × 0.000001235 × 3600` = USD 0.01778/hour
- Worker compute: approximately USD 0.09874/hour
- Maximum two-hour worker: approximately USD 0.1975
- Temporary public IPv4 allowance: approximately USD 0.01 for two hours at USD 0.005/hour
- Conservative maximum compute/network allowance per audit: USD 0.21

Alpha workload envelope:

| Item | Monthly assumption | Estimated cost before credits |
|---|---:|---:|
| Fargate + temporary IPv4 | 10 audits at the full 2-hour limit | USD 2.08 |
| S3 source/artifact storage | lifecycle-controlled, low single-digit GB-month | < USD 0.15 |
| ECR | one compressed image around 0.5 GiB | about USD 0.05 |
| Lambda | below 1M requests and 400k GB-seconds | USD 0 within monthly free tier |
| CloudFront/Cognito | far below always-free request/MAU limits | USD 0 expected |
| DynamoDB/API Gateway/CloudWatch | low alpha traffic | negligible; monitor |
| Total conservative alpha target | 10 worst-case audits/month | < USD 3/month |

Normal audits should finish well below two hours. Stop accepting new audits once project-attributed cumulative spend reaches USD 40, leaving USD 10 for delayed billing data, retained storage, and cleanup. A stale deployment should normally cost only low cents to approximately USD 1/month for retained S3 objects, ECR images, DynamoDB data, and CloudWatch logs; Lambda, API Gateway, Batch, and Fargate have no continuously running compute in this design.

AWS Budgets is an alerting control, not a universal hard spending cap. The API admission switch, Batch concurrency limit, retention rules, and the ability to disable the Batch queue are the enforceable workload controls. The generated AWS service domains remain in use during alpha to avoid Route 53 and domain-registration idle costs.

## Cost controls required before deployment

- Replace the current USD 100 monthly budget.
- Monthly alerts: USD 1 actual and USD 5 forecast.
- Cumulative project alerts: USD 25, USD 40, and USD 48.
- Reject new job submissions at USD 40 cumulative project spend and disable the Batch queue until explicitly re-enabled.
- Enable Free Tier and cost-anomaly notifications.
- Tag every resource with `Project=dqa`, `Environment`, and `ManagedBy`.
- Set explicit CloudWatch log retention; do not retain debug logs indefinitely.
- Apply S3 lifecycle expiration and DynamoDB TTL from the documented retention policy.
- Keep Batch maximum vCPU at 2 initially, enforcing global concurrency 1.

## Explicit exclusions

Do not use during alpha:

- NAT Gateway
- Application Load Balancer
- Always-running ECS service or EC2 instance
- RDS, OpenSearch, ElastiCache, EFS, or EKS
- Multi-region replication
- Provisioned Lambda concurrency
- Customer-managed KMS keys
- AWS WAF paid rules
- Custom domain/Route 53 hosted zone until the generated CloudFront/API domains are validated
- Savings Plans or other spend commitments

## Alternatives rejected

- Single EC2 host: initially simple, but continuously billed and weak for per-job isolation.
- App Runner: good for an HTTP service, but audit execution still requires asynchronous workers.
- Lambda audit workers: execution time and temporary workspace do not fit large audits.
- ECS service workers: creates idle cost; Batch already supplies queueing, retries, and on-demand capacity.
- Private subnets with NAT Gateway: the hourly NAT charge conflicts with the entire monthly budget.
- EKS: operational and cost complexity is unjustified.

## Pricing references

- [AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/)
- [AWS Batch pricing](https://aws.amazon.com/batch/pricing/)
- [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/)
- [Amazon DynamoDB pricing](https://aws.amazon.com/dynamodb/pricing/)
- [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/)
- [Amazon Cognito pricing](https://aws.amazon.com/cognito/pricing/)
- [Amazon CloudFront free tier](https://aws.amazon.com/cloudfront/faqs/)
- [Amazon VPC pricing](https://aws.amazon.com/vpc/pricing/)
