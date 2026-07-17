# Private alpha launch checklist

Date: 2026-07-17  
Owner: Atikul Munna  
Environment: `dev`, `us-east-1`  
Audience: selected testers with author-issued credentials

This checklist is the release record for Tasks 16–18. A checked item is backed by an automated test, an AWS query, or a live acceptance run. It is not a promise of production-grade availability.

## Security and abuse gate

- [x] Uploads are owner-scoped direct-to-S3 POSTs with declared size and SHA-256 constraints.
- [x] Archive validation rejects traversal, links, encryption, duplicate entries, unsafe expansion, unsupported compression, and oversized inputs.
- [x] Roboflow downloads require HTTPS and validated hosts; extraction uses the same safe ZIP boundary.
- [x] Cognito uses Authorization Code + S256 PKCE, admin-created accounts, no client secret, and no public signup.
- [x] API reads, artifacts, deletion, rates, and quotas are owner-scoped and fail closed.
- [x] S3 origins/data are private and encrypted; CloudFront adds CSP, HSTS, framing, MIME, referrer, and permissions headers.
- [x] Worker image is digest-pinned, non-root, read-only-root compatible, and scanned with no detected OS/Python vulnerabilities.
- [x] GitHub Actions dependencies are pinned to full commit SHAs and AWS deployment uses short-lived OIDC credentials.
- [x] No committed credentials were found by the repository scan.
- [x] Accepted alpha risks are recorded in `SECURITY.md`: no MFA, AWS-managed CloudFront compatibility TLS policy, and no WAF/custom domain.

## Load, failure, and cost gate

- [x] A 25,000-image local maximum-size audit completed with 25,000 indexed images and 25,001 findings in 60.385 audit seconds (60.797 wall seconds).
- [x] Worker timeout spans retries and is capped at two hours; crash/retry and stale-worker behavior are regression-tested.
- [x] Corrupt and malicious input cases fail before audit execution.
- [x] Simulated storage exhaustion leaves neither a destination nor a partial temporary extraction.
- [x] API submission failure returns a bounded `503` and now produces an operator diagnostic.
- [x] Concurrent authenticated history requests exposed and led to repair of missing DynamoDB GSI access.
- [x] One tiny hosted audit completed in 26.67 seconds end to end, indexed one image, published five downloadable artifacts, and returned `204` for source deletion (job `c0508c34f1674f2c8a3b01b386493635`).
- [x] The successful Batch task ran for 25.015 seconds; the projected billed worker cost is recorded below.

## Controlled-launch gate

- [x] The private-alpha URL, author-issued credential model, limits, and local alternative are documented.
- [x] Privacy, retention, deletion, and support expectations are documented in `PRIVACY.md` and `SUPPORT.md`.
- [x] Admission is fail-closed in infrastructure and can be disabled independently of the UI/API.
- [x] Monthly and project budgets, a USD 40 workload cutoff, global worker concurrency 1, and short retention are active.
- [x] Six CloudWatch alarms exist for failure, backlog, runtime, storage, API 5xx, and monitor errors.
- [x] Live telemetry captured RUNNABLE, STARTING, RUNNING, and SUCCEEDED events with the same job ID and a 25.015-second runtime; the intentional pre-fix failures also exercised the API/job-failure alarms.
- [x] Immutable-SHA rollback deployed `33634e8…` in run `29575979519`; run `29576160983` then restored verified release `8b157e9…` and passed its live smoke gate.
- [x] Final full regression suite passed with 134 tests; the release commit was clean before this evidence-only documentation update.

## Cost evidence

The worker requests 2 vCPU and 4 GiB memory. AWS bills Linux/x86 Fargate per second with a one-minute minimum; the included 20 GiB ephemeral storage has no additional storage charge. At the published `us-east-1` rates, the minimum-minute compute charge is:

`60 × ((2 × $0.000011244) + (4 × $0.000001235)) = $0.00164568`

The task's public IPv4 address exists only with the one-shot task. Even pessimistically treating the published USD 0.005/hour IPv4 rate as a full hour, the measured audit remains below USD 0.007 before tiny API, Lambda, DynamoDB, log, request, and storage usage or any applicable free allowance. Ten similar audits would therefore remain far below USD 1; the USD 40 admission cutoff and USD 50 project ceiling remain unchanged.

AWS Budgets reported USD 0 actual spend for both the USD 5 monthly and USD 50 project budgets immediately after testing; billing data can lag and is not used as a real-time hard cap.

## Decision

Public signup remains **not approved**. The approved launch mode is a private alpha for selected testers, one global worker, author-issued credentials, and a USD 40 admission cutoff under the USD 50 total project ceiling.
