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
- [ ] One tiny hosted audit completes, produces downloadable artifacts, and permits source deletion.
- [ ] Measured hosted task duration and projected per-audit cost are recorded below.

## Controlled-launch gate

- [x] The private-alpha URL, author-issued credential model, limits, and local alternative are documented.
- [x] Privacy, retention, deletion, and support expectations are documented in `PRIVACY.md` and `SUPPORT.md`.
- [x] Admission is fail-closed in infrastructure and can be disabled independently of the UI/API.
- [x] Monthly and project budgets, a USD 40 workload cutoff, global worker concurrency 1, and short retention are active.
- [x] Six CloudWatch alarms exist for failure, backlog, runtime, storage, API 5xx, and monitor errors.
- [ ] Live telemetry is healthy after the hosted acceptance audit.
- [ ] Immutable-SHA rollback is executed and the final release is restored.
- [ ] Final full regression suite and clean-worktree check pass.

## Cost evidence

The worker requests 2 vCPU and 4 GiB memory. AWS bills Linux/x86 Fargate per second with a one-minute minimum; the included 20 GiB ephemeral storage has no additional storage charge. A public IPv4 address is charged only while the one-shot task exists. Control-plane requests and short-lived storage are usage-based and expected to remain within free allowances or below one cent per small alpha audit.

Measured values and the final sign-off are appended after the hosted acceptance run.

## Decision

Public signup remains **not approved**. The approved launch mode is a private alpha for selected testers, one global worker, author-issued credentials, and a USD 40 admission cutoff under the USD 50 total project ceiling.
