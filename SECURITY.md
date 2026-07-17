# Security

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, datasets, artifacts, or AWS resources. Send a concise report to [atikul.munna@northsouth.edu](mailto:atikul.munna@northsouth.edu) with the affected surface, reproduction steps, and potential impact. Do not include real third-party datasets or credentials.

The project is a private alpha. There is no paid bug-bounty program or guaranteed response time, but credible reports will be investigated before broader access is granted.

## Supported surface

Security fixes target the current `main` branch and the single deployed private-alpha environment. Older local checkouts, generated reports, and expired worker images are not maintained releases.

## Private-alpha security model

- Public signup is disabled; users are created by the author.
- Cognito uses the authorization-code flow with S256 PKCE and no browser client secret.
- API Gateway validates the Cognito issuer, audience, and `dqa/jobs` scope.
- Every job, upload, source deletion, and artifact operation is checked against the token owner.
- Uploads use exact owner-scoped object keys, exact content-length policies, SHA-256 checksums, server-side encryption, and 15-minute presigned POSTs.
- Workers receive a job identifier rather than a command, filesystem path, URL, or user credential.
- ZIP ingestion rejects traversal, absolute paths, links, encryption, unsupported methods, duplicates, excessive entries, excessive expansion, and metadata/actual-size mismatches.
- The Batch worker runs as UID/GID `10001`, without privilege, with a read-only root filesystem, no inbound network rules, and one global worker maximum.
- Source data and artifacts expire automatically. Downloads use five-minute owner-checked presigned URLs.
- Deployment credentials are short-lived GitHub OIDC sessions; third-party Actions and container inputs are pinned to immutable digests.

## Final private-alpha review — 2026-07-17

Scope: browser bundle, Cognito/API authorization, upload and archive ingestion, job lifecycle, S3/DynamoDB ownership, Batch isolation, IAM, dependency/container supply chain, logging, cost-abuse controls, and the deployed dev environment.

| Finding | Severity | Resolution |
|---|---|---|
| Roboflow CLI extraction used `ZipFile.extractall` on a provider download | High | Replaced with the same validated, streamed, atomic ZIP extraction used by hosted workers; HTTPS and Roboflow reference-host validation added. |
| Debian worker base contained unfixed Perl CVEs reported by ECR (3 critical, 5 high) | Critical/High | Replaced with a digest-pinned Python 3.11 Alpine base. The complete local Trivy scan reports zero OS and Python vulnerabilities. |
| CloudFront responses lacked an explicit browser security policy | Medium | Added CSP, HSTS, frame denial, no-referrer, MIME-sniffing protection, and a restrictive Permissions Policy through Terraform. |
| SHA-1 finding fingerprints triggered cryptographic-use scanner findings | Informational | Marked each stable, non-security fingerprint hash with `usedforsecurity=False`; artifact identifiers remain backward compatible. |
| CloudFront's generated domain uses its fixed legacy-compatible TLS policy | Medium, accepted | AWS fixes the default-certificate policy to `TLSv1`; modern clients negotiate newer TLS. Requiring TLS 1.2 needs a custom domain and certificate, intentionally deferred to avoid alpha idle/domain cost. Revisit before public launch. |
| Cognito MFA is disabled | Medium, accepted for private alpha | Only three author-created accounts exist, signup is disabled, passwords require 12 characters and complexity, token revocation is enabled, and workload quotas limit abuse. Require MFA or passwordless/federated access before public signup. |

Review evidence:

- `pip-audit` against `requirements/worker.lock`: no known vulnerabilities.
- Bandit across `dqa/`: no medium or high findings; three low findings are intentional malformed-COCO skips.
- Trivy against the final worker image: zero OS or Python-package vulnerabilities.
- Container smoke: real audit succeeds as non-root with a read-only root filesystem.
- Terraform validation: valid configuration.
- Live AWS inspection: private S3 buckets, restricted CORS, encrypted storage, scoped JWT routes, API throttling, no worker ingress, HTTPS-only egress, immutable/scanned ECR, healthy alarms, enabled admission guard, USD 5 monthly budget, and USD 50 project budget.
- Secret-pattern review: matches are limited to defensive code/tests; no AWS access keys, private keys, client secrets, or test-user passwords are tracked.

No unresolved critical or high finding is accepted for the private alpha.

## Residual risks

- A valid invited user can intentionally submit adversarial but policy-compliant datasets. Runtime, queue, extraction, image-count, and cost ceilings bound the exposure; they do not make image decoders or the audit engine formally sandbox-safe.
- AWS Budgets data is delayed. The USD 40 admission cutoff, one-worker ceiling, per-owner queue limits, and operator kill switch are the enforceable controls.
- The alpha is not a compliance-certified service. Do not upload regulated, uniquely valuable, or legally restricted data.
- Browser tokens are held in session storage. A future same-origin script compromise could access them; the CSP and dependency-free frontend reduce this risk but cannot eliminate it.

## Before public availability

1. Move to a custom domain with a TLS 1.2-or-newer CloudFront policy.
2. Require MFA, passkeys, or a trusted federated identity provider.
3. Add an abuse/contact policy and decide whether AWS WAF is justified by measured traffic.
4. Repeat dependency, container, IAM, cross-owner, archive, load, and cost reviews.
5. Obtain legal review for the privacy notice and acceptable-use terms if access becomes public.
