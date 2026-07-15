# Production Runtime Packaging

The hosted alpha has two independent runtime artifacts, but only the audit worker is a container.

- API: API Gateway invokes Lambda code directly. Task 13 will package the AWS adapters and handler as a Lambda ZIP. A second container would add ECR storage, scanning, and cold-start work without adding an isolation boundary.
- Worker: AWS Batch starts `docker/worker.Dockerfile` on Fargate for each audit and stops it when the command exits.

The worker image pins its Python base by platform manifest digest and all Python dependencies exactly. It contains no compiler, package installer, shell entry point, or application secrets. Runtime UID/GID `10001` is non-root. The root filesystem is designed to be read-only; Batch supplies the bounded 20 GiB ephemeral workspace selected in the AWS ADR.

Batch jobs are finite processes, so their exit status and Batch state are the health check. A polling container health endpoint would add a second lifecycle with no consumer. `dqa.worker_entry` handles `SIGTERM` and `SIGINT` through normal Python unwinding; report writes remain atomic.

Run the local acceptance check:

```bash
python scripts/container_smoke.py --scan
```

It builds the image, verifies the configured runtime UID/GID, runs a real YOLO audit with a read-only root filesystem and 64 MiB temporary filesystem, verifies the generated summary, and rejects fixable critical/high vulnerabilities. The scanner itself is a versioned Trivy image pinned by its linux/amd64 manifest digest, so a Docker Hub login is not required. Unfixed findings must be reviewed during the release security task; suppressions require a documented expiry and rationale.
