"""Build and exercise the production worker image with a tiny real audit."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "dqa-worker:local"
TRIVY_IMAGE = "aquasec/trivy:0.69.3@sha256:7228e304ae0f610a1fad937baa463598cadac0c2ac4027cc68f3a8b997115689"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="enforce the fixable high/critical CVE policy")
    args = parser.parse_args()

    run("docker", "build", "--file", "docker/worker.Dockerfile", "--tag", IMAGE, ".")
    image_user = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Config.User}}", IMAGE],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if image_user != "10001:10001":
        raise RuntimeError(f"worker image has unexpected runtime user: {image_user!r}")

    with tempfile.TemporaryDirectory(prefix="dqa-container-") as raw:
        workspace = Path(raw)
        images = workspace / "dataset" / "train" / "images"
        labels = workspace / "dataset" / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / "sample.png").write_bytes(PNG)
        (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (workspace / "dataset" / "data.yaml").write_text(
            "path: .\ntrain: train/images\nnames: [object]\n", encoding="utf-8"
        )
        output = workspace / "out"
        output.mkdir()
        output.chmod(0o777)

        run(
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--volume",
            f"{(workspace / 'dataset').resolve()}:/workspace/dataset:ro",
            "--volume",
            f"{output.resolve()}:/workspace/out:rw",
            IMAGE,
            "audit",
            "--data",
            "/workspace/dataset/data.yaml",
            "--out",
            "/workspace/out",
            "--format",
            "json",
            "--fail-on",
            "critical",
        )

        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary["dataset"]["splits"]["train"]["images"] != 1:
            raise RuntimeError("worker smoke audit produced an unexpected summary")

    if args.scan:
        run(
            "docker",
            "run",
            "--rm",
            "--volume",
            "/var/run/docker.sock:/var/run/docker.sock",
            "--volume",
            "dqa-trivy-cache:/root/.cache/trivy",
            TRIVY_IMAGE,
            "image",
            "--exit-code",
            "1",
            "--ignore-unfixed",
            "--scanners",
            "vuln",
            "--severity",
            "CRITICAL,HIGH",
            "--skip-version-check",
            IMAGE,
        )

    print("worker image build, non-root/read-only audit, and artifact check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
