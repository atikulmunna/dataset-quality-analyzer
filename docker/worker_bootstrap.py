"""Prepare the Fargate volume, permanently drop privileges, and start DQA."""

from __future__ import annotations

import os
from pathlib import Path
import sys


RUNTIME_UID = 10001
RUNTIME_GID = 10001


def main() -> int:
    if os.geteuid() != 0:
        raise RuntimeError("Worker bootstrap must start as root to prepare the mounted workspace.")

    workspace = Path(os.environ.get("DQA_WORKSPACE", "/workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    os.chown(workspace, RUNTIME_UID, RUNTIME_GID)
    os.chmod(workspace, 0o700)

    os.setgroups([])
    os.setgid(RUNTIME_GID)
    os.setuid(RUNTIME_UID)

    if os.environ.get("DQA_VERIFY_RUNTIME_IDENTITY") == "1":
        print(f"{os.geteuid()}:{os.getegid()}")
        return 0

    os.execv(sys.executable, [sys.executable, "-m", "dqa.worker_entry", *sys.argv[1:]])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
