"""Build the dependency-free API/cost-guard Lambda ZIP deterministically."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    ROOT / "dqa" / "__init__.py",
    *(ROOT / "dqa" / "web").glob("*.py"),
    ROOT / "dqa" / "aws" / "__init__.py",
    ROOT / "dqa" / "aws" / "adapters.py",
    ROOT / "dqa" / "aws" / "api_handler.py",
    ROOT / "dqa" / "aws" / "cost_guard.py",
    ROOT / "dqa" / "aws" / "monitoring.py",
    ROOT / "dqa" / "aws" / "observability.py",
)


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(INCLUDE, key=lambda item: item.as_posix()):
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "lambda" / "dqa-api.zip")
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
