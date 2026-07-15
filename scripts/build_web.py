"""Build the dependency-free hosted UI with a public runtime configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web"
STATIC_FILES = ("index.html", "styles.css", "app.js")
ASSET_FILES = {"logo.png": ROOT / "assets" / "logo.png"}


def build(
    output: Path,
    *,
    mode: str,
    api_base_url: str = "",
    cognito_domain: str = "",
    cognito_client_id: str = "",
    cognito_redirect_uri: str = "",
) -> None:
    if mode not in {"preview", "live"}:
        raise ValueError("mode must be preview or live")
    values = {
        "mode": mode,
        "apiBaseUrl": api_base_url.rstrip("/"),
        "cognitoDomain": cognito_domain.rstrip("/"),
        "cognitoClientId": cognito_client_id,
        "cognitoRedirectUri": cognito_redirect_uri,
    }
    if mode == "live" and not all(values[key] for key in values if key != "mode"):
        raise ValueError("live mode requires API and Cognito public configuration")

    output.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_FILES:
        shutil.copyfile(SOURCE / filename, output / filename)
    for filename, source in ASSET_FILES.items():
        shutil.copyfile(source, output / filename)
    config = "window.DQA_CONFIG = Object.freeze(" + json.dumps(values, sort_keys=True, separators=(",", ":")) + ");\n"
    (output / "config.js").write_text(config, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "web")
    parser.add_argument("--mode", choices=("preview", "live"), default="preview")
    parser.add_argument("--api-base-url", default="")
    parser.add_argument("--cognito-domain", default="")
    parser.add_argument("--cognito-client-id", default="")
    parser.add_argument("--cognito-redirect-uri", default="")
    args = parser.parse_args()
    build(
        args.output.resolve(),
        mode=args.mode,
        api_base_url=args.api_base_url,
        cognito_domain=args.cognito_domain,
        cognito_client_id=args.cognito_client_id,
        cognito_redirect_uri=args.cognito_redirect_uri,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
