from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from scripts.build_web import ASSET_FILES, STATIC_FILES, build


ROOT = Path(__file__).resolve().parents[1]


class SurfaceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[str] = []
        self.has_language = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "script" and values.get("src"):
            self.scripts.append(str(values["src"]))
        if tag == "html" and values.get("lang") == "en":
            self.has_language = True


def test_hosted_ui_has_accessible_product_surfaces_without_remote_assets() -> None:
    parser = SurfaceParser()
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    parser.feed(html)

    assert parser.has_language
    assert {"main-content", "auditForm", "datasetFile", "recentRunsBody", "allRunsBody"} <= parser.ids
    assert parser.scripts == ["config.js", "app.js"]
    assert "https://" not in html
    assert "Credentials are provided by the author" in html
    assert "Public signup is disabled" in html
    assert 'id="accessNotice"' in html
    assert '<strong id="accountName">Sign in</strong>' in html


def test_web_build_is_exact_and_runtime_configuration_is_public_only(tmp_path: Path) -> None:
    output = tmp_path / "site"
    build(output, mode="preview")

    assert {path.name for path in output.iterdir()} == {*STATIC_FILES, *ASSET_FILES, "config.js"}
    assert (output / "logo.png").read_bytes() == (ROOT / "assets" / "logo.png").read_bytes()
    raw = (output / "config.js").read_text(encoding="utf-8")
    payload = raw.removeprefix("window.DQA_CONFIG = Object.freeze(").removesuffix(");\n")
    assert json.loads(payload) == {
        "apiBaseUrl": "",
        "cognitoClientId": "",
        "cognitoDomain": "",
        "cognitoRedirectUri": "",
        "mode": "preview",
    }
    assert "secret" not in raw.lower()


def test_live_web_build_requires_complete_public_auth_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires API and Cognito"):
        build(tmp_path / "site", mode="live", api_base_url="https://api.example")


def test_live_browser_uses_pkce_and_owner_api_without_secrets() -> None:
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert "code_challenge_method: \"S256\"" in script
    assert "/oauth2/token" in script
    assert 'apiFetch("/uploads"' in script
    assert 'apiFetch("/jobs?limit=50")' in script
    assert "/artifacts`" in script
    assert "client_secret" not in script
