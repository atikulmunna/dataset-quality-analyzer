from __future__ import annotations

import argparse
import html
import os
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent


class DashboardError(ValueError):
    """Raised when form input is invalid."""


def _normalize_input(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1].strip()
    return value


def _first(form: dict[str, list[str]], key: str, default: str = "") -> str:
    values = form.get(key, [])
    if not values:
        return default
    return _normalize_input(values[0])


def _has(form: dict[str, list[str]], key: str) -> bool:
    return key in form


def _require(value: str, label: str) -> str:
    if not value:
        raise DashboardError(f"{label} is required.")
    return value


def _render_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def _run_command(cmd: list[str], extra_env: dict[str, str] | None = None) -> dict[str, object]:
    started = time.time()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        env=env,
    )

    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "duration_sec": round(time.time() - started, 3),
        "command": _render_command(cmd),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _build_audit(form: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    data = _first(form, "audit_data")
    data_url = _first(form, "audit_data_url")
    if not data and not data_url:
        raise DashboardError("Provide either Data Path or Data URL for audit.")
    if data and data_url:
        raise DashboardError("Use either Data Path or Data URL, not both.")

    out_dir = _require(_first(form, "audit_out", "runs/web_audit"), "Output Directory")
    config = _first(form, "audit_config", "dqa.yaml")

    cmd = ["python", "-m", "dqa", "audit"]
    if data:
        cmd += ["--data", data]
    else:
        cmd += ["--data-url", data_url]
        data_url_format = _first(form, "audit_data_url_format", "yolov11")
        if data_url_format:
            cmd += ["--data-url-format", data_url_format]

    cmd += ["--out", out_dir, "--config", config]

    splits = _first(form, "audit_splits")
    if splits:
        cmd += ["--splits", splits]

    max_images = _first(form, "audit_max_images")
    if max_images:
        cmd += ["--max-images", max_images]

    formats = _first(form, "audit_formats", "html,json")
    if formats:
        cmd += ["--format", formats]

    fail_on = _first(form, "audit_fail_on")
    if fail_on:
        cmd += ["--fail-on", fail_on]

    if _has(form, "audit_near_dup"):
        cmd.append("--near-dup")

    if _has(form, "audit_no_remote_cache"):
        cmd.append("--no-remote-cache")

    ttl_hours = _first(form, "audit_remote_cache_ttl")
    if ttl_hours:
        cmd += ["--remote-cache-ttl-hours", ttl_hours]

    extra_env: dict[str, str] = {}
    api_key = _first(form, "audit_roboflow_api_key")
    if api_key:
        extra_env["ROBOFLOW_API_KEY"] = api_key

    return cmd, extra_env


def _build_explain(form: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    run = _first(form, "explain_run")
    summary = _first(form, "explain_summary")
    flags = _first(form, "explain_flags")

    cmd = ["python", "-m", "dqa", "explain"]
    if run:
        cmd += ["--run", run]
    else:
        if not summary or not flags:
            raise DashboardError("Explain requires Run Directory, or both Summary and Flags paths.")
        cmd += ["--summary", summary, "--flags", flags]

    explain_format = _first(form, "explain_format", "text")
    cmd += ["--format", explain_format]

    out_file = _first(form, "explain_out_file")
    if out_file:
        cmd += ["--out-file", out_file]

    return cmd, {}


def _build_validate(form: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    artifact = _require(_first(form, "validate_artifact"), "Artifact Path")
    schema = _require(_first(form, "validate_schema", "schemas/summary.schema.json"), "Schema Path")
    cmd = ["python", "-m", "dqa", "validate", "--artifact", artifact, "--schema", schema]
    return cmd, {}


def _build_diff(form: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    old = _require(_first(form, "diff_old"), "Old Run Directory")
    new = _require(_first(form, "diff_new"), "New Run Directory")
    cmd = ["python", "-m", "dqa", "diff", "--old", old, "--new", new]

    fail_on_regression = _first(form, "diff_fail_on_regression")
    if fail_on_regression:
        cmd += ["--fail-on-regression", fail_on_regression]

    return cmd, {}


def _input(name: str, value: str = "", placeholder: str = "") -> str:
    return (
        f'<input type="text" name="{html.escape(name)}" '
        f'value="{html.escape(value)}" placeholder="{html.escape(placeholder)}" />'
    )


def _select(name: str, options: list[tuple[str, str]], current: str = "") -> str:
    rows: list[str] = [f'<select name="{html.escape(name)}">']
    for value, label in options:
        selected = " selected" if value == current else ""
        rows.append(
            f'<option value="{html.escape(value)}"{selected}>{html.escape(label)}</option>'
        )
    rows.append("</select>")
    return "\n".join(rows)


def _checkbox(name: str, checked: bool = False, label: str = "") -> str:
    mark = " checked" if checked else ""
    return (
        f'<label class="check"><input type="checkbox" name="{html.escape(name)}"{mark} /> '
        f"{html.escape(label)}</label>"
    )


def _render_page(result: dict[str, object] | None = None, form: dict[str, list[str]] | None = None) -> str:
    form = form or {}

    def val(key: str, default: str = "") -> str:
        return _first(form, key, default)

    result_html = ""
    if result is not None:
        status = "Success" if result["ok"] else "Failed"
        result_html = f"""
<section class=\"card result {'ok' if result['ok'] else 'fail'}\">
  <h2>Last Run: {status}</h2>
  <p><strong>Command:</strong> <code>{html.escape(str(result['command']))}</code></p>
  <p><strong>Exit Code:</strong> {result['exit_code']} | <strong>Duration:</strong> {result['duration_sec']}s</p>
  <h3>STDOUT</h3>
  <pre>{html.escape(str(result['stdout']))}</pre>
  <h3>STDERR</h3>
  <pre>{html.escape(str(result['stderr']))}</pre>
</section>
"""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>DQA Web Dashboard</title>
  <style>
    :root {{
      --bg: #f7f6f2;
      --surface: #fffefb;
      --ink: #1c1d22;
      --muted: #5e6472;
      --line: #d7d4cc;
      --accent: #0b6e4f;
      --warn: #9b2226;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Tahoma, sans-serif; color: var(--ink); background:
      radial-gradient(circle at 0% 0%, #ece7de 0%, transparent 45%),
      radial-gradient(circle at 100% 100%, #e6efe8 0%, transparent 45%), var(--bg); }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px 20px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p.lead {{ margin-top: 0; color: var(--muted); }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 14px; box-shadow: 0 2px 0 rgba(0,0,0,0.02); }}
    .result.ok {{ border-color: #94d2bd; }}
    .result.fail {{ border-color: #e5989b; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    h3 {{ margin: 12px 0 8px; font-size: 14px; color: var(--muted); }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .check {{ display: block; margin: 8px 0; color: var(--ink); }}
    input, select, button {{ width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    .row2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .row3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }}
    button {{ margin-top: 10px; border: none; background: var(--accent); color: #fff; font-weight: 600; cursor: pointer; }}
    button:hover {{ filter: brightness(0.94); }}
    code {{ background: #f0ede6; padding: 2px 6px; border-radius: 6px; }}
    pre {{ overflow: auto; max-height: 220px; background: #111827; color: #f3f4f6; padding: 12px; border-radius: 10px; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    .hint {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>DQA Web Dashboard</h1>
    <p class=\"lead\">Run dataset quality checks without terminal commands. All actions execute locally in this repository.</p>

    <section class=\"card\">
      <h2>Quick Instructions</h2>
      <ul>
        <li>Use <strong>Audit</strong> to generate run outputs under <code>runs/...</code>.</li>
        <li>Use <strong>Explain</strong> for human-readable summary or JSON/Markdown export.</li>
        <li>Use <strong>Validate</strong> to enforce schema contracts in CI-like checks.</li>
        <li>Use <strong>Diff</strong> to compare old vs new run regressions.</li>
      </ul>
    </section>

    {result_html}

    <div class=\"grid\">
      <form method=\"post\" action=\"/run/audit\" class=\"card\">
        <h2>1) Audit Dataset</h2>
        <label>Data Path (local)</label>
        {_input('audit_data', val('audit_data'), 'C:\\path\\to\\data.yaml or COCO folder/json')}
        <label>or Data URL (remote)</label>
        {_input('audit_data_url', val('audit_data_url'), 'https://app.roboflow.com/workspace/project/1')}
        <div class=\"row2\">
          <div>
            <label>Output Directory</label>
            {_input('audit_out', val('audit_out', 'runs/web_audit'), 'runs/web_audit')}
          </div>
          <div>
            <label>Config</label>
            {_input('audit_config', val('audit_config', 'dqa.yaml'), 'dqa.yaml')}
          </div>
        </div>
        <div class=\"row2\">
          <div>
            <label>Fail On</label>
            {_select('audit_fail_on', [('', 'Use config default'), ('critical', 'critical'), ('high', 'high'), ('medium', 'medium'), ('low', 'low')], val('audit_fail_on'))}
          </div>
          <div>
            <label>Format</label>
            {_input('audit_formats', val('audit_formats', 'html,json'), 'html,json')}
          </div>
        </div>
        <div class=\"row3\">
          <div>
            <label>Splits</label>
            {_input('audit_splits', val('audit_splits', 'train,val,test'), 'train,val,test')}
          </div>
          <div>
            <label>Max Images</label>
            {_input('audit_max_images', val('audit_max_images'), '0 = all')}
          </div>
          <div>
            <label>Data URL Format</label>
            {_input('audit_data_url_format', val('audit_data_url_format', 'yolov11'), 'yolov11')}
          </div>
        </div>
        <label>Roboflow API Key (optional)</label>
        {_input('audit_roboflow_api_key', val('audit_roboflow_api_key'), 'ROBOFLOW_API_KEY override')}
        {_checkbox('audit_near_dup', _has(form, 'audit_near_dup'), 'Enable near-duplicate check')}
        {_checkbox('audit_no_remote_cache', _has(form, 'audit_no_remote_cache'), 'Disable remote cache')}
        <label>Remote cache TTL hours (optional)</label>
        {_input('audit_remote_cache_ttl', val('audit_remote_cache_ttl'), '24')}
        <button type=\"submit\">Run Audit</button>
      </form>

      <form method=\"post\" action=\"/run/explain\" class=\"card\">
        <h2>2) Explain Findings</h2>
        <label>Run Directory</label>
        {_input('explain_run', val('explain_run', 'runs/web_audit'), 'runs/web_audit')}
        <p class=\"hint\">Or use explicit summary + flags files:</p>
        <label>Summary Path</label>
        {_input('explain_summary', val('explain_summary'), 'runs/x/summary.json')}
        <label>Flags Path</label>
        {_input('explain_flags', val('explain_flags'), 'runs/x/flags.json')}
        <div class=\"row2\">
          <div>
            <label>Format</label>
            {_select('explain_format', [('text', 'text'), ('markdown', 'markdown'), ('json', 'json')], val('explain_format', 'text'))}
          </div>
          <div>
            <label>Out File (optional)</label>
            {_input('explain_out_file', val('explain_out_file'), 'runs/x/explain.md or .json')}
          </div>
        </div>
        <button type=\"submit\">Run Explain</button>
      </form>

      <form method=\"post\" action=\"/run/validate\" class=\"card\">
        <h2>3) Validate Artifact</h2>
        <label>Artifact Path</label>
        {_input('validate_artifact', val('validate_artifact', 'runs/web_audit/summary.json'), 'runs/x/summary.json')}
        <label>Schema Path</label>
        {_input('validate_schema', val('validate_schema', 'schemas/summary.schema.json'), 'schemas/summary.schema.json')}
        <button type=\"submit\">Run Validate</button>
      </form>

      <form method=\"post\" action=\"/run/diff\" class=\"card\">
        <h2>4) Diff Runs</h2>
        <label>Old Run Directory</label>
        {_input('diff_old', val('diff_old'), 'runs/old_run')}
        <label>New Run Directory</label>
        {_input('diff_new', val('diff_new'), 'runs/new_run')}
        <label>Fail On Regression (optional)</label>
        {_select('diff_fail_on_regression', [('', 'No gate'), ('critical', 'critical'), ('high', 'high'), ('medium', 'medium'), ('low', 'low')], val('diff_fail_on_regression'))}
        <button type=\"submit\">Run Diff</button>
      </form>
    </div>
  </div>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    routes: dict[str, Callable[[dict[str, list[str]]], tuple[list[str], dict[str, str]]]] = {
        "/run/audit": _build_audit,
        "/run/explain": _build_explain,
        "/run/validate": _build_validate,
        "/run/diff": _build_diff,
    }

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self._send_html("<h1>Not Found</h1>", HTTPStatus.NOT_FOUND)
            return
        self._send_html(_render_page())

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        builder = self.routes.get(path)
        if builder is None:
            self._send_html("<h1>Not Found</h1>", HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(payload, keep_blank_values=True)

        try:
            cmd, env = builder(form)
            result = _run_command(cmd, env)
        except DashboardError as exc:
            result = {
                "ok": False,
                "exit_code": 2,
                "duration_sec": 0.0,
                "command": "input validation",
                "stdout": "",
                "stderr": str(exc),
            }
        except OSError as exc:
            result = {
                "ok": False,
                "exit_code": 3,
                "duration_sec": 0.0,
                "command": "runtime error",
                "stdout": "",
                "stderr": str(exc),
            }

        self._send_html(_render_page(result=result, form=form))


def main() -> int:
    parser = argparse.ArgumentParser(description="DQA local web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"DQA dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

