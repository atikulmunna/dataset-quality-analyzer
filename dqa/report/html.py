from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _escape(value: object) -> str:
    return html.escape(str(value))


def render_html(summary: dict[str, Any], flags: dict[str, Any]) -> str:
    checks_rows = []
    for name, check in summary.get("checks", {}).items():
        counts = check.get("counts", {})
        checks_rows.append(
            "<tr>"
            f"<td>{_escape(name)}</td>"
            f"<td>{_escape(check.get('status', 'unknown'))}</td>"
            f"<td>{_escape(counts.get('critical', 0))}</td>"
            f"<td>{_escape(counts.get('high', 0))}</td>"
            f"<td>{_escape(counts.get('medium', 0))}</td>"
            f"<td>{_escape(counts.get('low', 0))}</td>"
            "</tr>"
        )

    finding_rows = []
    for finding in flags.get("findings", []):
        finding_rows.append(
            "<tr>"
            f"<td>{_escape(finding.get('id', ''))}</td>"
            f"<td>{_escape(finding.get('severity', ''))}</td>"
            f"<td>{_escape(finding.get('split', ''))}</td>"
            f"<td>{_escape(finding.get('image', finding.get('label', '')))}</td>"
            f"<td>{_escape(finding.get('message', ''))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>DQA Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #202124; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .meta {{ margin: 12px 0 20px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f7f7f7; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; background: #eceff1; }}
  </style>
</head>
<body>
  <h1>Dataset Quality Analyzer Report</h1>
  <div class=\"meta\">
    <div><strong>Run ID:</strong> {_escape(summary.get('run', {}).get('run_id', ''))}</div>
    <div><strong>Data YAML:</strong> {_escape(summary.get('dataset', {}).get('data_yaml', ''))}</div>
    <div><strong>Root:</strong> {_escape(summary.get('dataset', {}).get('root', ''))}</div>
    <div><strong>Total Findings:</strong> <span class=\"badge\">{_escape(summary.get('totals', {}).get('findings', 0))}</span></div>
    <div><strong>Build Failed:</strong> <span class=\"badge\">{_escape(summary.get('totals', {}).get('build_failed', False))}</span></div>
  </div>

  <h2>Check Summary</h2>
  <table>
    <thead>
      <tr><th>Check</th><th>Status</th><th>Critical</th><th>High</th><th>Medium</th><th>Low</th></tr>
    </thead>
    <tbody>
      {''.join(checks_rows)}
    </tbody>
  </table>

  <h2>Findings</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Severity</th><th>Split</th><th>Path</th><th>Message</th></tr>
    </thead>
    <tbody>
      {''.join(finding_rows) if finding_rows else '<tr><td colspan="5">No findings</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""


def write_html(path: Path, summary: dict[str, Any], flags: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(summary, flags), encoding="utf-8")
