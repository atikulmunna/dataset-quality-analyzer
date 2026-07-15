"""Low-cardinality structured events shared by hosted AWS runtimes."""

from __future__ import annotations

from datetime import datetime, timezone
import json


def emit_event(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        **fields,
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
