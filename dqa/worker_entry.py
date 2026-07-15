"""Signal-aware entry point for the one-shot audit worker container."""

from __future__ import annotations

import signal
import sys
from types import FrameType

from .cli import main as cli_main


def _terminate(signum: int, _frame: FrameType | None) -> None:
    """Stop through normal Python unwinding so atomic writers can clean up."""
    name = signal.Signals(signum).name
    print(f"worker received {name}; terminating", file=sys.stderr, flush=True)
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    if len(sys.argv) > 1 and sys.argv[1] == "hosted":
        from .aws.worker import main as hosted_main

        return hosted_main(sys.argv[2:])
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
