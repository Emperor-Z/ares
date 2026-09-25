#!/usr/bin/env python3
"""Ares — terminal REPL entry point (installed as the `ares` command)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Load .env before anything else so env vars are available to all modules.
_env_file = Path.cwd() / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

from ares.system import AresSystem

BANNER = """
╔══════════════════════════════════════╗
║  Ares  —  local AI  —  terminal     ║
║  /coder  /thinker  /runner          ║
║  /serena /learn    /quit            ║
╚══════════════════════════════════════╝
"""

COMMANDS = {
    "/coder":   "coder_run",
    "/thinker": "thinker_run",
    "/runner":  "runner_run",
    "/serena":  "serena_run",
}


def main() -> None:
    print(BANNER)
    print("Initialising...", flush=True)

    system = AresSystem()
    print("Ready.\n")

    history: list[dict] = []

    def _shutdown() -> None:
        system.shutdown()

    import atexit, signal
    atexit.register(_shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    while True:
        try:
            raw = input("ares> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw in ("/quit", "/exit", "exit", "quit"):
            print("Bye.")
            break

        if raw == "/learn":
            print("Running rlm learning cycle...")
            summary = system.learn_now()
            print(f"Done: {summary}\n")
            continue

        direct_method = None
        prompt = raw
        for cmd, method in COMMANDS.items():
            if raw.startswith(cmd + " "):
                direct_method = method
                prompt = raw[len(cmd):].strip()
                break

        try:
            if direct_method:
                response = getattr(system, direct_method)(prompt, history=history)
            else:
                response = system.run(prompt, history=history)

            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})

            print(f"\n{response}\n")
        except Exception as exc:
            print(f"\n[error] {exc}\n")


if __name__ == "__main__":
    main()
