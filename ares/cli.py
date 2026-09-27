#!/usr/bin/env python3
"""Ares — terminal REPL entry point (installed as the `ares` command)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ares is launched from whatever project you're working in (that's what
# /serena attaches to), so its settings can't come from the cwd's .env.
ENV_FILES = [
    Path.home() / ".ares" / ".env",
    Path(__file__).resolve().parent.parent / ".env",  # source checkout
]


def _load_env(files: list[Path]) -> None:
    """Load KEY=VALUE lines into os.environ. Real env vars and earlier files win."""
    for env_file in files:
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# Before the imports below, which read their settings at import time.
_load_env(ENV_FILES)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

from ares import memory
from ares.live import LiveView, already_shown
from ares.system import AresSystem

BANNER = """
Ares — local multi-agent AI. Type /help for commands, Ctrl-C to cancel, Ctrl-D to quit.
"""

HELP = """
  <prompt>           let the orchestrator route it
  /coder <task>      code generation, debugging, file edits
  /thinker <task>    deeper reasoning and planning
  /runner <task>     quick shell and file tasks
  /serena <task>     symbol-aware navigation and refactors (current project)
  /remember <fact>   store a fact in long-term memory
  /memories          list what Ares remembers
  /forget            delete all memories (asks first)
  /learn             run the learning cycle now
  /clear             forget this conversation (long-term memory is kept)
  /help, /quit
"""

AGENTS = {
    "/coder":   ("coder_run", "coder"),
    "/thinker": ("thinker_run", "thinker"),
    "/runner":  ("runner_run", "runner"),
    "/serena":  ("serena_run", "serena"),
}
COMMANDS = [*AGENTS, "/remember", "/memories", "/forget", "/learn", "/clear", "/help", "/quit"]
HISTORY_FILE = Path.home() / ".ares" / "history"


def _setup_readline() -> None:
    try:
        import readline
    except ImportError:
        return
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)

    def complete(text: str, state: int) -> str | None:
        matches = [c for c in COMMANDS if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" ")
    readline.parse_and_bind("tab: complete")

    import atexit

    def save() -> None:
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass

    atexit.register(save)


def _ask(system: AresSystem, method: str, agent: str, prompt: str, history: list[dict]) -> str | None:
    """Run one prompt with live output. Returns None if the user cancelled."""
    view = LiveView(system.bus, top_agent=agent)
    try:
        with view:
            response = getattr(system, method)(prompt, history=history)
    except KeyboardInterrupt:
        print("[cancelled]\n")
        return None
    if not already_shown(view.answer, response):
        print(response)
    print()
    return response


def _handle_command(system: AresSystem, raw: str, history: list[dict]) -> bool:
    """Handle a non-agent slash command. Returns False if it isn't one."""
    cmd, _, arg = raw.partition(" ")
    arg = arg.strip()
    if cmd == "/help":
        print(HELP)
    elif cmd == "/learn":
        print("Running learning cycle...")
        print(f"Done: {system.learn_now()}\n")
    elif cmd == "/remember":
        if not arg:
            print("Usage: /remember <fact>\n")
        else:
            print("Remembered.\n" if memory.remember(arg) else "[memory off]\n")
    elif cmd == "/memories":
        items = memory.list_all()
        if not items:
            print("No memories yet (or memory is off).")
        for m in items:
            print(f"  - {m.get('memory')}")
        print()
    elif cmd == "/forget":
        if input("Delete ALL memories? [y/N] ").strip().lower() == "y":
            memory.forget_all()
            print("Forgotten.\n")
    elif cmd == "/clear":
        history.clear()
        print("Conversation cleared.\n")
    else:
        return False
    return True


def main() -> None:
    print(BANNER)
    print("Initialising...", flush=True)

    system = AresSystem()
    print("Ready.\n")

    import atexit, signal
    atexit.register(system.shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    _setup_readline()

    history: list[dict] = []

    while True:
        try:
            raw = input("ares> ").strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print("\nBye.")
            break

        if not raw:
            continue
        if raw in ("/quit", "/exit", "exit", "quit"):
            print("Bye.")
            break

        cmd, _, prompt = raw.partition(" ")
        prompt = prompt.strip()
        if cmd in AGENTS:
            if not prompt:
                print(f"Usage: {cmd} <task>\n")
                continue
            method, agent = AGENTS[cmd]
        elif raw.startswith("/"):
            if not _handle_command(system, raw, history):
                print(f"Unknown command {cmd}. Try /help.\n")
            continue
        else:
            method, agent, prompt = "run", "orchestrator", raw

        try:
            response = _ask(system, method, agent, prompt, history)
        except Exception as exc:
            print(f"\n[error] {exc}\n")
            continue
        if response is not None:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
