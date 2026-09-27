"""Live REPL view: shows which agent is working and streams the answer.

Sub-agents run on the runtime's tool thread pool, so anything happening on
the REPL's own thread belongs to the agent the user addressed. Its tokens
are the answer; everything else becomes a one-line status.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from typing import Any, TextIO

from openjarvis.core.events import EventBus, EventType

from ares import engine

# NativeReAct agents write Thought/Action lines before this marker.
_REACT_AGENTS = {"coder", "runner", "serena"}
_FINAL = re.compile(r"final answer:\s*", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def visible_answer(agent: str, text: str) -> str:
    """The part of a partial reply that is safe to show as the answer."""
    if agent in _REACT_AGENTS:
        m = _FINAL.search(text)
        return text[m.end():] if m else ""
    text = _THINK.sub("", text)
    # Hold back a half-received "<think>" so it never flashes on screen.
    for i in range(1, len("<think>")):
        if text.endswith("<think>"[:i]):
            return text[:-i]
    return text.lstrip()


class LiveView:
    def __init__(self, bus: EventBus, top_agent: str, out: TextIO = sys.stdout) -> None:
        self._bus = bus
        self._out = out
        self._tty = out.isatty()
        self._main = threading.get_ident()
        self._lock = threading.Lock()
        self._labels: dict[int, str] = {self._main: top_agent}
        self._buf = ""
        self._shown = 0
        self.answer = ""  # answer text streamed by the latest top-level inference
        self._status = ""
        self._started = time.monotonic()
        self._handlers = [
            (EventType.AGENT_TURN_START, self._on_turn_start),
            (EventType.INFERENCE_START, self._on_inference_start),
            (EventType.TOOL_CALL_START, self._on_tool),
        ]

    def __enter__(self) -> "LiveView":
        for event, handler in self._handlers:
            self._bus.subscribe(event, handler)
        engine.set_token_sink(self._on_token)
        self._set_status(f"{self._labels[self._main]} · starting")
        return self

    def __exit__(self, *exc: Any) -> None:
        engine.set_token_sink(None)
        for event, handler in self._handlers:
            self._bus.unsubscribe(event, handler)
        with self._lock:
            self._clear_status()
            if self._shown:
                self._out.write("\n")
            self._out.flush()

    # ── bus events ─────────────────────────────────────────────────────────

    def _label(self) -> str:
        return self._labels.get(threading.get_ident(), "agent")

    def _on_turn_start(self, event: Any) -> None:
        tid = threading.get_ident()
        if tid != self._main:
            self._labels[tid] = event.data.get("agent") or "agent"
            self._set_status(f"{self._labels[self._main]} → {self._labels[tid]}")

    def _on_inference_start(self, event: Any) -> None:
        if threading.get_ident() == self._main:
            with self._lock:
                if self._shown:
                    self._out.write("\n\n")
                self._buf, self._shown, self.answer = "", 0, ""
        self._set_status(f"{self._label()} · thinking")

    def _on_tool(self, event: Any) -> None:
        tool = event.data.get("tool", "tool")
        if tool.startswith("call_"):
            return  # handoffs show up as "a → b" via the sub-agent's turn start
        self._set_status(f"{self._label()} · {tool}")

    # ── tokens ─────────────────────────────────────────────────────────────

    def _on_token(self, text: str) -> None:
        if threading.get_ident() != self._main:
            self._set_status(f"{self._label()} · writing", tick=True)
            return
        with self._lock:
            self._buf += text
            answer = visible_answer(self._labels[self._main], self._buf)
            new = answer[self._shown:]
            if new:
                self._clear_status()
                self._out.write(new)
                self._out.flush()
                self._shown = len(answer)
                self.answer = answer
        if not new:
            self._set_status(f"{self._label()} · thinking", tick=True)

    # ── status line ────────────────────────────────────────────────────────

    def _set_status(self, text: str, tick: bool = False) -> None:
        with self._lock:
            if self._shown:
                return  # the answer is on screen; don't draw over it
            if not self._tty:
                if text != self._status:
                    self._out.write(f"  · {text}\n")
                    self._status = text
                return
            elapsed = time.monotonic() - self._started
            spin = _SPINNER[int(elapsed * 10) % len(_SPINNER)]
            self._out.write(f"\r\033[2K\033[2m{spin} {text}  {elapsed:.0f}s\033[0m")
            self._out.flush()
            self._status = text

    def _clear_status(self) -> None:
        if self._tty and self._status:
            self._out.write("\r\033[2K")
        self._status = ""


def already_shown(streamed: str, final: str) -> bool:
    """True if the streamed text is the final answer, give or take whitespace."""
    norm = lambda s: " ".join(s.split())
    return bool(streamed.strip()) and norm(streamed) == norm(final)
