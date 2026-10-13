# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Tool interface and the executor that runs tool calls for agents.

Tools run on a small pool of daemon threads so a timeout can abandon a stuck
tool without blocking exit. Python can't kill a running thread, so the pool
and its queue are bounded: a pile of stuck tools gets a clear error instead
of an ever-growing number of threads.

Tools whose spec sets `requires_confirmation` (the shell) only run once the
user approves them through the callback set with `set_confirm()`. The REPL
sets one; without it, as in the A2A servers, they are refused unless
ARES_SHELL_AUTO=1.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import json
import os
import queue
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ares.runtime.events import EventBus, EventType
from ares.runtime.types import ToolCall, ToolResult

DEFAULT_TIMEOUT = 30.0
_MAX_WORKERS = 8
_MAX_PENDING = 8
_STOP = object()


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema object
    category: str = ""
    # None waits as long as the tool takes (agent handoffs).
    timeout_seconds: float | None = DEFAULT_TIMEOUT
    requires_confirmation: bool = False


# A valid JSON escape, or a lone backslash that doesn't start one (the \. in "*\.py").
_ESCAPE = re.compile(r'\\["\\/bfnrtu]|\\')


def loads_lenient(text: str) -> Any:
    """json.loads, retried once with stray backslashes escaped.

    Small models often write shell commands like {"command": "grep '\\.py$'"},
    which strict JSON rejects. Read the stray backslash as a literal one.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = _ESCAPE.sub(lambda m: m.group() if len(m.group()) == 2 else "\\\\", text)
        if fixed == text:
            raise
        return json.loads(fixed)


class BaseTool(ABC):
    tool_id: str

    @property
    @abstractmethod
    def spec(self) -> ToolSpec: ...

    @abstractmethod
    def execute(self, **params: Any) -> ToolResult: ...

    def describe_call(self, params: dict[str, Any]) -> str:
        """How a call is shown to the user when it needs their approval."""
        return f"{self.spec.name} {json.dumps(params, ensure_ascii=False)}"

    def to_openai_function(self) -> dict[str, Any]:
        s = self.spec
        return {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}


def describe_tools(tools: list[BaseTool]) -> str:
    """Tool list for the system prompt of agents that call tools in text."""
    if not tools:
        return "No tools available."
    sections = []
    for t in tools:
        s = t.spec
        lines = [f"### {s.name}", s.description]
        if s.category:
            lines.append(f"Category: {s.category}")
        props = s.parameters.get("properties", {})
        required = set(s.parameters.get("required", []))
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                req = ", required" if pname in required else ""
                desc = pinfo.get("description", "")
                lines.append(f"  - {pname} ({pinfo.get('type', 'any')}{req})" + (f": {desc}" if desc else ""))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ── approval ──────────────────────────────────────────────────────────────────

_confirm: Callable[[str], bool] | None = None


def set_confirm(callback: Callable[[str], bool] | None) -> None:
    """Set how tools that need approval ask the user. None means refuse."""
    global _confirm
    _confirm = callback


def _auto_approve() -> bool:
    return os.environ.get("ARES_SHELL_AUTO", "").lower() in ("1", "true", "yes")


# ── bounded runner ────────────────────────────────────────────────────────────

class _Runner:
    def __init__(self, workers: int, pending: int) -> None:
        self._workers = workers
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=pending)
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, fn: Callable[..., ToolResult], params: dict[str, Any]) -> concurrent.futures.Future | None:
        with self._lock:
            if self._closed:
                return None
            if not self._threads:
                for i in range(self._workers):
                    t = threading.Thread(target=self._work, name=f"ares-tool-{i}", daemon=True)
                    t.start()
                    self._threads.append(t)
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            self._queue.put_nowait((future, fn, params))
        except queue.Full:
            return None
        return future

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            future, fn, params = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(**params))
            except BaseException as exc:
                future.set_exception(exc)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for _ in self._threads:
                try:
                    self._queue.put_nowait(_STOP)
                except queue.Full:
                    break


_RUNNER = _Runner(_MAX_WORKERS, _MAX_PENDING)
atexit.register(_RUNNER.shutdown)


# ── executor ──────────────────────────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, tools: list[BaseTool], bus: EventBus | None = None, agent_id: str = "") -> None:
        self._tools = {t.spec.name: t for t in tools}
        self._bus = bus
        self.agent_id = agent_id

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_function() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(self._tools) or "none"
            return _fail(call.name, f"Unknown tool: {call.name}. Available tools: {known}.")
        try:
            params = loads_lenient(call.arguments) if call.arguments else {}
        except json.JSONDecodeError as exc:
            return _fail(call.name, f"Invalid arguments JSON: {exc}")
        if not isinstance(params, dict):
            return _fail(call.name, f"Invalid arguments: expected a JSON object, got {type(params).__name__}.")

        spec = tool.spec
        if spec.requires_confirmation and not _auto_approve():
            if _confirm is None:
                return _fail(call.name, f"'{call.name}' needs the user's approval, and nobody is here to give it. "
                                        "It runs from the Ares REPL, or on a server started with ARES_SHELL_AUTO=1.")
            if not _confirm(f"{self.agent_id or 'agent'} wants to run: {tool.describe_call(params)}"):
                return _fail(call.name, f"The user declined this {call.name} call. Don't retry it; "
                                        "carry on without it or ask the user what to do instead.")

        call_key = uuid.uuid4().hex
        self._publish(EventType.TOOL_CALL_START, {"tool": call.name, "arguments": params,
                                                  "agent": self.agent_id, "call_key": call_key})
        t0 = time.time()
        future = _RUNNER.submit(tool.execute, params)
        if future is None:
            result = _fail(call.name, "Too many tools are still running (some may be stuck). Try again later.")
        else:
            try:
                result = future.result(timeout=spec.timeout_seconds)
            except concurrent.futures.TimeoutError:
                future.cancel()
                self._publish(EventType.TOOL_TIMEOUT, {"tool": call.name, "timeout": spec.timeout_seconds})
                result = _fail(call.name, f"Tool '{call.name}' timed out after {spec.timeout_seconds:.0f}s.")
            except Exception as exc:
                result = _fail(call.name, f"Tool execution error: {exc}")
        result.latency_seconds = time.time() - t0
        self._publish(EventType.TOOL_CALL_END, {
            "tool": call.name,
            "success": result.success,
            "latency": result.latency_seconds,
            "result": (result.content or "")[:10240],
            "agent": self.agent_id,
            "call_key": call_key,
        })
        return result

    def _publish(self, event: EventType, data: dict[str, Any]) -> None:
        if self._bus is not None:
            self._bus.publish(event, data)


def _fail(tool_name: str, content: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, content=content, success=False)
