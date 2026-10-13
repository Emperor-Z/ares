# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""A2A protocol: agent cards and a JSON-RPC 2.0 task handler.

Handles tasks/send (runs the task synchronously), tasks/get and tasks/cancel.
Tasks are kept in memory, newest MAX_TASKS only.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

MAX_TASKS = 1000


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass(slots=True)
class AgentCard:
    """Served at /.well-known/agent.json."""

    name: str
    description: str = ""
    url: str = ""
    version: str = "0.1.0"
    capabilities: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    authentication: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class A2ATask:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    state: TaskState = TaskState.SUBMITTED
    input_text: str = ""
    output_text: str = ""
    history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.task_id, "state": self.state.value, "input": self.input_text,
                "output": self.output_text, "history": self.history, "metadata": {}}


class A2AServer:
    def __init__(self, agent_card: AgentCard, *, handler: Callable[[str], str]) -> None:
        self.agent_card = agent_card
        self._handler = handler
        self._tasks: OrderedDict[str, A2ATask] = OrderedDict()
        self._lock = threading.Lock()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id", "")
        method = request.get("method", "")
        params = request.get("params") or {}
        if method == "tasks/send":
            return _ok(req_id, self._send(params).to_dict())
        if method in ("tasks/get", "tasks/cancel"):
            with self._lock:
                task = self._tasks.get(params.get("id", ""))
                if task is None:
                    return _error(req_id, -32602, f"Task not found: {params.get('id', '')}")
                if method == "tasks/cancel":
                    task.state = TaskState.CANCELED
                return _ok(req_id, task.to_dict())
        return _error(req_id, -32601, f"Method not found: {method}")

    def _send(self, params: dict[str, Any]) -> A2ATask:
        parts = (params.get("message") or {}).get("parts") or [{}]
        text = parts[0].get("text", "") or params.get("input", "")
        task = A2ATask(input_text=text, state=TaskState.WORKING)
        with self._lock:
            self._tasks[task.task_id] = task
            while len(self._tasks) > MAX_TASKS:
                self._tasks.popitem(last=False)
        try:
            task.output_text = self._handler(text)
            task.state = TaskState.COMPLETED
            task.history.append({"role": "agent", "content": task.output_text})
        except Exception as exc:
            task.output_text = str(exc)
            task.state = TaskState.FAILED
        return task


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
