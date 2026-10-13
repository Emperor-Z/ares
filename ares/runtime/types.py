# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Data types shared by the engine, agents, tools and tracing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON object, as text


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """OpenAI-style chat message, as Ollama's /api/chat accepts it."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content or ""}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    content: str
    success: bool = True
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    content: str
    tool_results: list[ToolResult] = field(default_factory=list)
    turns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class StepType(str, Enum):
    GENERATE = "generate"
    TOOL_CALL = "tool_call"
    RESPOND = "respond"


@dataclass(slots=True)
class TraceStep:
    step_type: StepType
    timestamp: float
    duration_seconds: float = 0.0
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _trace_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(slots=True)
class Trace:
    """Everything one agent run did: each inference, each tool call, the answer."""

    trace_id: str = field(default_factory=_trace_id)
    query: str = ""
    agent: str = ""
    model: str = ""
    engine: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    result: str = ""
    outcome: str | None = None
    feedback: float | None = None
    started_at: float = 0.0
    ended_at: float = 0.0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
