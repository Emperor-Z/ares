# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Stop agents from calling tools in circles, and keep long runs in context.

Agents reset the guard at the start of every run, so its counts cover one
request, not the whole session.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass

from ares.runtime.events import EventBus, EventType
from ares.runtime.types import Message, Role


@dataclass(slots=True)
class LoopGuardConfig:
    max_identical_calls: int = 3   # same tool with the same arguments
    ping_pong_window: int = 6      # A-B-A-B or A-B-C-A-B-C cycles
    poll_tool_budget: int = 5      # calls to any one tool per run
    max_context_messages: int = 100
    warn_before_block: bool = True  # let each distinct loop through once, then block it


@dataclass(slots=True)
class LoopVerdict:
    blocked: bool = False
    reason: str = ""
    warned: bool = False
    kind: str = ""  # what was detected, and for which tool


class LoopGuard:
    def __init__(self, config: LoopGuardConfig, *, bus: EventBus | None = None) -> None:
        self._config = config
        self._bus = bus
        self.reset()

    def reset(self) -> None:
        self._call_counts: dict[str, int] = {}
        self._per_tool: dict[str, int] = {}
        self._sequence: deque[str] = deque(maxlen=self._config.ping_pong_window * 2)
        self._warned: set[str] = set()

    def check_call(self, tool_name: str, arguments: str) -> LoopVerdict:
        verdict = self._check(tool_name, arguments)
        # Keyed by kind, not reason: the reason text includes a running count.
        if verdict.blocked and self._config.warn_before_block and verdict.kind not in self._warned:
            self._warned.add(verdict.kind)
            return LoopVerdict(warned=True, reason=verdict.reason, kind=verdict.kind)
        return verdict

    def _check(self, tool_name: str, arguments: str) -> LoopVerdict:
        key = hashlib.sha256(f"{tool_name}:{arguments}".encode()).hexdigest()[:16]
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        if self._call_counts[key] > self._config.max_identical_calls:
            return self._block("identical_call", tool_name,
                               f"Identical call to '{tool_name}' repeated {self._call_counts[key]} times "
                               f"(max {self._config.max_identical_calls}).")

        self._per_tool[tool_name] = self._per_tool.get(tool_name, 0) + 1
        if self._per_tool[tool_name] > self._config.poll_tool_budget:
            return self._block("poll_budget", tool_name,
                               f"Tool '{tool_name}' exceeded its budget of {self._config.poll_tool_budget} calls.")

        self._sequence.append(tool_name)
        if len(self._sequence) >= self._config.ping_pong_window and self._ping_pong():
            return self._block("ping_pong", tool_name, "Repetitive tool-calling pattern detected (ping-pong).")
        return LoopVerdict()

    def _ping_pong(self) -> bool:
        seq = list(self._sequence)
        for period in (2, 3):
            if len(seq) >= period * 2:
                tail = seq[-period * 2:]
                if all(tail[i] == tail[i % period] for i in range(len(tail))):
                    return True
        return False

    def _block(self, reason_type: str, tool_name: str, reason: str) -> LoopVerdict:
        if self._bus is not None:
            self._bus.publish(EventType.LOOP_GUARD_TRIGGERED, {"reason_type": reason_type, "tool": tool_name})
        return LoopVerdict(blocked=True, reason=reason, kind=f"{reason_type}:{tool_name}")

    def compress_context(self, messages: list[Message]) -> list[Message]:
        """Shrink an overlong conversation in stages, least destructive first."""
        limit = self._config.max_context_messages
        if len(messages) <= limit:
            return messages

        # 1. Blank out tool results in the older half.
        half = len(messages) // 2
        out = [
            Message(role=Role.TOOL, content="[Tool result truncated]", tool_call_id=m.tool_call_id, name=m.name)
            if i < half and m.role == Role.TOOL else m
            for i, m in enumerate(messages)
        ]
        if len(out) <= limit:
            return out

        # 2. Keep the system prompt and the most recent messages.
        system = [m for m in out if m.role == Role.SYSTEM]
        rest = [m for m in out if m.role != Role.SYSTEM]
        return system + rest[-(limit - len(system)):]
