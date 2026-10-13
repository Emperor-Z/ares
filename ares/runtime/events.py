# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Thread-safe publish/subscribe bus that agents, tools and tracing share.

Subscribers run synchronously on the publishing thread, in the order they
subscribed. A failing subscriber is logged and skipped so an observer can
never break the agent that published.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    INFERENCE_START = "inference_start"
    INFERENCE_END = "inference_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_TIMEOUT = "tool_timeout"
    AGENT_TURN_START = "agent_turn_start"
    AGENT_TURN_END = "agent_turn_end"
    LOOP_GUARD_TRIGGERED = "loop_guard_triggered"
    TRACE_COMPLETE = "trace_complete"


@dataclass(slots=True)
class Event:
    event_type: EventType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Subscriber]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: EventType, callback: Subscriber) -> None:
        with self._lock:
            listeners = self._subscribers.get(event_type, [])
            if callback in listeners:
                listeners.remove(callback)

    def publish(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        event = Event(event_type=event_type, timestamp=time.time(), data=data or {})
        with self._lock:
            listeners = list(self._subscribers.get(event_type, []))
        for callback in listeners:
            try:
                callback(event)
            except Exception:
                logger.exception("Subscriber %r failed on %s", callback, event_type.value)
        return event
