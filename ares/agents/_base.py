"""Shared setup for every Ares agent: engine, per-agent config, loop guard."""

from __future__ import annotations

from typing import Any

from ares.config import AGENT_DEFAULTS, LOOP_GUARD
from ares.engine import get_engine
from ares.runtime.events import EventBus
from ares.runtime.loop_guard import LoopGuardConfig


def build_agent(cls: type, bus: EventBus, agent_id: str, *, config: str | None = None,
                loop_guard: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Build `cls` with the settings in AGENT_DEFAULTS[config or agent_id].

    agent_id names the agent on the event bus and in traces. loop_guard
    overrides individual LOOP_GUARD settings for this agent.
    """
    cfg = AGENT_DEFAULTS[config or agent_id]
    return cls(
        get_engine(),
        cfg["model"],
        bus=bus,
        max_turns=cfg["max_turns"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        agent_id=agent_id,
        loop_guard=LoopGuardConfig(**{**LOOP_GUARD, **(loop_guard or {})}),
        **kwargs,
    )
