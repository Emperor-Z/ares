"""Shared setup for every Ares agent: engine, per-agent config, loop guard."""

from __future__ import annotations

from typing import Any

from openjarvis.agents.loop_guard import LoopGuard, LoopGuardConfig
from openjarvis.core.events import EventBus

from ares.config import AGENT_DEFAULTS, LOOP_GUARD
from ares.engine import get_engine


def build_agent(cls: type, bus: EventBus, agent_id: str, *, config: str | None = None, **kwargs: Any) -> Any:
    """Build `cls` with the settings in AGENT_DEFAULTS[config or agent_id].

    agent_id names the agent on the event bus and in traces; the upstream
    default is the class name, which several Ares agents share.
    """
    cfg = AGENT_DEFAULTS[config or agent_id]
    agent = cls(
        get_engine(),
        cfg["model"],
        bus=bus,
        max_turns=cfg["max_turns"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        agent_id=agent_id,
        **kwargs,
    )
    agent.agent_id = agent_id  # bus events read the attribute, not the constructor arg
    agent._loop_guard = LoopGuard(LoopGuardConfig(**LOOP_GUARD), bus=bus)
    return agent
