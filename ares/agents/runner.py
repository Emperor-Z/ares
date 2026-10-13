"""Runner agent — swift (qwen2.5-coder:3b) with the ReAct loop.

Handles: quick tasks, file lookups, shell one-liners, system checks,
simple calculations. Same tools as the coder on a smaller, faster model.
"""

from __future__ import annotations

from ares.agents._base import build_agent
from ares.runtime.agents import ReActAgent
from ares.runtime.events import EventBus
from ares.tools.builtin import standard_tools


def build(bus: EventBus) -> ReActAgent:
    return build_agent(ReActAgent, bus, "runner", tools=standard_tools())
