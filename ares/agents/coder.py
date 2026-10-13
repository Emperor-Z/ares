"""Coder agent — forge (qwen2.5-coder:7b) with the ReAct loop.

Handles: code generation, debugging, refactoring, file I/O, shell execution.
"""

from __future__ import annotations

from ares.agents._base import build_agent
from ares.runtime.agents import ReActAgent
from ares.runtime.events import EventBus
from ares.tools.builtin import standard_tools


def build(bus: EventBus) -> ReActAgent:
    return build_agent(ReActAgent, bus, "coder", tools=standard_tools())
