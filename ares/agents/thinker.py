"""Thinker agent — rune (deepseek-r1:7b) with the Action loop.

Handles: deep research, planning, multi-step reasoning, analysis.
deepseek-r1 has native chain-of-thought which makes it strongest for
complex reasoning chains that qwen3:4b would shortcut. It has no native
tool calling, so it calls tools with text "Action:" blocks.
"""

from __future__ import annotations

from ares.agents._base import build_agent
from ares.runtime.agents import ActionAgent
from ares.runtime.events import EventBus
from ares.tools.builtin import standard_tools


def build(bus: EventBus) -> ActionAgent:
    return build_agent(ActionAgent, bus, "thinker", tools=standard_tools())
