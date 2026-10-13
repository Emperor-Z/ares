"""Serena agent — forge (qwen2.5-coder:7b) with IDE-level semantic tools.

Handles: symbol navigation, cross-file refactoring, precise edits,
         codebase structure analysis, atomic renames.

Uses the same model and parameters as the coder agent but its toolset
is Serena's semantic tools rather than raw shell/file ops.
"""

from __future__ import annotations

from ares.agents._base import build_agent
from ares.runtime.agents import ReActAgent
from ares.runtime.events import EventBus
from ares.tools.builtin import FileReadTool
from ares.tools.serena_tools import all_serena_tools


def build(bus: EventBus) -> ReActAgent:
    return build_agent(
        ReActAgent, bus, "serena",
        config="coder",  # forge model, same quality tier as coder
        tools=[*all_serena_tools(), FileReadTool()],
    )
