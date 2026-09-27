"""Serena agent — forge (qwen2.5-coder:7b) with IDE-level semantic tools.

Handles: symbol navigation, cross-file refactoring, precise edits,
         codebase structure analysis, atomic renames.

Uses the same model and parameters as the coder agent but its toolset
is Serena's semantic tools rather than raw shell/file ops.
"""

from __future__ import annotations

from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.agents.loop_guard import LoopGuard, LoopGuardConfig
from openjarvis.core.events import EventBus
from openjarvis.tools.file_read import FileReadTool

from ares.config import AGENT_DEFAULTS, LOOP_GUARD
from ares.engine import get_engine
from ares.tools.serena_tools import all_serena_tools


def build(bus: EventBus) -> NativeReActAgent:
    cfg = AGENT_DEFAULTS["coder"]  # forge model, same quality tier as coder

    agent = NativeReActAgent(
        get_engine(),
        cfg["model"],
        tools=[*all_serena_tools(), FileReadTool()],
        bus=bus,
        max_turns=cfg["max_turns"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    agent._loop_guard = LoopGuard(LoopGuardConfig(**LOOP_GUARD), bus=bus)
    agent.agent_id = "serena"  # upstream default is the class name, shared by several agents
    return agent
