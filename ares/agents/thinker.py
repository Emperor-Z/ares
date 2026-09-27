"""Thinker agent — rune (deepseek-r1:7b) via NativeOpenHandsAgent.

Handles: deep research, planning, multi-step reasoning, analysis.
deepseek-r1 has native chain-of-thought which makes it strongest for
complex reasoning chains that qwen3:4b would shortcut.
"""

from __future__ import annotations

from openjarvis.agents.native_openhands import NativeOpenHandsAgent
from openjarvis.agents.loop_guard import LoopGuard, LoopGuardConfig
from openjarvis.core.events import EventBus
from openjarvis.tools.shell_exec import ShellExecTool
from openjarvis.tools.file_read import FileReadTool
from openjarvis.tools.file_write import FileWriteTool
from openjarvis.tools.calculator import CalculatorTool

from ares.config import AGENT_DEFAULTS, LOOP_GUARD
from ares.engine import get_engine


def build(bus: EventBus) -> NativeOpenHandsAgent:
    cfg = AGENT_DEFAULTS["thinker"]

    agent = NativeOpenHandsAgent(
        get_engine(),
        cfg["model"],
        tools=[ShellExecTool(), FileReadTool(), FileWriteTool(), CalculatorTool()],
        bus=bus,
        max_turns=cfg["max_turns"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    agent._loop_guard = LoopGuard(LoopGuardConfig(**LOOP_GUARD), bus=bus)
    agent.agent_id = "thinker"  # upstream default is the class name, shared by several agents
    return agent
