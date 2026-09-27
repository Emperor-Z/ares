"""Coder agent — forge (qwen2.5-coder:7b) via NativeReActAgent.

Handles: code generation, debugging, refactoring, file I/O, shell execution.
"""

from __future__ import annotations

from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.agents.loop_guard import LoopGuard, LoopGuardConfig
from openjarvis.core.events import EventBus
from openjarvis.tools.shell_exec import ShellExecTool
from openjarvis.tools.file_read import FileReadTool
from openjarvis.tools.file_write import FileWriteTool
from openjarvis.tools.calculator import CalculatorTool

from ares.config import AGENT_DEFAULTS, LOOP_GUARD
from ares.engine import get_engine


def build(bus: EventBus) -> NativeReActAgent:
    cfg = AGENT_DEFAULTS["coder"]

    agent = NativeReActAgent(
        get_engine(),
        cfg["model"],
        tools=[ShellExecTool(), FileReadTool(), FileWriteTool(), CalculatorTool()],
        bus=bus,
        max_turns=cfg["max_turns"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    # Replace default loop guard with Ares config
    agent._loop_guard = LoopGuard(LoopGuardConfig(**LOOP_GUARD), bus=bus)
    agent.agent_id = "coder"  # upstream default is the class name, shared by several agents
    return agent
