"""Runner agent — swift (qwen2.5-coder:3b) via NativeReActAgent.

Handles: quick tasks, file lookups, shell one-liners, system checks,
simple calculations. Fire-and-forget, speed priority.

NativeReActAgent used because swift (3b) reliably follows Thought/Action/
Final Answer format. Promote to MonitorOperativeAgent + forge if long-horizon
background job scheduling is needed later.
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
    cfg = AGENT_DEFAULTS["runner"]

    agent = NativeReActAgent(
        get_engine(),
        cfg["model"],
        tools=[ShellExecTool(), FileReadTool(), FileWriteTool(), CalculatorTool()],
        bus=bus,
        max_turns=cfg["max_turns"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    agent._loop_guard = LoopGuard(LoopGuardConfig(**LOOP_GUARD), bus=bus)
    agent.agent_id = "runner"  # upstream default is the class name, shared by several agents
    return agent
