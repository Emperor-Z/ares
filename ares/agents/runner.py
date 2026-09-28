"""Runner agent — swift (qwen2.5-coder:3b) via NativeReActAgent.

Handles: quick tasks, file lookups, shell one-liners, system checks,
simple calculations. Same tools as the coder on a smaller, faster model.
"""

from __future__ import annotations

from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.core.events import EventBus
from openjarvis.tools.calculator import CalculatorTool
from openjarvis.tools.file_read import FileReadTool
from openjarvis.tools.file_write import FileWriteTool
from openjarvis.tools.shell_exec import ShellExecTool

from ares.agents._base import build_agent


def build(bus: EventBus) -> NativeReActAgent:
    return build_agent(
        NativeReActAgent, bus, "runner",
        tools=[ShellExecTool(), FileReadTool(), FileWriteTool(), CalculatorTool()],
    )
