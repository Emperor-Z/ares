"""Thinker agent — rune (deepseek-r1:7b) via NativeOpenHandsAgent.

Handles: deep research, planning, multi-step reasoning, analysis.
deepseek-r1 has native chain-of-thought which makes it strongest for
complex reasoning chains that qwen3:4b would shortcut. It has no native
tool calling, so it uses tools through the agent's text Action format.
"""

from __future__ import annotations

from openjarvis.agents.native_openhands import NativeOpenHandsAgent
from openjarvis.core.events import EventBus
from openjarvis.tools.calculator import CalculatorTool
from openjarvis.tools.file_read import FileReadTool
from openjarvis.tools.file_write import FileWriteTool
from openjarvis.tools.shell_exec import ShellExecTool

from ares.agents._base import build_agent


def build(bus: EventBus) -> NativeOpenHandsAgent:
    return build_agent(
        NativeOpenHandsAgent, bus, "thinker",
        tools=[ShellExecTool(), FileReadTool(), FileWriteTool(), CalculatorTool()],
    )
