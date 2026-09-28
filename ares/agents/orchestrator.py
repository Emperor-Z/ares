"""Orchestrator — routes input to coder/thinker/runner/serena via OrchestratorAgent.

Uses flux (qwen3.5:4b) for fast routing with native tool calling.
Wraps each sub-agent as a local tool for in-process handoff.
"""

from __future__ import annotations

import logging
from typing import Any

from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.agents._stubs import AgentResult
from openjarvis.core.events import EventBus
from openjarvis.tools._stubs import BaseTool, ToolSpec, ToolResult

from ares.agents._base import build_agent

logger = logging.getLogger(__name__)


class _AgentHandoffTool(BaseTool):
    """Wraps a local agent as a callable tool for the orchestrator."""

    tool_id: str

    def __init__(self, name: str, description: str, agent: Any) -> None:
        self._name = name
        self._description = description
        self._agent = agent
        self.tool_id = name

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question to send to this agent.",
                    }
                },
                "required": ["task"],
            },
            category="agent_handoff",
        )

    def execute(self, **params: Any) -> ToolResult:
        task = params.get("task", "")
        if not task:
            return ToolResult(tool_name=self._name, content="No task provided.", success=False)
        try:
            result: AgentResult = self._agent.run(task)
            return ToolResult(
                tool_name=self._name,
                content=result.content or "(no output)",
                success=True,
                metadata={"agent": self._name, "turns": result.turns},
            )
        except Exception as exc:
            logger.exception("Handoff to %s failed", self._name)
            return ToolResult(tool_name=self._name, content=f"Agent error: {exc}", success=False)


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are Ares, a local AI orchestrator. You have four specialist agents:

- call_coder  : code generation, debugging, shell commands, file operations, writing new code
- call_thinker: deep reasoning, research, planning, analysis, explanations
- call_runner : quick tasks, background jobs, monitoring, scheduling, system checks
- call_serena : IDE-level code intelligence — finding symbols, understanding structure,
                cross-file refactoring, atomic renames, impact analysis, precise edits
                to existing code using symbol names rather than line numbers

For each user request:
1. Decide which agent is best suited.
2. Call that agent with a clear, self-contained task description.
3. Return the agent's response directly — do not add commentary.

Use call_serena when: asked to find where something is defined/used, rename across files,
refactor by symbol, understand a codebase's structure, or make targeted edits to existing code.
Use call_coder when: writing new code from scratch, running shell commands, debugging with execution.
If the request is ambiguous, use call_thinker.
"""


def build(bus: EventBus, coder, thinker, runner, serena) -> OrchestratorAgent:
    handoff_tools = [
        _AgentHandoffTool(
            "call_coder",
            "Send a coding task to the coder agent (forge/qwen2.5-coder:7b). "
            "Use for: writing new code, debugging with execution, shell commands, file ops.",
            coder,
        ),
        _AgentHandoffTool(
            "call_thinker",
            "Send a reasoning task to the thinker agent (rune/deepseek-r1:7b). "
            "Use for: research, planning, analysis, multi-step reasoning.",
            thinker,
        ),
        _AgentHandoffTool(
            "call_runner",
            "Send a quick or background task to the runner agent (swift/qwen2.5-coder:3b). "
            "Use for: monitoring, scheduling, system checks, fast lookups.",
            runner,
        ),
        _AgentHandoffTool(
            "call_serena",
            "Send a code intelligence task to the Serena agent (forge + Serena LSP tools). "
            "Use for: finding symbols, tracing references, cross-file renames, refactoring by "
            "symbol name, understanding codebase structure, making precise edits to existing code.",
            serena,
        ),
    ]

    return build_agent(
        OrchestratorAgent, bus, "orchestrator",
        tools=handoff_tools,
        mode="function_calling",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    )
