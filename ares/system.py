"""AresSystem — top-level object that wires all components together."""

from __future__ import annotations

import logging

from openjarvis.core.events import EventBus

from ares.agents import coder as _coder_mod
from ares.agents import thinker as _thinker_mod
from ares.agents import runner as _runner_mod
from ares.agents import orchestrator as _orch_mod
from ares.agents.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
from ares.agents import serena_agent as _serena_mod
from ares import learning, memory
from ares.observability import get_trace_store, wire_observability, wrap_with_collector

logger = logging.getLogger(__name__)

# Maximum number of prior messages injected as context (10 = 5 exchanges)
_HISTORY_WINDOW = 10


def _format_history(history: list[dict], prompt: str) -> str:
    """Prepend recent conversation turns to the prompt so the orchestrator
    has context of what was said before."""
    if not history:
        return prompt
    window = history[-_HISTORY_WINDOW:]
    lines = []
    for msg in window:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    ctx = "\n".join(lines)
    return f"Conversation so far:\n{ctx}\n\nUser: {prompt}"


class AresSystem:
    """Initialise and hold all Ares agents + shared infrastructure."""

    def __init__(self, use_memory: bool = True) -> None:
        # Local Qdrant allows one process per store, so only the REPL owns it.
        self.use_memory = use_memory
        self.bus = EventBus()

        wire_observability(self.bus)

        logger.info("Building agents...")
        _coder   = _coder_mod.build(self.bus)
        _thinker = _thinker_mod.build(self.bus)
        _runner  = _runner_mod.build(self.bus)
        _serena  = _serena_mod.build(self.bus)
        _orch    = _orch_mod.build(self.bus, _coder, _thinker, _runner, _serena)

        self.coder        = wrap_with_collector(_coder,   self.bus)
        self.thinker      = wrap_with_collector(_thinker, self.bus)
        self.runner       = wrap_with_collector(_runner,  self.bus)
        self.serena       = wrap_with_collector(_serena,  self.bus)
        self.orchestrator = wrap_with_collector(_orch,    self.bus)

        self._router = _orch
        self._last: tuple[str, list[str], str | None] | None = None  # (prompt, agents, trace_id)
        self.learn_now()

        logger.info("Ares ready.")

    def run(self, prompt: str, history: list[dict] | None = None) -> str:
        """Route a prompt through the orchestrator."""
        return self._run("orchestrator", prompt, history)

    def coder_run(self, prompt: str, history: list[dict] | None = None) -> str:
        return self._run("coder", prompt, history)

    def thinker_run(self, prompt: str, history: list[dict] | None = None) -> str:
        return self._run("thinker", prompt, history)

    def runner_run(self, prompt: str, history: list[dict] | None = None) -> str:
        return self._run("runner", prompt, history)

    def serena_run(self, prompt: str, history: list[dict] | None = None) -> str:
        return self._run("serena", prompt, history)

    def _run(self, name: str, prompt: str, history: list[dict] | None) -> str:
        self._last = None
        agent = getattr(self, name)
        full_prompt = _format_history(history or [], prompt)
        if self.use_memory:
            full_prompt = memory.with_memories(full_prompt, memory.recall(prompt))
        content = agent.run(full_prompt).content or "(no output)"
        if self.use_memory:
            memory.remember_exchange(prompt, content)
        trace = getattr(agent, "last_trace", None)
        agents = _handoffs(trace) if name == "orchestrator" else [name]
        self._last = (prompt, agents, getattr(trace, "trace_id", None))
        return content

    def rate(self, score: float) -> bool:
        """Rate the last reply (1.0 good, 0.0 bad) and re-learn routing.

        Returns False if there is no reply to rate.
        """
        if self._last is None:
            return False
        prompt, agents, trace_id = self._last
        learning.record(prompt, agents, score, trace_id)
        if trace_id:
            get_trace_store().update_feedback(trace_id, score)
        self.learn_now()
        return True

    def learn_now(self) -> dict:
        """Rebuild the orchestrator's routing lessons from all ratings so far."""
        return learning.apply(self._router, ORCHESTRATOR_SYSTEM_PROMPT)

    def routing_lessons(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        return learning.routing_examples(learning.load())

    def shutdown(self) -> None:
        """Graceful shutdown: finish memory writes, close Serena subprocess."""
        if self.use_memory:
            memory.close()
        try:
            from ares.serena_client import _client
            if _client is not None:
                _client.close()
                logger.info("Serena MCP client closed.")
        except Exception as exc:
            logger.debug("Serena close skipped: %s", exc)


def _handoffs(trace) -> list[str]:
    """Agents the orchestrator handed off to, in order, from its trace."""
    agents: list[str] = []
    for step in getattr(trace, "steps", None) or []:
        if str(getattr(step.step_type, "value", step.step_type)) != "tool_call":
            continue
        tool = (step.input or {}).get("tool", "")
        if tool.startswith("call_") and tool[5:] not in agents:
            agents.append(tool[5:])
    return agents


# ---------------------------------------------------------------------------
# Lightweight factory — builds a single agent without the full system.
# Used by A2A subprocesses so each server only pays for the agent it serves.
# ---------------------------------------------------------------------------

_AGENT_BUILDERS = {
    "coder":   lambda bus: _coder_mod.build(bus),
    "thinker": lambda bus: _thinker_mod.build(bus),
    "runner":  lambda bus: _runner_mod.build(bus),
    "serena":  lambda bus: _serena_mod.build(bus),
}


def build_single_agent(name: str):
    """Build one agent with minimal infrastructure (not the full system).

    Returns (callable, bus) where callable accepts a prompt string and returns
    a response string. The orchestrator is excluded — it requires all sub-agents
    and should use AresSystem instead.
    """
    if name not in _AGENT_BUILDERS and name != "orchestrator":
        raise ValueError(f"Unknown agent: {name!r}")

    bus = EventBus()
    wire_observability(bus)

    if name == "orchestrator":
        # Orchestrator needs all sub-agents; build the full system.
        system = AresSystem(use_memory=False)
        return system.run, system.bus

    agent = wrap_with_collector(_AGENT_BUILDERS[name](bus), bus)

    def _run(prompt: str) -> str:
        result = agent.run(prompt)
        return result.content or "(no output)"

    return _run, bus
