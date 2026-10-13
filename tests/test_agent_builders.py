"""Tests for the shared agent builder."""

from ares.runtime.agents import ReActAgent

from ares.agents import _base
from ares.config import AGENT_DEFAULTS, LOOP_GUARD
from ares.runtime.events import EventBus


class _FakeAgent:
    def __init__(self, engine, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


def test_uses_named_config_and_sets_agent_id(monkeypatch):
    monkeypatch.setattr(_base, "get_engine", lambda: "engine")
    agent = _base.build_agent(_FakeAgent, EventBus(), "serena", config="coder", tools=["t"])
    cfg = AGENT_DEFAULTS["coder"]
    assert agent.kwargs["agent_id"] == "serena"
    assert agent.model == cfg["model"]
    assert agent.kwargs["max_turns"] == cfg["max_turns"]
    assert agent.kwargs["tools"] == ["t"]
    assert agent.kwargs["loop_guard"].max_identical_calls == LOOP_GUARD["max_identical_calls"]


def test_real_agents_take_their_id_from_the_builder(monkeypatch):
    monkeypatch.setattr(_base, "get_engine", lambda: "engine")
    agent = _base.build_agent(ReActAgent, EventBus(), "serena", config="coder")
    assert agent.agent_id == "serena"
