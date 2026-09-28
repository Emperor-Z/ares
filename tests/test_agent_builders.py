"""Tests for the shared agent builder."""

from openjarvis.core.events import EventBus

from ares.agents import _base
from ares.config import AGENT_DEFAULTS


class _FakeAgent:
    agent_id = "fake_class_default"

    def __init__(self, engine, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


def test_uses_named_config_and_sets_agent_id(monkeypatch):
    monkeypatch.setattr(_base, "get_engine", lambda: "engine")
    agent = _base.build_agent(_FakeAgent, EventBus(), "serena", config="coder", tools=["t"])
    cfg = AGENT_DEFAULTS["coder"]
    assert agent.agent_id == "serena"
    assert agent.kwargs["agent_id"] == "serena"
    assert agent.model == cfg["model"]
    assert agent.kwargs["max_turns"] == cfg["max_turns"]
    assert agent.kwargs["tools"] == ["t"]
    assert agent._loop_guard is not None
