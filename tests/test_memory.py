"""Tests for the local mem0 memory layer and how AresSystem uses it."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ares import memory
from ares.system import AresSystem


@pytest.fixture
def fake_mem0(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(memory, "_mem0", client)
    monkeypatch.setattr(memory, "_mem0_failed", False)
    monkeypatch.delenv("ARES_MEMORY", raising=False)
    return client


def test_disabled_by_env_is_a_noop(monkeypatch, fake_mem0):
    monkeypatch.setenv("ARES_MEMORY", "0")
    assert memory.recall("anything") == []
    assert memory.remember("a fact") is False
    fake_mem0.search.assert_not_called()
    fake_mem0.add.assert_not_called()


def test_recall_returns_memory_strings(fake_mem0):
    fake_mem0.search.return_value = {"results": [{"memory": "uses Arch"}, {"memory": ""}]}
    assert memory.recall("what OS?") == ["uses Arch"]
    fake_mem0.search.assert_called_once_with("what OS?", top_k=5, filters={"user_id": "ares"})


def test_recall_survives_backend_errors(fake_mem0):
    fake_mem0.search.side_effect = RuntimeError("qdrant locked")
    assert memory.recall("q") == []


def test_remember_stores_verbatim(fake_mem0):
    assert memory.remember("prefers tabs") is True
    fake_mem0.add.assert_called_once_with("prefers tabs", user_id="ares", infer=False)


def test_exchange_is_extracted_in_background(fake_mem0):
    memory.remember_exchange("I use FYERS", "ok")
    memory.flush(5)
    (messages,), kwargs = fake_mem0.add.call_args
    assert messages[0] == {"role": "user", "content": "I use FYERS"}
    assert kwargs["infer"] is True


def test_with_memories_prefixes_prompt():
    assert memory.with_memories("hi", []) == "hi"
    out = memory.with_memories("hi", ["uses Arch"])
    assert "- uses Arch" in out and out.endswith("hi")


def test_config_is_fully_local():
    cfg = memory._config()
    assert cfg["llm"]["provider"] == "ollama"
    assert cfg["embedder"]["provider"] == "ollama"
    assert cfg["vector_store"]["provider"] == "qdrant"
    assert "path" in cfg["vector_store"]["config"]


def _system(use_memory):
    system = AresSystem.__new__(AresSystem)
    system.use_memory = use_memory
    system._interaction_count = 0
    system._learn_every = 1000
    agent = MagicMock()
    agent.run.return_value = SimpleNamespace(content="answer")
    return system, agent


def test_run_injects_memories_and_saves_exchange(monkeypatch):
    monkeypatch.setattr(memory, "recall", lambda q: ["works on FundedIn"])
    saved = []
    monkeypatch.setattr(memory, "remember_exchange", lambda p, r: saved.append((p, r)))
    system, agent = _system(use_memory=True)

    assert system._run(agent, "what am I building?", []) == "answer"
    assert "works on FundedIn" in agent.run.call_args[0][0]
    assert saved == [("what am I building?", "answer")]


def test_run_without_memory_leaves_prompt_alone(monkeypatch):
    monkeypatch.setattr(memory, "recall", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(memory, "remember_exchange", MagicMock(side_effect=AssertionError))
    system, agent = _system(use_memory=False)

    system._run(agent, "hello", [])
    agent.run.assert_called_once_with("hello")


def test_direct_writes_share_the_background_writer(fake_mem0):
    import threading
    threads = []
    fake_mem0.add.side_effect = lambda *a, **k: threads.append(threading.current_thread().name)
    fake_mem0.delete_all.side_effect = lambda **k: threads.append(threading.current_thread().name)
    memory.remember_exchange("I use FYERS", "ok")
    memory.remember("prefers tabs")
    memory.forget_all()
    assert len(threads) == 3
    assert all(name.startswith("ares-memory") for name in threads)
