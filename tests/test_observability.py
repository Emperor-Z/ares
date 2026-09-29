"""Tests for the Langfuse exporter."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from openjarvis.core.types import Trace

from ares import observability


def _trace(query: str) -> Trace:
    return Trace(query=query, agent="coder", result="ok")


def test_traces_with_shared_prefix_get_distinct_ids():
    # Every prompt with recalled memories starts with the same text.
    prefix = "Things you remember about the user and their work:\n"
    a = observability._trace_to_langfuse_body(_trace(prefix + "first"))
    b = observability._trace_to_langfuse_body(_trace(prefix + "second"))
    assert a["batch"][0]["id"] != b["batch"][0]["id"]


def test_steps_belong_to_their_trace():
    trace = _trace("q")
    trace.steps = [SimpleNamespace(step_type="generate", timestamp=1.0, duration_seconds=0.5)]
    batch = observability._trace_to_langfuse_body(trace)["batch"]
    assert batch[1]["body"]["traceId"] == batch[0]["id"] == trace.trace_id


def test_no_export_without_keys(monkeypatch):
    monkeypatch.setattr(observability, "LANGFUSE_PK", "")
    submit = MagicMock()
    monkeypatch.setattr(observability._exporter, "submit", submit)
    observability._on_trace_complete(SimpleNamespace(data={"trace": _trace("q")}))
    submit.assert_not_called()


def test_export_runs_off_the_calling_thread(monkeypatch):
    monkeypatch.setattr(observability, "LANGFUSE_PK", "pk")
    monkeypatch.setattr(observability, "LANGFUSE_SK", "sk")
    submit = MagicMock()
    monkeypatch.setattr(observability._exporter, "submit", submit)
    trace = _trace("q")
    observability._on_trace_complete(SimpleNamespace(data={"trace": trace}))
    submit.assert_called_once_with(observability._export, trace)


def test_init_trace_store_creates_db(monkeypatch, tmp_path):
    path = tmp_path / "nested" / "traces.db"
    monkeypatch.setattr(observability, "TRACE_DB_PATH", str(path))
    observability.init_trace_store()
    assert path.is_file()
