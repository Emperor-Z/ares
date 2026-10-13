# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Record every agent run as a Trace in SQLite and announce it on the bus."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ares.runtime.events import Event, EventBus, EventType
from ares.runtime.types import AgentResult, StepType, Trace, TraceStep

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS traces (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id              TEXT    NOT NULL UNIQUE,
    query                 TEXT    NOT NULL DEFAULT '',
    agent                 TEXT    NOT NULL DEFAULT '',
    model                 TEXT    NOT NULL DEFAULT '',
    engine                TEXT    NOT NULL DEFAULT '',
    result                TEXT    NOT NULL DEFAULT '',
    outcome               TEXT,
    feedback              REAL,
    started_at            REAL    NOT NULL DEFAULT 0.0,
    ended_at              REAL    NOT NULL DEFAULT 0.0,
    total_tokens          INTEGER NOT NULL DEFAULT 0,
    total_latency_seconds REAL    NOT NULL DEFAULT 0.0,
    metadata              TEXT    NOT NULL DEFAULT '{}',
    messages              TEXT    NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS trace_steps (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT    NOT NULL,
    step_index       INTEGER NOT NULL,
    step_type        TEXT    NOT NULL,
    timestamp        REAL    NOT NULL DEFAULT 0.0,
    duration_seconds REAL    NOT NULL DEFAULT 0.0,
    input            TEXT    NOT NULL DEFAULT '{}',
    output           TEXT    NOT NULL DEFAULT '{}',
    metadata         TEXT    NOT NULL DEFAULT '{}',
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);
"""


class TraceStore:
    """Append-only SQLite store. One connection, shared by threads under a lock."""

    def __init__(self, db_path: str | Path) -> None:
        path = str(db_path)
        if path != ":memory:":
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            p.touch(mode=0o600, exist_ok=True)
            path = str(p)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, trace: Trace) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO traces (trace_id, query, agent, model, engine, result, outcome, feedback,"
                " started_at, ended_at, total_tokens, total_latency_seconds, metadata, messages)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trace.trace_id, trace.query, trace.agent, trace.model, trace.engine, trace.result,
                 trace.outcome, trace.feedback, trace.started_at, trace.ended_at, trace.total_tokens,
                 trace.total_latency_seconds, _json(trace.metadata), _json(trace.messages)),
            )
            for i, step in enumerate(trace.steps):
                self._conn.execute(
                    "INSERT INTO trace_steps (trace_id, step_index, step_type, timestamp, duration_seconds,"
                    " input, output, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (trace.trace_id, i, StepType(step.step_type).value, step.timestamp, step.duration_seconds,
                     _json(step.input), _json(step.output), _json(step.metadata)),
                )
            self._conn.commit()

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT trace_id, query, agent, model, engine, result, outcome, feedback, started_at,"
                " ended_at, total_tokens, total_latency_seconds, metadata, messages"
                " FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None:
                return None
            steps = self._conn.execute(
                "SELECT step_type, timestamp, duration_seconds, input, output, metadata"
                " FROM trace_steps WHERE trace_id = ? ORDER BY step_index", (trace_id,)).fetchall()
        return Trace(
            trace_id=row[0], query=row[1], agent=row[2], model=row[3], engine=row[4], result=row[5],
            outcome=row[6], feedback=row[7], started_at=row[8], ended_at=row[9], total_tokens=row[10],
            total_latency_seconds=row[11], metadata=json.loads(row[12]), messages=json.loads(row[13]),
            steps=[TraceStep(step_type=StepType(s[0]), timestamp=s[1], duration_seconds=s[2],
                             input=json.loads(s[3]), output=json.loads(s[4]), metadata=json.loads(s[5]))
                   for s in steps],
        )

    def update_feedback(self, trace_id: str, score: float) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE traces SET feedback = ? WHERE trace_id = ?", (score, trace_id))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _json(value: Any) -> str:
    # Tool arguments come from models; never let one odd value lose the trace.
    return json.dumps(value, default=str)


class TraceCollector:
    """Wraps an agent; each run() is recorded from the events it publishes."""

    def __init__(self, agent: Any, *, store: TraceStore | None = None, bus: EventBus | None = None) -> None:
        self._agent = agent
        self._store = store
        self._bus = bus
        self._last_trace: Trace | None = None

    @property
    def agent_id(self) -> str:
        return getattr(self._agent, "agent_id", "unknown")

    @property
    def last_trace(self) -> Trace | None:
        return self._last_trace

    def run(self, input: str) -> AgentResult:
        steps: list[TraceStep] = []
        state: dict[str, Any] = {"model": "", "engine": ""}

        # Handoffs run sub-agents on the same bus, so their steps land here too.
        # The trace's model is the traced agent's own; each step names its agent.
        def on_inference_start(e: Event) -> None:
            agent = e.data.get("agent", "")
            if agent == self.agent_id or not state["model"]:
                state["model"] = e.data.get("model", state["model"])
                state["engine"] = e.data.get("engine", state["engine"])
            state[("inference_at", agent)] = e.timestamp

        def on_inference_end(e: Event) -> None:
            agent = e.data.get("agent", "")
            start = state.pop(("inference_at", agent), e.timestamp)
            usage = e.data.get("usage", {})
            steps.append(TraceStep(
                step_type=StepType.GENERATE, timestamp=start, duration_seconds=e.timestamp - start,
                input={"model": e.data.get("model", state["model"]), "agent": agent},
                output={**{k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                        "tokens": usage.get("total_tokens", 0),
                        "content": e.data.get("content", ""),
                        "tool_calls": e.data.get("tool_calls", []),
                        "finish_reason": e.data.get("finish_reason", "")},
                metadata={"engine": state["engine"]},
            ))

        # A handoff's own tool calls start and end inside it, so match by call.
        def on_tool_start(e: Event) -> None:
            state[("tool", e.data.get("call_key"))] = (e.timestamp, e.data.get("arguments", {}))

        def on_tool_end(e: Event) -> None:
            start, arguments = state.pop(("tool", e.data.get("call_key")), (e.timestamp, {}))
            steps.append(TraceStep(
                step_type=StepType.TOOL_CALL, timestamp=start,
                duration_seconds=e.data.get("latency", e.timestamp - start),
                input={"tool": e.data.get("tool", ""), "arguments": arguments,
                       "agent": e.data.get("agent", "")},
                output={"success": e.data.get("success", False), "result": e.data.get("result", "")},
            ))

        handlers = [
            (EventType.INFERENCE_START, on_inference_start),
            (EventType.INFERENCE_END, on_inference_end),
            (EventType.TOOL_CALL_START, on_tool_start),
            (EventType.TOOL_CALL_END, on_tool_end),
        ]
        if self._bus is not None:
            for event, handler in handlers:
                self._bus.subscribe(event, handler)
        started = time.time()
        try:
            result = self._agent.run(input)
        finally:
            if self._bus is not None:
                for event, handler in handlers:
                    self._bus.unsubscribe(event, handler)
        ended = time.time()

        steps.append(TraceStep(step_type=StepType.RESPOND, timestamp=ended,
                               output={"content": result.content, "turns": result.turns}))
        trace = Trace(query=input, agent=self.agent_id, model=state["model"], engine=state["engine"],
                      steps=steps, result=result.content, messages=result.metadata.get("messages", []),
                      started_at=started, ended_at=ended)
        # Wall time: a handoff's steps overlap the call_* step that contains them.
        trace.total_latency_seconds = ended - started
        trace.total_tokens = sum(s.output.get("tokens", 0) for s in steps)
        self._last_trace = trace
        if self._store is not None:
            self._store.save(trace)
        if self._bus is not None:
            self._bus.publish(EventType.TRACE_COMPLETE, {"trace": trace})
        return result
