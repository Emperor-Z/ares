"""Observability — TraceCollector wiring + Langfuse exporter.

TraceCollector wraps each agent and records every run() to a shared
SQLite TraceStore. A Langfuse exporter subscribes to TRACE_COMPLETE
events and ships them to self-hosted Langfuse on a background thread,
so a slow or missing Langfuse never delays a reply.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from openjarvis.core.events import EventBus, EventType
from openjarvis.traces.collector import TraceCollector
from openjarvis.traces.store import TraceStore

logger = logging.getLogger(__name__)

LANGFUSE_HOST     = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PK       = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SK       = os.environ.get("LANGFUSE_SECRET_KEY", "")
TRACE_DB_PATH     = os.environ.get("ARES_TRACE_DB_PATH", os.path.expanduser("~/.ares/traces.db"))

_exporter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ares-langfuse")


# ---------------------------------------------------------------------------
# Shared TraceStore
# ---------------------------------------------------------------------------

_store: TraceStore | None = None

def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        os.makedirs(os.path.dirname(TRACE_DB_PATH), exist_ok=True)
        _store = TraceStore(db_path=TRACE_DB_PATH)
    return _store


def init_trace_store() -> None:
    """Create the trace DB and its schema, then close it (safe before forking)."""
    os.makedirs(os.path.dirname(TRACE_DB_PATH), exist_ok=True)
    TraceStore(db_path=TRACE_DB_PATH).close()


# ---------------------------------------------------------------------------
# Langfuse exporter (bus subscriber on TRACE_COMPLETE)
# ---------------------------------------------------------------------------

def _trace_to_langfuse_body(trace: Any) -> dict:
    """Convert a Trace to Langfuse ingestion format."""
    trace_id = trace.trace_id
    steps = []
    for i, step in enumerate(getattr(trace, "steps", [])):
        step_ts = _ts(getattr(step, "timestamp", time.time()))
        steps.append({
            "id": f"{trace_id}-step-{i}",
            "type": "span-create",
            "timestamp": step_ts,
            "body": {
                "id": f"{trace_id}-step-{i}",
                "traceId": trace_id,
                "name": str(getattr(step, "step_type", "step")),
                "startTime": _ts(getattr(step, "timestamp", time.time())),
                "endTime": _ts(getattr(step, "timestamp", time.time()) + getattr(step, "duration_seconds", 0)),
                "input": getattr(step, "input", {}),
                "output": getattr(step, "output", {}),
                "metadata": getattr(step, "metadata", {}),
            },
        })

    started = getattr(trace, "started_at", time.time())
    body = [
        {
            "id": trace_id,
            "type": "trace-create",
            "timestamp": _ts(started),
            "body": {
                "id": trace_id,
                "name": getattr(trace, "agent", "unknown"),
                "input": trace.query,
                "output": getattr(trace, "result", ""),
                "timestamp": _ts(started),
                "startTime": _ts(started),
                "endTime": _ts(getattr(trace, "ended_at", time.time())),
                "metadata": {
                    "model": getattr(trace, "model", ""),
                    "engine": getattr(trace, "engine", ""),
                    "total_tokens": getattr(trace, "total_tokens", 0),
                    "total_latency_s": getattr(trace, "total_latency_seconds", 0),
                },
                "tags": ["ares"],
            },
        },
        *steps,
    ]
    return {"batch": body}


def _ts(unix: float) -> str:
    import datetime
    dt = datetime.datetime.fromtimestamp(unix, datetime.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _on_trace_complete(event: Any) -> None:
    trace = event.data.get("trace") if hasattr(event, "data") else event.get("trace")
    if trace is None or not (LANGFUSE_PK and LANGFUSE_SK):
        return
    _exporter.submit(_export, trace)


def _export(trace: Any) -> None:
    try:
        body = _trace_to_langfuse_body(trace)
        resp = httpx.post(
            f"{LANGFUSE_HOST}/api/public/ingestion",
            json=body,
            auth=(LANGFUSE_PK, LANGFUSE_SK),
            timeout=5.0,
        )
        if resp.status_code not in (200, 207):
            logger.debug("Langfuse ingestion returned %s", resp.status_code)
    except Exception as exc:
        # Langfuse is optional — never block the agent on export failure
        logger.debug("Langfuse export skipped: %s", exc)


# ---------------------------------------------------------------------------
# Public wiring API
# ---------------------------------------------------------------------------

def wire_observability(bus: EventBus) -> None:
    """Subscribe the Langfuse exporter to the bus. Call once at startup."""
    bus.subscribe(EventType.TRACE_COMPLETE, _on_trace_complete)
    if LANGFUSE_PK and LANGFUSE_SK:
        logger.info("Langfuse exporter wired to EventBus (target: %s)", LANGFUSE_HOST)
    else:
        logger.info("Langfuse keys not set; traces stay in %s", TRACE_DB_PATH)


def wrap_with_collector(agent: Any, bus: EventBus) -> TraceCollector:
    """Wrap an agent in a TraceCollector backed by the shared TraceStore."""
    return TraceCollector(agent, store=get_trace_store(), bus=bus)
