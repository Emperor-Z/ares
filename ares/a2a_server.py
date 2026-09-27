"""A2A HTTP servers — expose coder, thinker, runner, serena as A2A endpoints.

Each agent is served on its own port as a Google A2A-compliant JSON-RPC
server. The orchestrator calls them in-process via _AgentHandoffTool, but
external systems (other Ares nodes, MCP clients, scripts) can call any
agent directly over HTTP.

Ports:
    8100 — orchestrator
    8101 — coder   (forge / qwen2.5-coder:7b)
    8102 — thinker (rune  / deepseek-r1:7b)
    8103 — runner  (swift / qwen2.5-coder:3b)
    8104 — serena  (forge / qwen2.5-coder:7b + Serena IDE tools)
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from openjarvis.a2a.protocol import AgentCard
from openjarvis.a2a.server import A2AServer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent card definitions
# ---------------------------------------------------------------------------

CARDS = {
    "orchestrator": AgentCard(
        name="Ares Orchestrator",
        description="Routes requests to the best specialist agent (coder/thinker/runner).",
        url="http://localhost:8100",
        version="0.1.0",
        capabilities=["routing", "tool_use", "a2a_delegation"],
        skills=["intent_classification", "agent_handoff"],
    ),
    "coder": AgentCard(
        name="Ares Coder",
        description="Code generation, debugging, refactoring, shell execution. (forge / qwen2.5-coder:7b)",
        url="http://localhost:8101",
        version="0.1.0",
        capabilities=["code_generation", "debugging", "shell_exec", "file_io"],
        skills=["python", "bash", "refactor", "debug"],
    ),
    "thinker": AgentCard(
        name="Ares Thinker",
        description="Deep reasoning, research, planning, multi-step analysis. (rune / deepseek-r1:7b)",
        url="http://localhost:8102",
        version="0.1.0",
        capabilities=["reasoning", "research", "planning", "analysis"],
        skills=["chain_of_thought", "structured_planning", "comparison"],
    ),
    "runner": AgentCard(
        name="Ares Runner",
        description="Fast tasks, system checks, calculations, fire-and-forget. (swift / qwen2.5-coder:3b)",
        url="http://localhost:8103",
        version="0.1.0",
        capabilities=["fast_execution", "tool_use", "scheduling"],
        skills=["shell", "calculator", "file_read", "quick_lookup"],
    ),
    "serena": AgentCard(
        name="Ares Serena",
        description="IDE-level code intelligence: symbol navigation, cross-file refactoring, atomic renames. (forge / qwen2.5-coder:7b + Serena LSP)",
        url="http://localhost:8104",
        version="0.1.0",
        capabilities=["semantic_search", "refactoring", "symbol_nav", "precise_editing"],
        skills=["find_symbol", "rename_symbol", "replace_symbol_body", "find_references"],
    ),
}

PORTS = {
    "orchestrator": 8100,
    "coder": 8101,
    "thinker": 8102,
    "runner": 8103,
    "serena": 8104,
}


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def make_app(agent_name: str, handler_fn: Any, bus: Any = None) -> FastAPI:
    """Create a FastAPI app exposing the A2A endpoints for one agent."""
    card = CARDS[agent_name]
    server = A2AServer(agent_card=card, handler=handler_fn, bus=bus)
    app = FastAPI(title=card.name, docs_url=None, redoc_url=None)

    @app.get("/.well-known/agent.json")
    async def agent_card() -> JSONResponse:
        return JSONResponse(card.to_dict())

    # Plain def: agent runs block, so FastAPI must run this in its threadpool
    # or /health and every other request stall until the task finishes.
    @app.post("/a2a/tasks")
    def handle_task(request: dict) -> JSONResponse:
        result = server.handle_request(request)
        return JSONResponse(result)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "agent": agent_name})

    return app


# ---------------------------------------------------------------------------
# Per-process server runner (called via multiprocessing)
# ---------------------------------------------------------------------------

def _run_agent_server(agent_name: str, port: int) -> None:
    """Build only the requested agent and serve it via A2A.

    Each server builds only its own agent, not the full AresSystem, so it
    only pays for the agent it serves.
    """
    import logging
    logging.basicConfig(level=logging.WARNING)

    from ares.system import build_single_agent
    handler_fn, bus = build_single_agent(agent_name)

    app = make_app(agent_name, handler_fn, bus=bus)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def launch_all() -> list[multiprocessing.Process]:
    """Start every agent server as a daemon subprocess and return the handles.

    Daemon children die with this process, so the caller must stay alive for
    as long as the servers should run. `serve_all()` does that.
    """
    procs = []
    for name, port in PORTS.items():
        p = multiprocessing.Process(
            target=_run_agent_server,
            args=(name, port),
            name=f"ares-a2a-{name}",
            daemon=True,
        )
        p.start()
        logger.info("A2A server %s started on port %d (pid=%d)", name, port, p.pid)
        procs.append(p)
    return procs


def serve_all() -> None:
    """Run every agent server in the foreground until SIGTERM or Ctrl-C."""
    procs = launch_all()

    def stop(*_: Any) -> None:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for p in procs:
        p.join()


def launch_one(agent_name: str) -> None:
    """Run a single agent's A2A server in the foreground (for testing)."""
    _run_agent_server(agent_name, PORTS[agent_name])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve_all()
