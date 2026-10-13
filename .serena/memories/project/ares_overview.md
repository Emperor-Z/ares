# Ares project overview

Ares is a local, offline-first multi-agent AI system (`/home/z/ares`). The
full project record is `context.md` at the repo root. It is local only and
gitignored: read it first, keep it updated, and never commit it.

Key files:
- `ares/cli.py`: the terminal REPL (`ares` / `python -m ares`). It loads settings from `~/.ares/.env` and then the checkout's `.env`. Commands: `/coder`, `/thinker`, `/runner`, `/serena`, `/good`, `/bad`, `/learn`, `/remember`, `/memories`, `/forget`, `/clear`, `/help`, `/quit`.
- `ares/system.py`: `AresSystem` wires the agents, memory, tracing and learning, and records which agent answered, for ratings. `build_single_agent` is used by the A2A servers.
- `ares/runtime/`: Ares's own agent runtime. `agents.py` (ReAct, Action and tool-calling loops), `tools.py` (tool interface, executor, shell approval via `set_confirm`), `loop_guard.py`, `events.py`, `types.py`, `tracing.py`, `a2a.py`.
- `ares/agents/`: one module per agent, all built through `_base.build_agent()`.
- `ares/engine.py`: Ares's Ollama client, with streaming for the live view (`ares/live.py`, which also asks the user to approve shell commands).
- `ares/tools/builtin.py`: shell (asks first), file read/write (refuse secrets) and calculator tools.
- `ares/learning.py`: attaches `/good` and `/bad` ratings of similar past requests to each orchestrator prompt as a routing note.
- `ares/memory.py`: mem0 on Ollama with local Qdrant. All writes go through one worker thread.
- `ares/observability.py`: traces to `~/.ares/traces.db`, plus optional Langfuse export.
- `ares/a2a_server.py`: A2A servers on `127.0.0.1:8100-8104`. `python -m ares.a2a_server` runs them all.
- `ares/tools/serena_tools.py`, `ares/serena_client.py`: Serena MCP tools. Parameter names follow Serena 1.3's schemas.
- `start.sh`, `Dockerfile`, `docker/`: startup, containerised agents, and the Langfuse stack.

Conventions:
- Refer to the project only as Ares in docs, comments and user-facing text. The exception is Apache-2.0 attribution: files adapted from upstream code keep their one-line notice and are listed in `NOTICE`.
- One focused commit per logical change. Verify against real Ollama and Serena where possible, not just unit tests.
