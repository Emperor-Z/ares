# Ares — Context

Last updated: 2026-09-27

## What this is
A local, offline-first multi-agent AI system for a personal machine, in
daily use for personal coding and work. Everything runs on local Ollama
models: no hosted model APIs, no telemetry. The terminal REPL is the main
interface. The same agents are also served over A2A HTTP.

Hardware it's tuned for: a laptop RTX 3050 Ti (4 GB VRAM) with 16 GB RAM.
Model choices and memory limits follow from that.

## Naming rule
The project is presented only as **Ares**. Docs, docstrings, comments and
startup text don't name the agent runtime it currently depends on. The
dependency itself stays in `pyproject.toml` and the imports until the
runtime is replaced (see Roadmap).

## Architecture
- **Agents** (`ares/agents/`, all built by `_base.build_agent()`):
  - `orchestrator`: `qwen3.5:4b`. Routes each prompt to a specialist through `call_*` handoff tools, or answers directly.
  - `coder`: `qwen2.5-coder:7b`. ReAct, with shell, file read/write and calculator.
  - `thinker`: `deepseek-r1:7b`. CodeAct-style. It has no native tool calling, so it uses tools through the text `Action:` format.
  - `runner`: `qwen2.5-coder:3b`. The coder's tools on a faster model.
  - `serena`: `qwen2.5-coder:7b` with Serena's semantic code tools (`ares/tools/serena_tools.py`). Parameter names follow Serena 1.3's schemas, which are checked on first use.
- **`ares/system.py`** (`AresSystem`): wires the agents. It adds conversation history (last 10 messages) and recalled memories to each prompt, and records which agent answered, for ratings.
- **`ares/engine.py`**: a shared Ollama engine. It streams `/api/chat` when the REPL's live view is listening.
- **`ares/live.py`**: the live REPL view. It streams answers and shows a status line for handoffs and tool calls.
- **`ares/memory.py`**: long-term memory. mem0 runs on Ollama (`qwen2.5-coder:3b` extracts facts, `nomic-embed-text` embeds them) over an on-disk Qdrant at `~/.ares/memory`. All writes go through one background worker, and only the REPL owns the store.
- **`ares/learning.py`**: `/good` and `/bad` ratings go to `~/.ares/feedback.jsonl`. They become routing examples in the orchestrator's system prompt, up to 8 good and 8 bad.
- **`ares/observability.py`**: every run is traced to `~/.ares/traces.db`. Traces are also exported to self-hosted Langfuse on a background thread, only when keys are set.
- **`ares/a2a_server.py`**: one A2A server per agent on `127.0.0.1:8100-8104`. `python -m ares.a2a_server` runs them all in the foreground.
- **Docker** (`Dockerfile`, `docker/agents/`): the five A2A servers in one container, using host networking, with Ollama on the host.

## Settings
Settings are read from `~/.ares/.env`, then this checkout's `.env`. Real environment variables win over both. See `.env.example`.

## Running
- `./start.sh`: starts Ollama, Langfuse (if Docker is available) and the A2A servers (under nohup, log in `~/.ares/logs/a2a.log`), then opens the REPL.
- `./start.sh stop`: stops the A2A servers.
- `ares`: just the REPL, from any directory. `/serena` works on that directory.
- Tests: `uv pip install -e ".[dev]" && pytest`. CI runs them on 3.12 and 3.13.

## Key decisions
- **Learning is routing only.** The earlier learning cycle never ran: nothing recorded feedback, and its output was never read. Its tool and turn-limit recommendations also came from a crude regex classifier and mixed up which agent used which tool, so they would have made agents worse. Ratings now teach routing, the one thing a few ratings can teach reliably.
- **Docker uses host networking.** Ollama and Langfuse bind to `127.0.0.1`, and host networking reaches them without exposing either.
- **Langfuse is optional** and bound to localhost. Traces always go to the local SQLite DB.
- **Qdrant local is single-process**, so only the REPL uses memory. The A2A orchestrator runs with memory off.

## Known limitations
- With a browser and IDE open, free RAM can drop below what the 4B and 7B models need to load, and Ollama returns a 500 error.
- The A2A orchestrator picks up new ratings only when it restarts.
- Not yet verified with the real models: whether the `qwen3.5:4b` router changes its routing because of the rating examples, and a full `/serena` agent run inside Docker.

## Roadmap
1. Merge the open PRs: #4 reliability, #5 agent builders, #6 learning from ratings, #7 Serena 1.3 fix, #8 Docker.
2. **Last:** replace the agent runtime dependency with Ares's own implementation of the engine, agents, loop guard, tools, event bus, traces and A2A. If any upstream code is adapted rather than rewritten, keep its Apache-2.0 license text and a NOTICE file with it.
