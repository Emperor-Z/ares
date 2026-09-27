# Ares

[![tests](https://github.com/Emperor-Z/ares/actions/workflows/tests.yml/badge.svg)](https://github.com/Emperor-Z/ares/actions/workflows/tests.yml)

Local multi-agent AI system for running specialised assistants on a personal machine. Ares combines local Ollama models, role-specific agents, A2A HTTP services, memory, Serena code-navigation integration, and Langfuse observability behind a terminal REPL.

Its persona-per-agent design with a hidden orchestrator was the model for the multi-agent approach in [VeriSim](https://github.com/Emperor-Z/verisim), my MSc dissertation prototype. VeriSim is a separate codebase and doesn't run on Ares.

## Demo

![Ares REPL demo](demo/ares-demo.gif)

A real local session: `./start.sh` boots Ollama, Langfuse, and all five A2A agent servers, then `/coder` and `/thinker` run against local models (qwen2.5-coder and deepseek-r1) with real responses.

## What It Does

- Routes prompts through an orchestrator agent
- Provides direct agent commands for coder, thinker, runner, and Serena workflows
- Runs local models through Ollama instead of hosted model APIs
- Exposes A2A health endpoints for individual agents
- Tracks runs through a local/self-hosted Langfuse stack
- Remembers facts about you and your projects across sessions, fully locally (mem0 + Ollama + on-disk Qdrant)
- Includes a learning cycle hook for behaviour refinement experiments

## Architecture

```text
Terminal REPL
    |
    v
AresSystem
    |
    +-- orchestrator agent
    +-- coder agent
    +-- thinker agent
    +-- runner agent
    +-- Serena/code-navigation agent
    |
    +-- Ollama local models
    +-- memory layer
    +-- Langfuse observability
    +-- A2A HTTP services (:8100-:8104)
```

## Commands

```text
/coder <task>    code-focused task execution
/thinker <task>  deeper reasoning or planning
/runner <task>   quick execution-oriented task
/serena <task>   codebase-aware navigation and edits
/learn           run the learning cycle
/remember <fact> store a fact verbatim
/memories        list what Ares remembers
/forget          delete all memories (asks first)
/quit            exit the REPL
```

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) with the models in `ares/config.py` pulled (`start.sh` tells you which are missing)
- Docker, for the optional Langfuse tracing stack
- [Serena](https://github.com/oraios/serena) on your `PATH`, for `/serena`

## Install

```bash
git clone https://github.com/Emperor-Z/ares.git
cd ares
uv venv && uv pip install -e ".[memory]"  # drop [memory] to skip long-term memory
ollama pull nomic-embed-text             # embeddings for memory
cp .env.example .env                    # then fill in your Langfuse keys
```

## Run

```bash
./start.sh     # starts Ollama, Langfuse and the A2A servers, then opens the REPL
ares           # or just the REPL, if the services are already up
```

`start.sh` checks or starts:

- Ollama on `localhost:11434`
- Langfuse on `localhost:3000`
- A2A agent services on ports `8100` to `8104`

`/serena` works on the directory you launch Ares from. Set `ARES_SERENA_PROJECT` to point it somewhere else.

## Memory

With the `memory` extra installed, Ares recalls related memories before each prompt and adds them to the context. After each reply it hands the exchange to a small local model (`qwen2.5-coder:3b` by default) in the background, which pulls out facts worth keeping: your projects, tools, hardware, preferences and deadlines. Everything stays in `~/.ares/memory`. mem0's telemetry is switched off.

Extraction takes a few seconds per reply on an RTX 3050 Ti and doesn't block the REPL. A 3B model misses vague statements, so use `/remember` for anything that matters. Set `ARES_MEMORY=0` to turn memory off.

## Status

Ares is in daily use for personal coding and work. It is still experimental and expects the local Ollama models above to be pulled. Next improvement: Dockerised agent services.

Tests live in `tests/` and run on every push (`uv pip install -e ".[dev]" && pytest`), covering A2A endpoints, config loading, agent handoff, conversation history, and the Serena tool wrappers.
