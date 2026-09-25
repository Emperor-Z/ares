# Ares

[![tests](https://github.com/Emperor-Z/ares/actions/workflows/tests.yml/badge.svg)](https://github.com/Emperor-Z/ares/actions/workflows/tests.yml)

Local multi-agent AI system for running specialised assistants on a personal machine. Ares combines local Ollama models, role-specific agents, A2A HTTP services, memory, Serena code-navigation integration, and Langfuse observability behind a terminal REPL.

Ares is also the orchestration backbone underneath the [VeriSim](https://github.com/Emperor-Z/verisim) dissertation system, handling local agent routing and memory for that project.

## Demo

![Ares REPL demo](demo/ares-demo.gif)

A real local session: `./start.sh` boots Ollama, Langfuse, and all five A2A agent servers, then `/coder` and `/thinker` run against local models (qwen2.5-coder and deepseek-r1) with real responses.

## What It Does

- Routes prompts through an orchestrator agent
- Provides direct agent commands for coder, thinker, runner, and Serena workflows
- Runs local models through Ollama instead of hosted model APIs
- Exposes A2A health endpoints for individual agents
- Tracks runs through a local/self-hosted Langfuse stack
- Includes a learning cycle hook for memory and behaviour refinement experiments

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
uv venv && uv pip install -e .          # add ".[memory]" for mem0 persistence
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

## Built on

Ares uses [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) (Apache-2.0) as its agent runtime: the ReAct and orchestrator agents, loop guard, tools, event bus, trace store and A2A server. The agent roster, routing, Serena integration, memory wiring, Langfuse exporter and REPL are Ares's own.

## Status

This is an experimental personal AI system. It is useful as a portfolio project for local AI orchestration, and it expects the local Ollama models above to be pulled. Next improvement: Dockerised agent services.

Tests live in `tests/` and run on every push (`uv pip install -e ".[dev]" && pytest`), covering A2A endpoints, config loading, agent handoff, conversation history, and the Serena tool wrappers.
