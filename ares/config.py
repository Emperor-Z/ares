"""Model aliases and global constants for Ares.

All values can be overridden via environment variables — see .env.example.
"""

from __future__ import annotations

import os

OLLAMA_HOST = os.environ.get("ARES_OLLAMA_HOST", "http://localhost:11434")

# Model aliases — match existing Ollama pulls
SWIFT   = os.environ.get("ARES_RUNNER_MODEL",       "qwen2.5-coder:3b")
FORGE   = os.environ.get("ARES_CODER_MODEL",        "qwen2.5-coder:7b")
FLUX    = os.environ.get("ARES_ORCHESTRATOR_MODEL", "qwen3.5:4b")
RUNE    = os.environ.get("ARES_THINKER_MODEL",      "deepseek-r1:7b")
EMBED   = "nomic-embed-text"

# Per-agent assignment rationale:
#   orchestrator → FLUX:  qwen3.5:4b MoE — fastest routing, native tool calling, direct responses
#   coder        → FORGE: qwen2.5-coder:7b — best code quality
#   thinker      → RUNE:  deepseek-r1:7b — native CoT for deep reasoning
#   runner       → SWIFT: qwen2.5-coder:3b — fastest, full GPU, simple tasks

AGENT_DEFAULTS = {
    # max_tokens=2048 on orchestrator: enough to relay full sub-agent responses
    "orchestrator": {"model": FLUX,  "temperature": 0.2, "max_tokens": 2048, "max_turns": 6},
    "coder":        {"model": FORGE, "temperature": 0.3, "max_tokens": 2048, "max_turns": 12},
    "thinker":      {"model": RUNE,  "temperature": 0.6, "max_tokens": 4096, "max_turns": 15},
    "runner":       {"model": SWIFT, "temperature": 0.4, "max_tokens": 1024, "max_turns": 8},
}

LOOP_GUARD = {
    "max_identical_calls": 3,
    "ping_pong_window": 6,
    "poll_tool_budget": 5,
    "max_context_messages": 80,
    "warn_before_block": True,
}

# Models that must be present in Ollama before startup
REQUIRED_MODELS = [FLUX, FORGE, RUNE, SWIFT]
