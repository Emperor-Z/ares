"""Learning from feedback — /good and /bad teach the orchestrator how to route.

Each rating is appended to ~/.ares/feedback.jsonl with the user's prompt and
the agent that answered it. `apply()` turns the most recent ratings into
routing examples in the orchestrator's system prompt: well-rated requests
show which agent to pick, badly rated ones which choice to avoid.

Routing is the one thing a handful of ratings can teach reliably. Tool sets
and turn limits are left alone: the traces don't show why a run went well,
and trimming either from a small sample makes agents worse.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path(os.environ.get("ARES_FEEDBACK_PATH", os.path.expanduser("~/.ares/feedback.jsonl")))
# Per polarity. The router is a 4B model, so the prompt stays short.
MAX_EXAMPLES = 8
_PROMPT_CHARS = 160
DIRECT = "direct"  # the orchestrator answered without handing off


def record(prompt: str, agents: list[str], score: float, trace_id: str | None, path: Path | None = None) -> None:
    """Append one rating. `agents` are the agents that produced the reply, in order."""
    path = path or FEEDBACK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.time(), "prompt": prompt, "agents": agents, "score": score, "trace_id": trace_id}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or FEEDBACK_FILE
    if not path.is_file():
        return []
    ratings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ratings.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed feedback line: %r", line[:80])
    return ratings


def routing_examples(ratings: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split ratings into (good, bad) lists of (prompt, agent), newest first.

    A prompt rated more than once counts only by its latest rating.
    """
    good: list[tuple[str, str]] = []
    bad: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in reversed(ratings):
        prompt = " ".join(str(r.get("prompt", "")).split())
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        agents = r.get("agents") or [DIRECT]
        target = good if r.get("score", 0) >= 0.5 else bad
        if len(target) < MAX_EXAMPLES:
            target.append((prompt[:_PROMPT_CHARS], agents[0]))
    return good, bad


def _choice(agent: str) -> str:
    return "answered directly" if agent == DIRECT else f"call_{agent}"


def routing_prompt(base: str, ratings: list[dict[str, Any]]) -> str:
    """`base` plus the routing lessons learned from `ratings`."""
    good, bad = routing_examples(ratings)
    if not good and not bad:
        return base
    lines = [base.rstrip(), "", "Routing lessons from the user's ratings:"]
    if good:
        lines.append("Good choices. Route similar requests the same way:")
        lines += [f'- "{p}" -> {_choice(a)}' for p, a in good]
    if bad:
        lines.append("Poor choices. Route similar requests differently:")
        lines += [f'- "{p}" -> {_choice(a)}' for p, a in bad]
    return "\n".join(lines) + "\n"


def apply(orchestrator: Any, base_prompt: str, path: Path | None = None) -> dict[str, int]:
    """Rebuild the orchestrator's system prompt from the ratings on disk."""
    ratings = load(path)
    orchestrator._system_prompt = routing_prompt(base_prompt, ratings)
    good, bad = routing_examples(ratings)
    return {"ratings": len(ratings), "good_examples": len(good), "bad_examples": len(bad)}
