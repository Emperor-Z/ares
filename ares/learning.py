"""Learning from feedback — /good and /bad teach the orchestrator how to route.

Each rating is appended to ~/.ares/feedback.jsonl with the user's prompt and
the agent that answered it. `routing_note()` finds the rated requests most
like a new one and attaches them to it as a note: which choice was right,
which was wrong. The router is a 4B model, and in tests a note next to the
request moved 8 of 9 routing decisions, where the same lessons listed in the
system prompt moved 3. Ratings are read per request, so new ones apply at once.

Routing is the one thing a handful of ratings can teach reliably. Tool sets
and turn limits are left alone: the traces don't show why a run went well,
and trimming either from a small sample makes agents worse.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path(os.environ.get("ARES_FEEDBACK_PATH", os.path.expanduser("~/.ares/feedback.jsonl")))
# Per polarity, for /learn.
MAX_EXAMPLES = 8
# Rated requests attached to one prompt, and how alike they must be (Jaccard over words).
MAX_NOTE_LESSONS = 4
MIN_SIMILARITY = 0.2
_STOPWORDS = frozenset(
    "a an the this that these those is are was were be to of in on at for with and or but it its "
    "my me i you your we our can could would should will do does did how what which who when where "
    "why please some any all from by as into about".split())
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


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOPWORDS}


def _similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def routing_note(prompt: str, ratings: list[dict[str, Any]]) -> str:
    """A note on how the user rated requests like `prompt`, or "" if none are alike."""
    words = _words(prompt)
    lessons: list[tuple[float, str, str, bool]] = []
    seen: set[str] = set()
    for r in reversed(ratings):  # latest rating per prompt wins
        rated = " ".join(str(r.get("prompt", "")).split())
        if not rated or rated in seen:
            continue
        seen.add(rated)
        score = _similarity(words, _words(rated))
        if score >= MIN_SIMILARITY:
            agents = r.get("agents") or [DIRECT]
            lessons.append((score, rated[:_PROMPT_CHARS], agents[0], r.get("score", 0) >= 0.5))
    if not lessons:
        return ""
    lessons.sort(key=lambda lesson: lesson[0], reverse=True)
    parts = [f'for "{p}" {_choice(a)} was {"right" if good else "wrong"}'
             for _, p, a, good in lessons[:MAX_NOTE_LESSONS]]
    return f"(Routing note from the user's past ratings: {'; '.join(parts)}.)"
