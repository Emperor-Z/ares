"""Long-term memory — mem0 running entirely on local Ollama + on-disk Qdrant.

Before each prompt, Ares recalls related memories and adds them to the
context. After each reply, the exchange is handed to mem0 in a background
thread, where a small local model extracts facts worth keeping.

Optional: needs `pip install -e ".[memory]"`. Without it, or with
ARES_MEMORY=0, every function here is a no-op.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ares.config import EMBED, OLLAMA_HOST, SWIFT

logger = logging.getLogger(__name__)

USER_ID = "ares"
MEMORY_DIR = os.environ.get("ARES_MEMORY_DIR", os.path.expanduser("~/.ares/memory"))
# Fact extraction runs on every exchange, so it gets the fastest model.
EXTRACT_MODEL = os.environ.get("ARES_MEMORY_MODEL", SWIFT)
EMBED_DIMS = 768  # nomic-embed-text

# Small local models extract noticeably more with a narrow brief.
_INSTRUCTIONS = (
    "The user is a developer talking to their local coding assistant. "
    "Keep durable facts: projects they work on (by name) and what those projects "
    "use, their tools, hardware and OS, preferences, deadlines and people involved. "
    "Skip greetings, one-off questions and any code the assistant wrote."
)

_mem0: Any = None
_mem0_failed = False
_lock = threading.Lock()
# Every write goes through this one worker: it keeps them in order, stops
# them piling onto the GPU, and keeps two threads from writing to the local
# Qdrant store at once.
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ares-memory")


def enabled() -> bool:
    return os.environ.get("ARES_MEMORY", "1").lower() not in ("0", "false", "no")


def _config() -> dict:
    return {
        "llm": {
            "provider": "ollama",
            "config": {"model": EXTRACT_MODEL, "ollama_base_url": OLLAMA_HOST, "temperature": 0.1},
        },
        "embedder": {
            "provider": "ollama",
            "config": {"model": EMBED, "ollama_base_url": OLLAMA_HOST},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "ares",
                "path": os.path.join(MEMORY_DIR, "qdrant"),
                "on_disk": True,
                "embedding_model_dims": EMBED_DIMS,
            },
        },
        "history_db_path": os.path.join(MEMORY_DIR, "history.db"),
        "custom_instructions": _INSTRUCTIONS,
    }


def _get() -> Any:
    """Return the mem0 client, or None if memory is off or unavailable."""
    global _mem0, _mem0_failed
    if not enabled() or _mem0_failed:
        return None
    with _lock:
        if _mem0 is None and not _mem0_failed:
            # mem0 reads this at import time and otherwise phones home.
            os.environ.setdefault("MEM0_TELEMETRY", "False")
            # Its optional-extra hints (spaCy, fastembed) would clutter the REPL.
            logging.getLogger("mem0").setLevel(logging.ERROR)
            try:
                from mem0 import Memory
                os.makedirs(MEMORY_DIR, exist_ok=True)
                _mem0 = Memory.from_config(_config())
                logger.info("Memory ready at %s", MEMORY_DIR)
            except ImportError:
                _mem0_failed = True
                logger.info("mem0 not installed; long-term memory disabled")
            except Exception as exc:
                _mem0_failed = True
                logger.warning("Memory disabled, could not start mem0: %s", exc)
    return _mem0


def recall(query: str, limit: int = 5) -> list[str]:
    """Return up to `limit` stored memories relevant to `query`."""
    mem = _get()
    if mem is None or not query.strip():
        return []
    try:
        res = mem.search(query, top_k=limit, filters={"user_id": USER_ID})
        return [r["memory"] for r in res.get("results", []) if r.get("memory")]
    except Exception as exc:
        logger.warning("Memory recall failed: %s", exc)
        return []


def remember_exchange(prompt: str, response: str) -> None:
    """Queue an exchange for fact extraction. Returns immediately."""
    if _get() is None or not prompt.strip():
        return
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    _writer.submit(_add, messages, True)


def remember(fact: str) -> bool:
    """Store `fact` verbatim, skipping LLM extraction."""
    if _get() is None or not fact.strip():
        return False
    return _writer.submit(_add, fact, False).result()


def _add(content: Any, infer: bool) -> bool:
    try:
        _mem0.add(content, user_id=USER_ID, infer=infer)
        return True
    except Exception as exc:
        logger.warning("Memory store failed: %s", exc)
        return False


def list_all(limit: int = 50) -> list[dict]:
    mem = _get()
    if mem is None:
        return []
    try:
        return mem.get_all(filters={"user_id": USER_ID}, top_k=limit).get("results", [])
    except Exception as exc:
        logger.warning("Memory list failed: %s", exc)
        return []


def forget_all() -> None:
    mem = _get()
    if mem is not None:
        _writer.submit(mem.delete_all, user_id=USER_ID).result()


def with_memories(prompt: str, memories: list[str]) -> str:
    if not memories:
        return prompt
    facts = "\n".join(f"- {m}" for m in memories)
    return f"Things you remember about the user and their work:\n{facts}\n\n{prompt}"


def flush(timeout: float = 30.0) -> None:
    """Wait for queued writes so they aren't lost on exit."""
    done = threading.Event()
    try:
        _writer.submit(done.set)
    except RuntimeError:
        return  # interpreter exit already drained the executor
    if not done.wait(timeout):
        logger.warning("Exited with memory writes still pending")


def close() -> None:
    """Flush writes and release the on-disk store."""
    flush()
    if _mem0 is not None:
        try:
            _mem0.vector_store.client.close()
        except Exception as exc:
            logger.debug("Memory close skipped: %s", exc)
