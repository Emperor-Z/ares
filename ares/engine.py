"""Shared OllamaEngine singleton for all Ares agents."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import httpx
from openjarvis.engine.ollama import OllamaEngine

from ares.config import OLLAMA_HOST, REQUIRED_MODELS

logger = logging.getLogger(__name__)

_engine: OllamaEngine | None = None

# Set by the REPL's live view. When present, chat calls stream and each
# content delta is passed here; otherwise the engine behaves exactly as upstream.
_token_sink: Callable[[str], None] | None = None


def set_token_sink(sink: Callable[[str], None] | None) -> None:
    global _token_sink
    _token_sink = sink


class _StreamingClient:
    """Wraps the engine's httpx client so /api/chat streams under the hood.

    OllamaEngine.generate() builds the payload and parses the reply; this
    turns the stream back into the single JSON body it expects, so none of
    that logic (tool-call parsing, retries, usage) has to be copied here.
    """

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def post(self, url: str, *, json: dict | None = None, **kwargs: Any) -> httpx.Response:
        sink = _token_sink
        if sink is None or url != "/api/chat" or json is None:
            return self._client.post(url, json=json, **kwargs)
        return _stream_chat(self._client, {**json, "stream": True}, sink)


def _stream_chat(client: httpx.Client, payload: dict, sink: Callable[[str], None]) -> httpx.Response:
    content: list[str] = []
    tool_calls: list[dict] = []
    final: dict = {}
    with client.stream("POST", "/api/chat", json=payload) as resp:
        if resp.status_code >= 400:
            resp.read()
            return httpx.Response(resp.status_code, content=resp.content, request=resp.request)
        for line in resp.iter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            msg = chunk.get("message") or {}
            if msg.get("content"):
                content.append(msg["content"])
                try:
                    sink(msg["content"])
                except Exception:
                    logger.debug("token sink failed", exc_info=True)
            tool_calls.extend(msg.get("tool_calls") or [])
            if chunk.get("done"):
                final = chunk
    body = {k: v for k, v in final.items() if k != "message"}
    body["message"] = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        body["message"]["tool_calls"] = tool_calls
    return httpx.Response(200, json=body, request=resp.request)


def get_engine() -> OllamaEngine:
    global _engine
    if _engine is None:
        _engine = OllamaEngine(host=OLLAMA_HOST)
        if not _engine.health():
            raise RuntimeError(f"Ollama not reachable at {OLLAMA_HOST}")
        _engine._client = _StreamingClient(_engine._client)
    return _engine


def check_required_models() -> list[str]:
    """Return a list of required models that are not yet pulled in Ollama.

    Logs a warning for each missing model. Does not raise — callers decide
    whether missing models are fatal.
    """
    try:
        engine = get_engine()
        available = {m.get("name", m) if isinstance(m, dict) else str(m)
                     for m in (engine.list_models() or [])}
        missing = [m for m in REQUIRED_MODELS if m not in available]
        for m in missing:
            logger.warning("Model not pulled: %s  (run: ollama pull %s)", m, m)
        return missing
    except Exception as exc:
        logger.warning("Could not check model availability: %s", exc)
        return []
