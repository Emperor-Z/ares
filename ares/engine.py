# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Ollama client shared by every Ares agent.

When the REPL's live view sets a token sink, chat calls stream and each
content delta goes to the sink as it arrives; the caller still gets one
complete reply either way.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Sequence

import httpx

from ares.config import OLLAMA_HOST, REQUIRED_MODELS
from ares.runtime.types import Message

logger = logging.getLogger(__name__)

NUM_CTX = int(os.environ.get("ARES_NUM_CTX", "16384"))
# Qwen3 soft-switch tokens. Small models sometimes emit one as a tool call's
# only argument, e.g. {"command": "/no_think"}; such calls are dropped.
_CONTROL_TOKENS = frozenset({"/think", "/no_think"})

_engine: OllamaEngine | None = None
_token_sink: Callable[[str], None] | None = None


class EngineError(RuntimeError):
    """Ollama is unreachable or rejected the request."""


def set_token_sink(sink: Callable[[str], None] | None) -> None:
    global _token_sink
    _token_sink = sink


class OllamaEngine:
    engine_id = "ollama"
    num_ctx = NUM_CTX

    def __init__(self, host: str = OLLAMA_HOST, *, timeout: float = 1800.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.host = host.rstrip("/")
        self._client = httpx.Client(base_url=self.host, timeout=timeout, transport=transport)

    def generate(self, messages: Sequence[Message], *, model: str, temperature: float = 0.7,
                 max_tokens: int = 1024, tools: list[dict] | None = None) -> dict[str, Any]:
        """One chat completion. Returns content, tool_calls, usage and finish_reason."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_ollama_message(m) for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx},
            # With thinking on, Qwen3.5 can spend the whole budget thinking
            # and return empty content. deepseek-r1 reasons in its reply anyway.
            "think": False,
        }
        if tools:
            payload["tools"] = tools
        sink = _token_sink
        try:
            status, body, data = self._chat(payload, sink)
            if status == 400 and tools:
                # The model doesn't support tool calling; ask again without tools.
                payload.pop("tools")
                status, body, data = self._chat(payload, sink)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EngineError(f"Ollama not reachable at {self.host}") from exc
        if status >= 400:
            raise EngineError(f"Ollama returned {status}: {body[:500]}")
        return _parse_reply(data, model)

    def _chat(self, payload: dict, sink: Callable[[str], None] | None) -> tuple[int, str, dict]:
        if sink is None:
            resp = self._client.post("/api/chat", json={**payload, "stream": False})
            return resp.status_code, resp.text, resp.json() if resp.status_code < 400 else {}
        return self._stream_chat({**payload, "stream": True}, sink)

    def _stream_chat(self, payload: dict, sink: Callable[[str], None]) -> tuple[int, str, dict]:
        """Stream a reply to `sink`, then reassemble it into one response body."""
        content: list[str] = []
        tool_calls: list[dict] = []
        final: dict = {}
        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code >= 400:
                resp.read()
                return resp.status_code, resp.text, {}
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
        data = {k: v for k, v in final.items() if k != "message"}
        data["message"] = {"role": "assistant", "content": "".join(content), "tool_calls": tool_calls}
        return 200, "", data

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Could not list Ollama models at %s: %s", self.host, exc)
            return []
        return [m["name"] for m in resp.json().get("models", [])]

    def health(self) -> bool:
        try:
            return self._client.get("/api/tags", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()


def _ollama_message(m: Message) -> dict[str, Any]:
    d = m.to_dict()
    # Ollama wants tool-call arguments as objects, not JSON text.
    for tc in d.get("tool_calls", []):
        fn = tc["function"]
        try:
            fn["arguments"] = json.loads(fn["arguments"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _parse_reply(data: dict, model: str) -> dict[str, Any]:
    msg = data.get("message") or {}
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    result: dict[str, Any] = {
        "content": msg.get("content") or "",
        "model": data.get("model", model),
        # "length" when num_predict cut the reply short.
        "finish_reason": data.get("done_reason") or "stop",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "tool_calls": [],
    }
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if _control_token_only(args):
            logger.warning("Dropping control-token tool call %s(%r)", fn.get("name", ""), args)
            continue
        result["tool_calls"].append({
            "id": tc.get("id") or f"call_{i}",
            "name": fn.get("name", ""),
            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
        })
    return result


def _control_token_only(args: Any) -> bool:
    """True if the arguments hold a control token and nothing else of use."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    if isinstance(args, str):
        return args.strip().lower() in _CONTROL_TOKENS
    if not isinstance(args, dict):
        return False
    seen = False
    for value in args.values():
        if not isinstance(value, str):
            return False  # real content
        value = value.strip().lower()
        if not value:
            continue
        if value not in _CONTROL_TOKENS:
            return False
        seen = True
    return seen


def get_engine() -> OllamaEngine:
    global _engine
    if _engine is None:
        engine = OllamaEngine()
        if not engine.health():
            raise EngineError(f"Ollama not reachable at {OLLAMA_HOST}")
        _engine = engine
    return _engine


def check_required_models() -> list[str]:
    """Return the required models that aren't pulled yet, warning about each."""
    try:
        available = set(get_engine().list_models())
    except Exception as exc:
        logger.warning("Could not check model availability: %s", exc)
        return []
    missing = [m for m in REQUIRED_MODELS if m not in available]
    for m in missing:
        logger.warning("Model not pulled: %s  (run: ollama pull %s)", m, m)
    return missing
