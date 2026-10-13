"""Tests for the Ollama client against a mocked /api/chat."""

import json

import httpx
import pytest

from ares import engine as engine_mod
from ares.engine import EngineError, OllamaEngine
from ares.runtime.types import Message, Role, ToolCall


def _engine(handler):
    return OllamaEngine("http://ollama", transport=httpx.MockTransport(handler))


def _msgs():
    return [Message(role=Role.USER, content="hi")]


@pytest.fixture(autouse=True)
def _no_sink():
    engine_mod.set_token_sink(None)
    yield
    engine_mod.set_token_sink(None)


def test_plain_reply_and_payload():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hello"}, "prompt_eval_count": 5,
                                         "eval_count": 2, "done_reason": "stop"})

    out = _engine(handler).generate(_msgs(), model="m", temperature=0.1, max_tokens=50)
    assert out["content"] == "hello" and out["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert seen["stream"] is False and seen["think"] is False
    assert seen["options"]["num_predict"] == 50 and seen["options"]["num_ctx"] == engine_mod.NUM_CTX


def test_tool_call_arguments_round_trip_as_objects():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "", "tool_calls": [
            {"function": {"name": "add", "arguments": {"a": 1}}}]}})

    history = [Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="c", name="add", arguments='{"a": 1}')])]
    out = _engine(handler).generate(history, model="m", tools=[{"type": "function"}])
    assert seen["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert out["tool_calls"] == [{"id": "call_0", "name": "add", "arguments": '{"a": 1}'}]


def test_control_token_tool_calls_are_dropped():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "", "tool_calls": [
            {"function": {"name": "shell_exec", "arguments": {"command": "/no_think"}}},
            {"function": {"name": "shell_exec", "arguments": {"command": "echo /no_think"}}}]}})

    calls = _engine(handler).generate(_msgs(), model="m")["tool_calls"]
    assert [json.loads(c["arguments"])["command"] for c in calls] == ["echo /no_think"]


def test_retries_without_tools_when_model_rejects_them():
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        if "tools" in payloads[-1]:
            return httpx.Response(400, text="model does not support tools")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    assert _engine(handler).generate(_msgs(), model="m", tools=[{"type": "function"}])["content"] == "ok"
    assert len(payloads) == 2


def test_http_errors_raise():
    with pytest.raises(EngineError, match="500"):
        _engine(lambda r: httpx.Response(500, text="boom")).generate(_msgs(), model="m")


def test_unreachable_raises():
    def handler(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(EngineError, match="not reachable"):
        _engine(handler).generate(_msgs(), model="m")


def test_streams_to_sink_and_reassembles():
    lines = [
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"message": {"content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]}},
        {"done": True, "done_reason": "stop", "eval_count": 3},
    ]
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(l) for l in lines).encode())

    tokens = []
    engine_mod.set_token_sink(tokens.append)
    out = _engine(handler).generate(_msgs(), model="m")
    assert seen["stream"] is True
    assert tokens == ["Hel", "lo"] and out["content"] == "Hello"
    assert out["tool_calls"][0]["name"] == "t" and out["usage"]["completion_tokens"] == 3


def test_list_models_and_health():
    e = _engine(lambda r: httpx.Response(200, json={"models": [{"name": "qwen"}]}))
    assert e.list_models() == ["qwen"] and e.health()
