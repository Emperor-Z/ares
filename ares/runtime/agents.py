# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""The three agent loops Ares uses.

ReActAgent        Thought / Action / Final Answer text protocol (coder, runner, serena).
ActionAgent       free-form answers with optional "Action:" tool calls, for models
                  without native tool calling (thinker on deepseek-r1).
ToolCallingAgent  Ollama's native function calling (orchestrator).
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

from ares.runtime.events import EventBus, EventType
from ares.runtime.loop_guard import LoopGuard, LoopGuardConfig
from ares.runtime.tools import BaseTool, ToolExecutor, describe_tools
from ares.runtime.types import AgentResult, Message, Role, ToolCall, ToolResult

_OBSERVATION_CHARS = 4000


class BaseAgent:
    agent_id = "agent"

    def __init__(self, engine: Any, model: str, *, bus: EventBus | None = None,
                 tools: list[BaseTool] | None = None, max_turns: int = 10,
                 temperature: float = 0.7, max_tokens: int = 1024,
                 agent_id: str | None = None, loop_guard: LoopGuardConfig | None = None,
                 system_prompt: str | None = None) -> None:
        self._engine = engine
        self._model = model
        self._bus = bus
        self._tools = list(tools or [])
        self._max_turns = max_turns
        self._temperature = temperature
        self._max_tokens = max_tokens
        if agent_id:
            self.agent_id = agent_id
        self._executor = ToolExecutor(self._tools, bus=bus, agent_id=self.agent_id)
        self._loop_guard = LoopGuard(loop_guard or LoopGuardConfig(), bus=bus)
        self._system_prompt = system_prompt

    def run(self, input: str) -> AgentResult:
        self._loop_guard.reset()
        self._emit(EventType.AGENT_TURN_START, input=input)
        try:
            result = self._loop(input)
        except Exception:
            self._emit(EventType.AGENT_TURN_END, error=True)
            raise
        self._emit(EventType.AGENT_TURN_END, turns=result.turns)
        return result

    def _loop(self, input: str) -> AgentResult:
        raise NotImplementedError

    # ── helpers ──────────────────────────────────────────────────────────────

    def _emit(self, event: EventType, **data: Any) -> None:
        if self._bus is not None:
            self._bus.publish(event, {"agent": self.agent_id, **data})

    def _messages(self, input: str, system_prompt: str | None) -> list[Message]:
        messages = [Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []
        messages.append(Message(role=Role.USER, content=input))
        return messages

    def _generate(self, messages: list[Message], usage: dict[str, int], **kwargs: Any) -> dict[str, Any]:
        engine_id = getattr(self._engine, "engine_id", "")
        self._emit(EventType.INFERENCE_START, model=self._model, engine=engine_id)
        result = self._engine.generate(messages, model=self._model, temperature=self._temperature,
                                       max_tokens=self._max_tokens, **kwargs)
        for key, value in result.get("usage", {}).items():
            usage[key] = usage.get(key, 0) + value
        self._emit(EventType.INFERENCE_END, model=self._model, usage=result.get("usage", {}),
                   content=result.get("content", ""), tool_calls=result.get("tool_calls", []),
                   finish_reason=result.get("finish_reason", ""))
        return result

    def _run_tool(self, call: ToolCall) -> ToolResult:
        """Execute one call unless the loop guard blocks it."""
        verdict = self._loop_guard.check_call(call.name, call.arguments)
        if verdict.blocked:
            return ToolResult(tool_name=call.name, content=f"Loop guard: {verdict.reason}", success=False)
        result = self._executor.execute(call)
        if verdict.warned:
            # The model has to see the warning, or it's just one more free call.
            result.content = (f"{result.content}\n\nLoop guard warning: {verdict.reason} "
                              "The next repeat will be blocked. Try something different or give your final answer.")
        return result

    def _result(self, content: str, tool_results: list[ToolResult], turns: int,
                usage: dict[str, int], messages: list[Message] | None = None, **meta: Any) -> AgentResult:
        metadata: dict[str, Any] = {**usage, **meta}
        if messages is not None:
            metadata["messages"] = [m.to_dict() for m in messages]
        return AgentResult(content=content, tool_results=tool_results, turns=turns, metadata=metadata)


def strip_think(text: str) -> str:
    """Remove <think> blocks, including a leading one whose opening tag is missing."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _clip(text: str) -> str:
    if len(text) > _OBSERVATION_CHARS:
        return text[:_OBSERVATION_CHARS] + "\n\n[Output truncated]"
    return text


# ── ReAct ─────────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are a ReAct agent. For each step, respond with exactly one of:

1. To think and act:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <json arguments>

2. To give a final answer:
Thought: <your reasoning>
Final Answer: <your answer>

{tool_descriptions}"""


class ReActAgent(BaseAgent):
    agent_id = "react"

    def _loop(self, input: str) -> AgentResult:
        prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=describe_tools(self._tools))
        messages = self._messages(input, prompt)
        tool_results: list[ToolResult] = []
        usage: dict[str, int] = {}

        for turn in range(1, self._max_turns + 1):
            messages = self._loop_guard.compress_context(messages)
            content = self._generate(messages, usage).get("content", "")
            parsed = parse_react(content)

            if "parse_error" in parsed:
                return self._result("Could not parse the model's ReAct response. " + parsed["parse_error"],
                                    tool_results, turn, usage, messages, response_parse_error=parsed["parse_error"])
            if parsed["final_answer"]:
                return self._result(parsed["final_answer"], tool_results, turn, usage, messages)
            if not parsed["action"]:
                return self._result(content, tool_results, turn, usage, messages)

            messages.append(Message(role=Role.ASSISTANT, content=content))
            result = self._run_tool(ToolCall(id=f"react_{turn}", name=parsed["action"],
                                             arguments=parsed["action_input"] or "{}"))
            tool_results.append(result)
            messages.append(Message(role=Role.USER, content=f"Observation: {result.content}"))

        return self._result("Maximum turns reached without a final answer.", tool_results,
                            self._max_turns, usage, messages, max_turns_exceeded=True)


_REACT_FIELDS = ("thought", "action", "action_input", "final_answer")


def parse_react(text: str) -> dict[str, str]:
    """Parse a ReAct reply, in the text format or as one JSON object."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced and fenced.group(1).lstrip().startswith(("{", "[")):
        return _parse_react_json(fenced.group(1))
    if stripped.startswith("{") or re.match(r"\[\s*\{", stripped):
        return _parse_react_json(stripped)

    result = dict.fromkeys(_REACT_FIELDS, "")
    thought = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if thought:
        result["thought"] = thought.group(1).strip()
    final = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final:
        result["final_answer"] = final.group(1).strip()
        return result
    action = re.search(r"Action:\s*(.+)", text, re.IGNORECASE)
    if action:
        result["action"] = action.group(1).strip()
    action_input = re.search(r"Action Input:\s*(.+?)(?=\n\n|\nThought:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if action_input:
        result["action_input"] = action_input.group(1).strip()
    return result


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def _parse_react_json(text: str) -> dict[str, str]:
    """A whole-reply JSON object. Anything malformed is an error, never a guessed action."""
    result = dict.fromkeys(_REACT_FIELDS, "")
    try:
        payload = json.loads(text, object_pairs_hook=_unique_keys)
        if not isinstance(payload, dict):
            raise ValueError("not an object")
        fields: dict[str, Any] = {}
        for key, value in payload.items():
            name = key.strip().lower().replace(" ", "_")
            if name in result:
                if name in fields:
                    raise ValueError("ambiguous fields")
                fields[name] = value
        if not fields:
            return result  # plain JSON answer, not a ReAct step
        for name in ("thought", "action", "final_answer"):
            value = fields.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                result[name] = value.strip()
        if result["action"] and result["final_answer"]:
            raise ValueError("both action and final answer")
        if result["final_answer"]:
            return result
        if not result["action"]:
            raise ValueError("no action or final answer")
        arguments = fields.get("action_input", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments, object_pairs_hook=_unique_keys)
        if not isinstance(arguments, dict):
            raise ValueError("action_input must be an object")
        result["action_input"] = json.dumps(arguments, ensure_ascii=False)
        return result
    except (ValueError, RecursionError):
        # Don't echo decoder messages or salvage a partial action.
        return {**dict.fromkeys(_REACT_FIELDS, ""),
                "parse_error": "Expected one JSON object with either a nonempty action and "
                               "object action_input, or a nonempty final_answer."}


# ── Action (text tool calls, no native function calling) ──────────────────────

ACTION_SYSTEM_PROMPT = """\
You are a careful assistant with access to tools. Use a tool when it would help \
answer the user's question.

## How to use tools

To call a tool, write on its own lines:

Action: <tool_name>
Action Input: <json_arguments>

You will receive the result, then continue your response.

## Available tools

{tool_descriptions}

## Rules

- When the user asks a math question, use calculator.
- When the user asks about a file, use file_read.
- If no tool is needed, respond directly with your answer.
- Your final answer must not contain an Action line."""


class ActionAgent(BaseAgent):
    """For models without native tool calling.

    Each reply either calls a tool with an "Action:" block or is the answer.
    Tools are described in the prompt and never sent as function schemas, so
    no request is wasted on a model that would reject them.
    """

    agent_id = "action"

    def _loop(self, input: str) -> AgentResult:
        prompt = ACTION_SYSTEM_PROMPT.format(tool_descriptions=describe_tools(self._tools))
        messages = self._messages(input, prompt)
        tool_results: list[ToolResult] = []
        usage: dict[str, int] = {}
        content = ""
        nudged = False
        tool_names = {t.spec.name for t in self._tools}

        for turn in range(1, self._max_turns + 1):
            messages = self._fit(self._loop_guard.compress_context(messages))
            content = strip_think(self._generate(messages, usage).get("content") or "")
            parsed = _parse_action_span(content)
            if parsed is None:
                # deepseek-r1 sometimes puts a tool call in a code block. Never run
                # code-block text (an example could be a shell command); ask once instead.
                fenced = None if nudged else fenced_tool_name(content, tool_names)
                if fenced is None:
                    return self._result(_strip_actions(content), tool_results, turn, usage)
                nudged = True
                messages.append(Message(role=Role.ASSISTANT, content=content))
                messages.append(Message(role=Role.USER, content=_FENCED_CALL_NUDGE.format(tool=fenced)))
                continue
            name, arguments, end = parsed
            # Drop anything after the call: models often go on to invent its result.
            messages.append(Message(role=Role.ASSISTANT, content=content[:end].rstrip()))
            result = self._run_tool(ToolCall(id=f"action_{turn}", name=name, arguments=arguments))
            tool_results.append(result)
            messages.append(Message(role=Role.USER, content=f"Result: {_clip(result.content)}"))

        final = _strip_actions(content) or "Maximum turns reached without a final answer."
        return self._result(final, tool_results, self._max_turns, usage, max_turns_exceeded=True)

    def _fit(self, messages: list[Message]) -> list[Message]:
        """Trim the last user message if the prompt won't fit beside the reply."""
        budget = getattr(self._engine, "num_ctx", 16384) - self._max_tokens
        estimate = sum(len(m.content or "") for m in messages) // 4 + 4 * len(messages)
        if estimate <= budget:
            return messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == Role.USER:
                text = messages[i].content or ""
                excess = (estimate - budget) * 4
                if len(text) > excess + 200:
                    messages[i] = Message(role=Role.USER,
                                          content=text[:len(text) - excess] + "\n\n[Input truncated to fit context window]")
                break
        return messages


def parse_action(text: str) -> tuple[str, str] | None:
    """(tool, JSON arguments) from an "Action:" block, or None if it's an answer."""
    parsed = _parse_action_span(text)
    return parsed[:2] if parsed else None


def _parse_action_span(text: str) -> tuple[str, str, int] | None:
    """parse_action, plus where the action ends in `text`."""
    action = re.search(r"^\s*Action:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if not action:
        return None
    name = action.group(1).strip().strip("`*").strip()
    action_input = re.search(r"Action Input:\s*", text[action.end():], re.IGNORECASE)
    if not action_input:
        return name, "{}", action.end()
    start = action.end() + action_input.end()
    rest = text[start:]
    # The first whole JSON value, which may span lines and sit in a code block.
    if rest[:1] in ('{', '[', '"'):
        try:
            value, length = json.JSONDecoder().raw_decode(rest)
            return name, json.dumps(value, ensure_ascii=False), start + length
        except ValueError:
            pass  # the executor reports it, or reads stray backslashes leniently
    block = re.match(r"(.+?)(?=\n\n|\n```|\Z)", rest, re.DOTALL)
    raw = block.group(1).strip() if block else ""
    return name, raw or "{}", start + (block.end() if block else 0)


_FENCE = re.compile(r"```[\w-]*[ \t]*\r?\n(.*?)```", re.DOTALL)

_FENCED_CALL_NUDGE = (
    "Your reply put a {tool} call in a code block, so it was not run. To use the tool, reply with "
    "only these lines:\n\nAction: {tool}\nAction Input: <json_arguments>\n\n"
    "If you don't need it, give your final answer without the code block."
)


def fenced_tool_name(text: str, tool_names: set[str]) -> str | None:
    """The tool a code block in `text` tries to call, if any.

    Matches a block that starts with a tool name (calculator "2+2") or is a JSON
    object naming one ({"name": "calculator", ...}).
    """
    for match in _FENCE.finditer(text):
        body = match.group(1).strip()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            for key in ("name", "tool", "action"):
                if payload.get(key) in tool_names:
                    return payload[key]
            continue
        first = re.match(r"[\w.-]+", body)
        if first and first.group() in tool_names:
            return first.group()
    return None


def _strip_actions(text: str) -> str:
    text = re.sub(r"^\s*Action:\s*.+?(?:Action Input:\s*.+?)?(?=\n\n|\Z)", "", text,
                  flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    return text.strip()


# ── native tool calling ───────────────────────────────────────────────────────

class ToolCallingAgent(BaseAgent):
    """Uses Ollama's function calling; runs several tool calls in parallel."""

    agent_id = "tool_calling"
    _max_continuations = 2

    def _loop(self, input: str) -> AgentResult:
        messages = self._messages(input, self._system_prompt)
        tools = self._executor.openai_tools()
        tool_results: list[ToolResult] = []
        usage: dict[str, int] = {}
        content = ""

        for turn in range(1, self._max_turns + 1):
            messages = self._loop_guard.compress_context(messages)
            result = self._generate(messages, usage, tools=tools or None)
            content = result.get("content") or ""
            raw_calls = result.get("tool_calls") or []

            if not raw_calls:
                content = strip_think(self._continue(result, messages, usage))
                return self._result(content, tool_results, turn, usage)

            calls = [ToolCall(id=tc.get("id") or f"call_{i}", name=tc.get("name", ""),
                              arguments=tc.get("arguments") or "{}")
                     for i, tc in enumerate(raw_calls)]
            messages.append(Message(role=Role.ASSISTANT, content=content, tool_calls=calls))
            for call, res in zip(calls, self._run_all(calls)):
                tool_results.append(res)
                messages.append(Message(role=Role.TOOL, content=res.content, tool_call_id=call.id, name=call.name))

        final = strip_think(content) or "Maximum turns reached without a final answer."
        return self._result(final, tool_results, self._max_turns, usage, max_turns_exceeded=True)

    def _run_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        # Loop-guard checks stay sequential and in order; only execution is parallel.
        verdicts = [self._loop_guard.check_call(c.name, c.arguments) for c in calls]
        results: list[ToolResult | None] = [
            ToolResult(tool_name=c.name, content=f"Loop guard: {v.reason}", success=False) if v.blocked else None
            for c, v in zip(calls, verdicts)
        ]
        todo = [i for i, r in enumerate(results) if r is None]
        if len(todo) == 1:
            results[todo[0]] = self._executor.execute(calls[todo[0]])
        elif todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as pool:
                for i, res in zip(todo, pool.map(lambda i: self._executor.execute(calls[i]), todo)):
                    results[i] = res
        return [r for r in results if r is not None]

    def _continue(self, result: dict[str, Any], messages: list[Message], usage: dict[str, int]) -> str:
        """If the reply was cut off by the token limit, ask the model to go on."""
        content = result.get("content") or ""
        finish = result.get("finish_reason", "")
        for _ in range(self._max_continuations):
            if finish != "length":
                break
            messages.append(Message(role=Role.ASSISTANT, content=content))
            messages.append(Message(role=Role.USER, content="Continue from where you left off."))
            more = self._generate(messages, usage)
            content += more.get("content") or ""
            finish = more.get("finish_reason", "")
        return content
