"""Tests for _AgentHandoffTool in orchestrator.py."""

from unittest.mock import MagicMock

from openjarvis.agents._stubs import AgentResult

from ares.agents.orchestrator import _AgentHandoffTool


def test_handoff_success():
    mock_agent = MagicMock()
    mock_agent.run.return_value = AgentResult(content="result text", turns=2)

    tool = _AgentHandoffTool("call_coder", "Coder agent", mock_agent)
    result = tool.execute(task="write a hello world script")

    assert result.success is True
    assert result.content == "result text"
    mock_agent.run.assert_called_once_with("write a hello world script")


def test_handoff_empty_task():
    mock_agent = MagicMock()
    tool = _AgentHandoffTool("call_coder", "Coder agent", mock_agent)
    result = tool.execute(task="")

    assert result.success is False
    mock_agent.run.assert_not_called()


def test_handoff_agent_exception():
    mock_agent = MagicMock()
    mock_agent.run.side_effect = RuntimeError("model timeout")

    tool = _AgentHandoffTool("call_coder", "Coder agent", mock_agent)
    result = tool.execute(task="do something")

    assert result.success is False
    assert "model timeout" in result.content


def test_tool_spec_shape():
    tool = _AgentHandoffTool("call_thinker", "Thinker", MagicMock())
    spec = tool.spec
    assert spec.name == "call_thinker"
    params = spec.parameters
    assert "task" in params["properties"]
    assert "task" in params["required"]
