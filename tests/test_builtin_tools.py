"""Tests for the shell, file and calculator tools."""

import pytest

from ares.tools.builtin import CalculatorTool, FileReadTool, FileWriteTool, ShellExecTool, is_sensitive_file, safe_eval


def test_shell_runs_and_reports_exit_status(tmp_path):
    ok = ShellExecTool().execute(command="echo hi; echo err >&2", working_dir=str(tmp_path))
    assert ok.success and "=== STDOUT ===\nhi" in ok.content and "=== STDERR ===\nerr" in ok.content
    assert not ShellExecTool().execute(command="exit 3").success


def test_shell_hides_secrets_in_the_environment(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-live")
    assert "sk-live" not in ShellExecTool().execute(command="env").content


def test_shell_timeout_and_bad_dir():
    assert "timed out" in ShellExecTool().execute(command="sleep 5", timeout=1).content
    assert not ShellExecTool().execute(command="ls", working_dir="/no/such/dir").success


def test_shell_needs_approval_and_shows_the_command():
    tool = ShellExecTool()
    assert tool.spec.requires_confirmation
    assert tool.describe_call({"command": "rm -rf build", "working_dir": "/p"}) == "$ rm -rf build  (in /p)"


def test_file_write_then_read(tmp_path):
    path = tmp_path / "a" / "notes.txt"
    assert not FileWriteTool().execute(path=str(path), content="x").success  # parent missing
    assert FileWriteTool().execute(path=str(path), content="one\n", create_dirs=True).success
    assert FileWriteTool().execute(path=str(path), content="two\n", mode="append").success
    assert FileReadTool().execute(path=str(path)).content == "one\ntwo\n"
    assert FileReadTool().execute(path=str(path), max_lines=1).content == "one\n"


@pytest.mark.parametrize("name", [".env", "prod.env", "id_rsa", "server.pem", "credentials.json"])
def test_sensitive_files_are_off_limits(tmp_path, name):
    path = tmp_path / name
    path.write_text("secret")
    assert is_sensitive_file(path)
    assert not FileReadTool().execute(path=str(path)).success
    assert not FileWriteTool().execute(path=str(path), content="x").success


def test_file_read_missing():
    assert "Not found: /no/such/file" in FileReadTool().execute(path="/no/such/file").content


def test_file_read_hints_relative_path(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    monkeypatch.chdir(tmp_path)
    content = FileReadTool().execute(path="/pkg").content
    assert "Did you mean the relative path pkg?" in content and str(tmp_path) in content


def test_file_read_lists_a_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("")
    result = FileReadTool().execute(path=str(tmp_path))
    assert result.success and result.content.splitlines()[1:] == ["sub/", "a.py"]


@pytest.mark.parametrize("expr,value", [("2+3*4", 14), ("sqrt(16)", 4), ("2^10", 1024), ("-pi", -3.14159)])
def test_calculator(expr, value):
    assert safe_eval(expr) == pytest.approx(value, rel=1e-4)


@pytest.mark.parametrize("expr", ["__import__('os')", "9**9**9", "open('x')", "True + 1"])
def test_calculator_rejects_unsafe(expr):
    assert not CalculatorTool().execute(expression=expr).success
