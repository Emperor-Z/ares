# Adapted from OpenJarvis (Apache-2.0) and modified for Ares. See NOTICE.
"""Shell, file and calculator tools for the coder, runner and thinker."""

from __future__ import annotations

import ast
import fnmatch
import math
import operator
import os
import subprocess
from pathlib import Path
from typing import Any

from ares.runtime.tools import BaseTool, ToolSpec
from ares.runtime.types import ToolResult

# Files agents may neither read nor write: secrets, credentials, keys.
SENSITIVE_PATTERNS = frozenset({
    ".env", ".env.*", "*.env", ".secret", "*.secrets", "credentials.*",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks",
    "id_rsa", "id_ed25519", ".htpasswd", ".pgpass", ".netrc",
})


def is_sensitive_file(path: str | Path) -> bool:
    p = Path(path)
    return any(fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(str(p), pat) for pat in SENSITIVE_PATTERNS)


def _ok(tool: str, content: str, **metadata: Any) -> ToolResult:
    return ToolResult(tool_name=tool, content=content, success=True, metadata=metadata)


def _fail(tool: str, content: str, **metadata: Any) -> ToolResult:
    return ToolResult(tool_name=tool, content=content, success=False, metadata=metadata)


# ── shell ─────────────────────────────────────────────────────────────────────

_MAX_OUTPUT = 102_400
_MAX_TIMEOUT = 300
# Commands see only these variables, so API keys in Ares's environment stay out of reach.
_ENV_KEYS = ("PATH", "HOME", "USER", "LANG", "TERM")


class ShellExecTool(BaseTool):
    """Runs a shell command, after the user approves it (see ares.runtime.tools)."""

    tool_id = "shell_exec"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell_exec",
            description="Execute a shell command and return its stdout/stderr. "
                        "The user approves each command before it runs.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 300)."},
                    "working_dir": {"type": "string", "description": "Directory to run in. Must exist."},
                },
                "required": ["command"],
            },
            category="system",
            timeout_seconds=_MAX_TIMEOUT + 10,  # the command's own timeout fires first
            requires_confirmation=True,
        )

    def describe_call(self, params: dict[str, Any]) -> str:
        where = f"  (in {params['working_dir']})" if params.get("working_dir") else ""
        return f"$ {params.get('command', '')}{where}"

    def execute(self, **params: Any) -> ToolResult:
        command = params.get("command", "")
        if not command:
            return _fail("shell_exec", "No command provided.")
        try:
            timeout = min(max(int(params.get("timeout", 30)), 1), _MAX_TIMEOUT)
        except (TypeError, ValueError):
            timeout = 30
        cwd = params.get("working_dir")
        if cwd is not None and not Path(cwd).is_dir():
            return _fail("shell_exec", f"Working directory does not exist: {cwd}")
        env = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
        try:
            proc = subprocess.run(command, shell=True, capture_output=True, text=True,
                                  timeout=timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired:
            return _fail("shell_exec", f"Command timed out after {timeout} seconds.", returncode=-1)
        except OSError as exc:
            return _fail("shell_exec", f"OS error: {exc}")
        sections = []
        for label, text in (("STDOUT", proc.stdout), ("STDERR", proc.stderr)):
            if text:
                if len(text) > _MAX_OUTPUT:
                    text = text[:_MAX_OUTPUT] + f"\n... ({label.lower()} truncated)"
                sections.append(f"=== {label} ===\n{text}")
        return ToolResult(tool_name="shell_exec", content="\n".join(sections) or "(no output)",
                          success=proc.returncode == 0, metadata={"returncode": proc.returncode})


# ── files ─────────────────────────────────────────────────────────────────────

_MAX_READ = 1_048_576
_MAX_WRITE = 10_485_760


_MAX_ENTRIES = 200


def _list_dir(path: Path) -> ToolResult:
    """Small models often file_read a directory; give them its listing."""
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError as exc:
        return _fail("file_read", f"Read error: {exc}")
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries[:_MAX_ENTRIES]]
    if len(entries) > _MAX_ENTRIES:
        lines.append(f"... and {len(entries) - _MAX_ENTRIES} more")
    header = f"Directory {path.resolve()} ({len(entries)} entries):"
    return _ok("file_read", "\n".join([header, *lines]), path=str(path.resolve()), entries=len(entries))


def _not_found(file_path: str, path: Path) -> str:
    cwd = Path.cwd()
    message = f"Not found: {file_path}. The working directory is {cwd}."
    # "/ares/runtime" when the model meant "ares/runtime".
    if path.is_absolute() and (cwd / file_path.lstrip("/")).exists():
        message += f" Did you mean the relative path {file_path.lstrip('/')}?"
    return message


class FileReadTool(BaseTool):
    tool_id = "file_read"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description="Read the contents of a file, or list the entries of a directory. "
                        "Relative paths start from the working directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file or directory."},
                    "max_lines": {"type": "integer", "description": "Max lines to return (default: all)."},
                },
                "required": ["path"],
            },
            category="filesystem",
        )

    def execute(self, **params: Any) -> ToolResult:
        file_path = params.get("path", "")
        if not file_path:
            return _fail("file_read", "No path provided.")
        path = Path(file_path).expanduser()
        if is_sensitive_file(path):
            return _fail("file_read", f"Access denied: {file_path} is a sensitive file.")
        if path.is_dir():
            return _list_dir(path)
        if not path.is_file():
            return _fail("file_read", _not_found(file_path, path))
        size = path.stat().st_size
        if size > _MAX_READ:
            return _fail("file_read", f"File too large: {size} bytes (max {_MAX_READ}).")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _fail("file_read", f"Read error: {exc}")
        max_lines = params.get("max_lines")
        if isinstance(max_lines, int) and max_lines > 0:
            text = "".join(text.splitlines(keepends=True)[:max_lines])
        return _ok("file_read", text, path=str(path.resolve()), size_bytes=size)


class FileWriteTool(BaseTool):
    tool_id = "file_write"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_write",
            description="Write content to a file. Supports write and append modes.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Content to write to the file."},
                    "mode": {"type": "string", "description": "'write' (overwrite/create) or 'append'. Default: 'write'."},
                    "create_dirs": {"type": "boolean", "description": "Create missing parent directories. Default: false."},
                },
                "required": ["path", "content"],
            },
            category="filesystem",
        )

    def execute(self, **params: Any) -> ToolResult:
        file_path = params.get("path", "")
        content = params.get("content")
        mode = params.get("mode", "write")
        if not file_path:
            return _fail("file_write", "No path provided.")
        if not isinstance(content, str):
            return _fail("file_write", "No content provided.")
        if mode not in ("write", "append"):
            return _fail("file_write", f"Invalid mode: {mode!r}. Use 'write' or 'append'.")
        path = Path(file_path).expanduser()
        if is_sensitive_file(path):
            return _fail("file_write", f"Access denied: {file_path} is a sensitive file.")
        if len(content.encode("utf-8")) > _MAX_WRITE:
            return _fail("file_write", f"Content too large (max {_MAX_WRITE} bytes).")
        try:
            if params.get("create_dirs"):
                path.parent.mkdir(parents=True, exist_ok=True)
            elif not path.parent.exists():
                return _fail("file_write", f"Parent directory does not exist: {path.parent}. "
                                           "Set create_dirs=true to create it.")
            with path.open("a" if mode == "append" else "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            return _fail("file_write", f"Write error: {exc}")
        return _ok("file_write", f"Successfully wrote to {file_path}",
                   path=str(path.resolve()), size_bytes=path.stat().st_size)


# ── calculator ────────────────────────────────────────────────────────────────

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log,
          "ln": math.log, "log10": math.log10, "log2": math.log2, "sin": math.sin, "cos": math.cos,
          "tan": math.tan, "ceil": math.ceil, "floor": math.floor}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic and a few math functions, nothing else. ^ means power."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
                raise ValueError("exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"unsupported expression: {ast.unparse(node)}")

    return float(ev(tree.body))


class CalculatorTool(BaseTool):
    tool_id = "calculator"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Evaluate a mathematical expression safely. Supports arithmetic, "
                        "math functions (sqrt, log, sin, cos), and constants.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate (e.g. '2+3*4', 'sqrt(16)')"},
                },
                "required": ["expression"],
            },
            category="math",
        )

    def execute(self, **params: Any) -> ToolResult:
        expression = params.get("expression", "")
        if not expression:
            return _fail("calculator", "No expression provided.")
        try:
            return _ok("calculator", str(safe_eval(expression)))
        except ZeroDivisionError:
            return _fail("calculator", "Error: division by zero")
        except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
            return _fail("calculator", f"Error: {exc}")


def standard_tools() -> list[BaseTool]:
    return [ShellExecTool(), FileReadTool(), FileWriteTool(), CalculatorTool()]
