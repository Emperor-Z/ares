"""Ares tool wrappers around Serena's key MCP tools.

Each tool delegates to the shared SerenaClient singleton, translating
the BaseTool.execute() interface into MCP stdio calls.
"""

from __future__ import annotations

import logging
from typing import Any

from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.core.types import ToolResult

from ares.serena_client import get_serena_client

logger = logging.getLogger(__name__)

# Parameter names follow Serena's own tool schemas (checked against 1.3).
# check_schemas() warns at first use if a Serena upgrade renames them again.
_NAME_PATH = 'Name path: a symbol name like "AresSystem", or "AresSystem/run" for a method.'
# Serena reports tool failures as ordinary text, not JSON-RPC errors.
_SERENA_ERROR = "Error executing tool"
_checked = False


class _SerenaTool(BaseTool):
    tool_id: str
    is_local: bool = True

    def _call(self, mcp_name: str, **kwargs: Any) -> ToolResult:
        try:
            client = get_serena_client()
            _check_once(client)
            content = client.call_tool(mcp_name, {k: v for k, v in kwargs.items() if v is not None})
        except Exception as exc:
            return ToolResult(tool_name=self.spec.name, content=str(exc), success=False)
        return ToolResult(tool_name=self.spec.name, content=content, success=not content.startswith(_SERENA_ERROR))


def schema_mismatches(serena_schemas: dict[str, dict]) -> list[str]:
    """Differences between our tool specs and Serena's, one line per problem."""
    problems = []
    for tool in all_serena_tools():
        spec = tool.spec
        theirs = serena_schemas.get(spec.name)
        if theirs is None:
            problems.append(f"{spec.name}: not offered by this Serena")
            continue
        props = set(theirs.get("properties", {}))
        unknown = sorted(set(spec.parameters["properties"]) - props)
        missing = sorted(set(theirs.get("required", [])) - set(spec.parameters["properties"]))
        if unknown or missing:
            problems.append(f"{spec.name}: unknown params {unknown}, missing required {missing}")
    return problems


def _check_once(client: Any) -> None:
    global _checked
    if _checked:
        return
    _checked = True
    try:
        problems = schema_mismatches(client.list_tools())
    except Exception as exc:
        logger.debug("Serena schema check skipped: %s", exc)
        return
    for problem in problems:
        logger.warning("Serena tool mismatch, update ares/tools/serena_tools.py: %s", problem)


class FindSymbolTool(_SerenaTool):
    tool_id = "find_symbol"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="find_symbol",
            description="Find a symbol (function, class, method) by name path. Returns file locations and signatures.",
            parameters={
                "type": "object",
                "properties": {
                    "name_path_pattern": {"type": "string", "description": _NAME_PATH},
                    "relative_path": {"type": "string", "description": "Only search this file or directory. Omit to search the whole project."},
                    "substring_matching": {"type": "boolean", "description": "Match the last part as a substring, so \"Foo/get\" finds \"Foo/getValue\". Default false."},
                    "include_body": {"type": "boolean", "description": "Include the symbol's source code. Default false."},
                },
                "required": ["name_path_pattern"],
            },
            category="serena_semantic",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("find_symbol", **params)


class GetSymbolsOverviewTool(_SerenaTool):
    tool_id = "get_symbols_overview"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_symbols_overview",
            description="Get a structural outline of a file or directory — all top-level symbols, classes, functions.",
            parameters={
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "Relative path to file or directory."},
                    "depth": {"type": "integer", "description": "Recursion depth for nested symbols. Default 1."},
                },
                "required": ["relative_path"],
            },
            category="serena_semantic",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("get_symbols_overview", **params)


class FindReferencingSymbolsTool(_SerenaTool):
    tool_id = "find_referencing_symbols"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="find_referencing_symbols",
            description="Find all symbols that call or use a given symbol — essential for impact analysis before refactoring.",
            parameters={
                "type": "object",
                "properties": {
                    "name_path": {"type": "string", "description": f"The symbol to find references to. {_NAME_PATH}"},
                    "relative_path": {"type": "string", "description": "File that contains the symbol definition."},
                },
                "required": ["name_path", "relative_path"],
            },
            category="serena_semantic",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("find_referencing_symbols", **params)


class ReplaceSymbolBodyTool(_SerenaTool):
    tool_id = "replace_symbol_body"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="replace_symbol_body",
            description="Replace a function or class definition by name path — more reliable than line-based editing. Look the symbol up with find_symbol (include_body) first.",
            parameters={
                "type": "object",
                "properties": {
                    "name_path": {"type": "string", "description": f"The symbol to replace. {_NAME_PATH}"},
                    "relative_path": {"type": "string", "description": "File containing the symbol."},
                    "body": {"type": "string", "description": "The full new definition, including the def/class signature line. Excludes any docstring or comments above it."},
                },
                "required": ["name_path", "relative_path", "body"],
            },
            category="serena_semantic",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("replace_symbol_body", **params)


class InsertAfterSymbolTool(_SerenaTool):
    tool_id = "insert_after_symbol"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="insert_after_symbol",
            description="Insert new code immediately after a symbol definition. Use to add new methods, functions, or constants.",
            parameters={
                "type": "object",
                "properties": {
                    "name_path": {"type": "string", "description": f"The symbol after which to insert code. {_NAME_PATH}"},
                    "relative_path": {"type": "string", "description": "File containing the symbol."},
                    "body": {"type": "string", "description": "Code to insert, starting on the line after the symbol."},
                },
                "required": ["name_path", "relative_path", "body"],
            },
            category="serena_semantic",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("insert_after_symbol", **params)


class SearchForPatternTool(_SerenaTool):
    tool_id = "search_for_pattern"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_for_pattern",
            description="Regex search across the codebase. Returns matching lines with file paths and line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "substring_pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "relative_path": {"type": "string", "description": "Scope to this path. Omit to search whole project."},
                },
                "required": ["substring_pattern"],
            },
            category="serena_semantic",
            timeout_seconds=30.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("search_for_pattern", **params)


class RenameSymbolTool(_SerenaTool):
    tool_id = "rename_symbol"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="rename_symbol",
            description="Rename a symbol atomically across the entire codebase — all references updated in one operation.",
            parameters={
                "type": "object",
                "properties": {
                    "name_path": {"type": "string", "description": f"The symbol to rename. {_NAME_PATH}"},
                    "relative_path": {"type": "string", "description": "File containing the symbol definition."},
                    "new_name": {"type": "string", "description": "New name for the symbol."},
                },
                "required": ["name_path", "relative_path", "new_name"],
            },
            category="serena_semantic",
            timeout_seconds=60.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        return self._call("rename_symbol", **params)


def all_serena_tools() -> list[BaseTool]:
    return [
        FindSymbolTool(),
        GetSymbolsOverviewTool(),
        FindReferencingSymbolsTool(),
        ReplaceSymbolBodyTool(),
        InsertAfterSymbolTool(),
        SearchForPatternTool(),
        RenameSymbolTool(),
    ]
