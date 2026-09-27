"""Ares tool wrappers around Serena's key MCP tools.

Each tool delegates to the shared SerenaClient singleton, translating
the BaseTool.execute() interface into MCP stdio calls.
"""

from __future__ import annotations

from typing import Any

from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.core.types import ToolResult

from ares.serena_client import get_serena_client


class _SerenaTool(BaseTool):
    tool_id: str
    is_local: bool = True

    def _call(self, mcp_name: str, **kwargs: Any) -> ToolResult:
        try:
            content = get_serena_client().call_tool(mcp_name, {k: v for k, v in kwargs.items() if v is not None})
            return ToolResult(tool_name=self.spec.name, content=content, success=True)
        except Exception as exc:
            return ToolResult(tool_name=self.spec.name, content=str(exc), success=False)


class FindSymbolTool(_SerenaTool):
    tool_id = "find_symbol"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="find_symbol",
            description="Find a symbol (function, class, variable) by name or pattern in the codebase. Returns file locations and signatures.",
            parameters={
                "type": "object",
                "properties": {
                    "name_or_pattern": {"type": "string", "description": "Symbol name or regex pattern."},
                    "substring_matching": {"type": "boolean", "description": "Match substrings. Default false."},
                },
                "required": ["name_or_pattern"],
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
                    "symbol_name": {"type": "string", "description": "Name of the symbol to find references for."},
                    "relative_path": {"type": "string", "description": "File that contains the symbol definition."},
                },
                "required": ["symbol_name", "relative_path"],
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
            description="Replace a function or class body by symbol name — more reliable than line-based editing, works across file sizes.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string", "description": "Name of the function/class to replace."},
                    "relative_path": {"type": "string", "description": "File containing the symbol."},
                    "new_body": {"type": "string", "description": "New body content (indented, excluding the def/class header line)."},
                },
                "required": ["symbol_name", "relative_path", "new_body"],
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
                    "symbol_name": {"type": "string", "description": "Symbol after which to insert code."},
                    "relative_path": {"type": "string", "description": "File containing the symbol."},
                    "new_content": {"type": "string", "description": "Code to insert."},
                },
                "required": ["symbol_name", "relative_path", "new_content"],
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
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "relative_path": {"type": "string", "description": "Scope to this path. Omit to search whole project."},
                },
                "required": ["pattern"],
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
                    "symbol_name": {"type": "string", "description": "Current symbol name."},
                    "relative_path": {"type": "string", "description": "File containing the symbol definition."},
                    "new_name": {"type": "string", "description": "New name for the symbol."},
                },
                "required": ["symbol_name", "relative_path", "new_name"],
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
