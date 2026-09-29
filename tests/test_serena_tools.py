"""Tests for Serena tool wrappers with a fake SerenaClient."""

from unittest.mock import MagicMock, patch


import pytest


@pytest.fixture(autouse=True)
def _skip_schema_check(monkeypatch):
    import ares.tools.serena_tools as st
    monkeypatch.setattr(st, "_checked", True)


def _make_fake_client(return_value="symbol found at main.py:10"):
    client = MagicMock()
    client.call_tool.return_value = return_value
    return client


def test_find_symbol_calls_mcp():
    with patch("ares.tools.serena_tools.get_serena_client", return_value=_make_fake_client("def foo at foo.py:5")):
        from ares.tools.serena_tools import FindSymbolTool
        tool = FindSymbolTool()
        result = tool.execute(name_path_pattern="foo")
        assert result.success is True
        assert "foo.py" in result.content


def test_find_symbol_mcp_error_returns_failure():
    client = MagicMock()
    client.call_tool.side_effect = RuntimeError("MCP process died")
    with patch("ares.tools.serena_tools.get_serena_client", return_value=client):
        from ares.tools.serena_tools import FindSymbolTool
        tool = FindSymbolTool()
        result = tool.execute(name_path_pattern="foo")
        assert result.success is False
        assert "MCP process died" in result.content


def test_rename_symbol_passes_args():
    client = _make_fake_client("renamed foo → bar in 3 files")
    with patch("ares.tools.serena_tools.get_serena_client", return_value=client):
        from ares.tools.serena_tools import RenameSymbolTool
        tool = RenameSymbolTool()
        tool.execute(name_path="foo", relative_path="main.py", new_name="bar")
        client.call_tool.assert_called_once_with(
            "rename_symbol",
            {"name_path": "foo", "relative_path": "main.py", "new_name": "bar"},
        )


def test_all_serena_tools_instantiate():
    with patch("ares.tools.serena_tools.get_serena_client", return_value=_make_fake_client()):
        from ares.tools.serena_tools import all_serena_tools
        tools = all_serena_tools()
        assert len(tools) == 7
        names = {t.spec.name for t in tools}
        for expected in ["find_symbol", "rename_symbol", "replace_symbol_body",
                         "get_symbols_overview", "find_referencing_symbols",
                         "insert_after_symbol", "search_for_pattern"]:
            assert expected in names


def test_none_args_excluded_from_mcp_call():
    client = _make_fake_client("ok")
    with patch("ares.tools.serena_tools.get_serena_client", return_value=client):
        from ares.tools.serena_tools import FindSymbolTool
        tool = FindSymbolTool()
        # substring_matching is optional — if not passed, should not appear in MCP call
        tool.execute(name_path_pattern="bar", substring_matching=None)
        assert "substring_matching" not in client.call_tool.call_args[0][1]


def test_client_is_started_with_a_project(monkeypatch, tmp_path):
    import ares.serena_client as sc
    monkeypatch.setattr(sc, "_client", None)
    monkeypatch.setenv("ARES_SERENA_PROJECT", str(tmp_path))
    with patch.object(sc, "SerenaClient") as fake:
        sc.get_serena_client()
    fake.assert_called_once_with(project=str(tmp_path))


def test_client_project_defaults_to_cwd(monkeypatch, tmp_path):
    import ares.serena_client as sc
    monkeypatch.setattr(sc, "_client", None)
    monkeypatch.delenv("ARES_SERENA_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    with patch.object(sc, "SerenaClient") as fake:
        sc.get_serena_client()
    fake.assert_called_once_with(project=str(tmp_path))


def test_serena_error_text_is_a_failure():
    client = _make_fake_client("Error executing tool find_symbol: 1 validation error")
    with patch("ares.tools.serena_tools.get_serena_client", return_value=client):
        from ares.tools.serena_tools import FindSymbolTool
        assert FindSymbolTool().execute(name_path_pattern="foo").success is False


# Input schemas as Serena 1.3 reports them (tools/list), trimmed to what matters.
SERENA_1_3 = {
    "find_symbol": {"properties": {"name_path_pattern": {}, "relative_path": {}, "substring_matching": {}, "include_body": {}, "depth": {}}, "required": ["name_path_pattern"]},
    "get_symbols_overview": {"properties": {"relative_path": {}, "depth": {}}, "required": ["relative_path"]},
    "find_referencing_symbols": {"properties": {"name_path": {}, "relative_path": {}}, "required": ["name_path", "relative_path"]},
    "replace_symbol_body": {"properties": {"name_path": {}, "relative_path": {}, "body": {}}, "required": ["name_path", "relative_path", "body"]},
    "insert_after_symbol": {"properties": {"name_path": {}, "relative_path": {}, "body": {}}, "required": ["name_path", "relative_path", "body"]},
    "search_for_pattern": {"properties": {"substring_pattern": {}, "relative_path": {}}, "required": ["substring_pattern"]},
    "rename_symbol": {"properties": {"name_path": {}, "relative_path": {}, "new_name": {}}, "required": ["name_path", "relative_path", "new_name"]},
}


def test_specs_match_serena_1_3():
    from ares.tools.serena_tools import schema_mismatches
    assert schema_mismatches(SERENA_1_3) == []


def test_schema_check_reports_renamed_params():
    from ares.tools.serena_tools import schema_mismatches
    renamed = {**SERENA_1_3, "rename_symbol": {"properties": {"symbol": {}, "new_name": {}}, "required": ["symbol", "new_name"]}}
    del renamed["search_for_pattern"]
    problems = schema_mismatches(renamed)
    assert any(p.startswith("rename_symbol:") and "symbol" in p for p in problems)
    assert any(p.startswith("search_for_pattern:") for p in problems)
