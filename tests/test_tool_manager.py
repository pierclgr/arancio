"""Tests for ToolManager: constructing tools from granted categories."""

from arancio.core.tools.manager import ToolManager
from arancio.core.tools.web.fetch import FetchWebTool
from arancio.core.types.permissions import PermissionCategory, PermissionLevel


def test_create_tools_empty_returns_no_tools() -> None:
    """No grants yields no tools."""
    assert ToolManager().create_tools({}) == []


def test_create_tools_constructs_granted_category_tools() -> None:
    """A granted category builds exactly its tool classes."""
    tools = ToolManager().create_tools({PermissionCategory.READ: PermissionLevel.ASK})

    assert {tool.name for tool in tools} == {"ReadFileTool", "GlobTool", "GrepTool"}


def test_create_tools_multiple_categories_unions_tools() -> None:
    """Multiple categories build the union of their tools."""
    tools = ToolManager().create_tools(
        {
            PermissionCategory.READ: PermissionLevel.ASK,
            PermissionCategory.EXECUTE: PermissionLevel.AUTO,
        }
    )

    assert {tool.name for tool in tools} == {
        "ReadFileTool",
        "GlobTool",
        "GrepTool",
        "BashCommandTool",
        "PowershellCommandTool",
    }


def test_create_tools_web_includes_fetch_with_client() -> None:
    """WEB builds FetchWebTool (with a client) and SearchWebTool."""
    tools = ToolManager().create_tools({PermissionCategory.WEB: PermissionLevel.ASK})

    assert {tool.name for tool in tools} == {"SearchWebTool", "FetchWebTool"}
    fetch = next(tool for tool in tools if isinstance(tool, FetchWebTool))
    assert fetch.client is not None
