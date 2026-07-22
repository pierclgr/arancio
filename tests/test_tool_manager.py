"""Tests for ToolManager: constructing tools from granted categories."""

from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.manager import ToolManager
from arancio.core.tools.web.fetch import FetchWebTool


def _summary_client() -> LiteLLMClient:
    """Build a summarization client for :class:`FetchWebTool` in tests.

    Returns:
        A non-streaming :class:`LiteLLMClient` suitable for injection.
    """
    return LiteLLMClient(model_id="ollama_chat/deepseek-v4-flash:cloud", stream=False)


def test_create_tools_empty_returns_no_tools() -> None:
    """No grants yields no tools."""
    assert ToolManager(web_summary_client=_summary_client()).create_tools({}) == []


def test_create_tools_constructs_granted_category_tools() -> None:
    """A granted category builds exactly its tool classes."""
    tools = ToolManager(web_summary_client=_summary_client()).create_tools(
        {PermissionCategory.READ: PermissionLevel.ASK}
    )

    assert {tool.name for tool in tools} == {"ReadFileTool"}


def test_create_tools_multiple_categories_unions_tools() -> None:
    """Multiple categories build the union of their tools."""
    tools = ToolManager(web_summary_client=_summary_client()).create_tools(
        {
            PermissionCategory.READ: PermissionLevel.ASK,
            PermissionCategory.EXECUTE: PermissionLevel.AUTO,
        }
    )

    assert {tool.name for tool in tools} == {
        "ReadFileTool",
        "ShellCommandTool",
    }


def test_create_tools_web_injects_summary_client_by_reference() -> None:
    """WEB builds SearchWebTool and FetchWebTool, injecting the summary client."""
    summary_client = _summary_client()
    tools = ToolManager(web_summary_client=summary_client).create_tools(
        {PermissionCategory.WEB: PermissionLevel.ASK}
    )

    assert {tool.name for tool in tools} == {"SearchWebTool", "FetchWebTool"}
    fetch = next(tool for tool in tools if isinstance(tool, FetchWebTool))
    assert fetch.client is summary_client


def test_available_tools_returns_classes_without_instantiating() -> None:
    """available_tools returns granted categories' tool classes, uninstantiated."""
    classes = ToolManager.available_tools(
        {PermissionCategory.READ: PermissionLevel.ASK}
    )

    assert all(isinstance(cls, type) for cls in classes)
    names = {cls.__name__ for cls in classes}
    assert names == {"ReadFileTool"}


def test_available_tools_empty_returns_no_classes() -> None:
    """No grants yields no available tool classes."""
    assert ToolManager.available_tools({}) == []


def test_is_tool_available() -> None:
    """is_tool_available is True for available tools, False otherwise."""
    permissions = {
        PermissionCategory.READ: PermissionLevel.ASK,
        PermissionCategory.WRITE: PermissionLevel.NONE,
    }

    assert ToolManager.is_tool_available("ReadFileTool", permissions) is True
    assert ToolManager.is_tool_available("WriteFileTool", permissions) is False
    assert ToolManager.is_tool_available("EchoTool", permissions) is False


def test_available_tools_skips_none_categories() -> None:
    """A category at NONE contributes no tool classes."""
    classes = ToolManager.available_tools(
        {
            PermissionCategory.READ: PermissionLevel.NONE,
            PermissionCategory.WRITE: PermissionLevel.AUTO,
        }
    )

    assert {cls.__name__ for cls in classes} == {"WriteFileTool", "EditFileTool"}


def test_create_tools_skips_none_categories() -> None:
    """A category at NONE builds no tools."""
    tools = ToolManager(web_summary_client=_summary_client()).create_tools(
        {
            PermissionCategory.READ: PermissionLevel.NONE,
            PermissionCategory.WRITE: PermissionLevel.AUTO,
        }
    )

    assert {tool.name for tool in tools} == {"WriteFileTool", "EditFileTool"}


def test_tool_repr_default_and_fetch_override() -> None:
    """BaseTool reprs as ``ClassName()``; FetchWebTool includes its client."""
    summary_client = _summary_client()
    tools = {
        tool.name: tool
        for tool in ToolManager(web_summary_client=summary_client).create_tools(
            {
                PermissionCategory.READ: PermissionLevel.ASK,
                PermissionCategory.WEB: PermissionLevel.ASK,
            }
        )
    }

    assert repr(tools["ReadFileTool"]) == "ReadFileTool()"
    assert repr(tools["FetchWebTool"]) == f"FetchWebTool(client={summary_client!r})"
