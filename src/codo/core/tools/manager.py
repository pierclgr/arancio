"""Tool construction driven by a permission list."""

from codo.core.clients.litellm import LiteLLMClient
from codo.core.tools.base import BaseTool
from codo.core.tools.web.fetch import FetchWebTool
from codo.core.types.permissions import PermissionCategory, PermissionLevel


class ToolManager:
    """Construct tool instances for a set of granted categories.

    Reads each category's tool classes from :class:`PermissionCategory` (the
    enum value is the frozen set of classes) and instantiates them. Every tool
    builds with no required arguments except :class:`FetchWebTool`, which
    receives a default summarization client owned by this manager.

    Attributes:
        _fetch_client: default summarization client passed to
            :class:`FetchWebTool` at construction.
    """

    def __init__(self) -> None:
        """Initialize the manager and its default fetch summarization client."""
        self._fetch_client = LiteLLMClient(
            model_id="ollama_chat/deepseek-v4-flash:cloud",
            thinking_effort="low",
            stream=False,
        )
        self._fetch_client.thinking_summary = None

    def create_tools(
        self, permissions: dict[PermissionCategory, PermissionLevel]
    ) -> list[BaseTool]:
        """Instantiate the tools for the granted categories.

        Args:
            permissions: mapping of granted category to level; only the
                categories (keys) determine which tools are built.

        Returns:
            One instance per tool class across the granted categories.
        """
        tools: list[BaseTool] = []
        for category in permissions:
            for tool_cls in category.value:
                tools.append(
                    tool_cls(client=self._fetch_client)
                    if tool_cls is FetchWebTool
                    else tool_cls()
                )
        return tools
