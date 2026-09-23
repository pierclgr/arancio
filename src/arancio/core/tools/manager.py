"""Tool construction driven by a permission list."""

from arancio.core.clients.base import BaseClient
from arancio.core.hooks.manager import HookManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.base import BaseTool
from arancio.core.tools.web.fetch import FetchWebTool


class ToolManager:
    """Construct tool instances for a set of granted categories.

    Reads each category's tool classes from :class:`PermissionCategory`'s
    :attr:`~arancio.core.permissions.types.PermissionCategory.tools` and
    instantiates them. Every tool builds with no required arguments except
    :class:`FetchWebTool`, which receives the summarization client this
    manager holds and injects.

    Attributes:
        _web_summary_client: the client injected into :class:`FetchWebTool`.
        _hook_manager: the hook manager injected into every tool this
            manager builds.
    """

    def __init__(
        self, web_summary_client: BaseClient, hook_manager: HookManager
    ) -> None:
        """Initialize the manager with the summarization client and hooks to inject.

        Args:
            web_summary_client: the client injected into :class:`FetchWebTool`;
                held by reference so its settings can be changed in place at
                runtime and seen by the tool.
            hook_manager: the hook manager injected into every tool this
                manager builds.
        """
        self._web_summary_client = web_summary_client
        self._hook_manager = hook_manager

    @staticmethod
    def available_tools(
        permissions: dict[PermissionCategory, PermissionLevel],
    ) -> list[type[BaseTool]]:
        """Return the granted categories' tool classes without instantiating them.

        Args:
            permissions: mapping of category to level; categories at
                :attr:`PermissionLevel.NONE` are skipped.

        Returns:
            One class per tool across the granted categories, not instantiated.
        """
        return [
            tool_cls
            for category, level in permissions.items()
            if level is not PermissionLevel.NONE
            for tool_cls in category.tools
        ]

    @classmethod
    def is_tool_available(
        cls,
        tool_name: str,
        permissions: dict[PermissionCategory, PermissionLevel],
    ) -> bool:
        """Return whether a tool is among those available for the given grants.

        Args:
            tool_name: the tool's class name (``BaseTool.name``).
            permissions: mapping of category to level; categories at
                :attr:`PermissionLevel.NONE` are skipped.

        Returns:
            ``True`` when the tool's class is among the available tools for the
            given grants, ``False`` otherwise.
        """
        return any(
            tool_cls.__name__ == tool_name
            for tool_cls in cls.available_tools(permissions)
        )

    def create_tools(
        self, permissions: dict[PermissionCategory, PermissionLevel]
    ) -> list[BaseTool]:
        """Instantiate the tools for the granted categories.

        Args:
            permissions: mapping of category to level; categories at
                :attr:`PermissionLevel.NONE` are skipped.

        Returns:
            One instance per tool class across the granted categories, each
            built with this manager's hook manager injected.
        """
        tools = []
        for tool_cls in self.available_tools(permissions):
            if tool_cls is FetchWebTool:
                tools.append(
                    FetchWebTool(
                        client=self._web_summary_client,
                        hook_manager=self._hook_manager,
                    )
                )
            else:
                tools.append(tool_cls(hook_manager=self._hook_manager))
        return tools
