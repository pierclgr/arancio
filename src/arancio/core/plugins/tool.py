"""Plugin tool metadata and execution."""

from typing import Any, Callable, ClassVar

from arancio.core.parsers.tool_result.plugin import PluginToolResultParser
from arancio.core.tools.base import Tool

CUSTOM = "CUSTOM"


class PluginTool(Tool):
    """A tool synthesized from one @tool-decorated plugin method.

    Overrides :meth:`~arancio.core.tools.base.Tool._load_prompt` so the
    description and input schema come from the decorator instead of
    ``~/.arancio/harness/tools/<name>/``, since a plugin cannot ship files
    there. The plugin decorator builds a subclass per tool method with metadata
    stored directly as class attributes. The manager binds the method to the
    owning plugin instance before registering the tool's permission category.
    It stays constructible as ``tool_cls(hook_manager=...)`` like any built-in tool.
    Its exposed name is ``<plugin_name>-<tool_name>``, using the decorator's
    name (defaulting to the class name) and the method name or tool override.
    Dictionary results keep their raw content and are displayed as JSON by
    :class:`~arancio.core.parsers.tool_result.plugin.PluginToolResultParser`.

    Attributes:
        tool_name: the tool portion of the full name, defaulting to the method name.
        category: the uppercased permission category, defaulting to PLUGIN.
        _method: the declaring function, bound to its plugin instance during loading.
        _description: the description supplied by the decorator, without expansion.
        _input_schema: the raw argument schema supplied by the decorator.
    """

    _result_parser = PluginToolResultParser
    tool_name: ClassVar[str]
    category: ClassVar[str] = "PLUGIN"
    _method: ClassVar[Callable]
    _description: ClassVar[str]
    _input_schema: ClassVar[dict[str, Any]]

    @classmethod
    def build(
        cls,
        plugin_name: str,
        method: Callable,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        category: str,
    ) -> type["PluginTool"]:
        """Synthesize a tool subclass for one @tool-decorated method.

        Args:
            plugin_name: the owning plugin's name, used as the tool-name prefix.
            method: the original function, bound by the manager after construction.
            name: the tool portion of the exposed name.
            description: the tool description, used verbatim.
            input_schema: the raw JSON Schema for the tool's arguments.
            category: the uppercased permission category.

        Returns:
            A fresh tool subclass named from ``plugin_name`` and ``name``.
        """
        return type(
            f"{plugin_name}-{name}",
            (cls,),
            {
                "tool_name": name,
                "category": category,
                "_method": method,
                "_description": description,
                "_input_schema": input_schema,
            },
        )

    def _load_prompt(self) -> tuple[str, dict[str, Any]]:
        """Return the description and raw input schema stored on the tool class.

        Returns:
            The declared description and raw input schema.
        """
        return self._description, self._input_schema

    def _call(self, **kwargs: Any) -> Any:
        """Run the underlying plugin method with the tool's arguments.

        Args:
            **kwargs: the tool arguments matching ``input_schema``.

        Returns:
            The plugin method's return value.
        """
        return self._method(**kwargs)
