"""Plugin tool metadata and execution."""

from typing import Callable, ClassVar

from arancio.core.builders.tool_input_schemas.base import InputSchemaBuilder
from arancio.core.parsers.tool_results.plugin import PluginToolResultParser
from arancio.core.tools.base import Tool

CUSTOM = "CUSTOM"


class PluginTool(Tool):
    """A tool synthesized from one @tool-decorated plugin method.

    Overrides :meth:`~arancio.core.tools.base.Tool._load_description` so the description
    comes from the decorator instead of ``~/.arancio/harness/tools/<name>.md``,
    since a plugin cannot ship files there. The plugin decorator creates a subclass
    per tool method with :meth:`create_class`, storing metadata directly as class
    attributes. The manager binds the method to the owning plugin instance, then
    registers the class as a tool definition with the tool manager, which builds it
    like any built-in tool; the tool runs, and its input schema is derived from, the
    bound method. Its exposed name is
    ``<plugin_name>-<tool_name>``, using the decorator's name (defaulting to the
    class name) and the method name or tool override. Dictionary results keep their
    raw content and are displayed as JSON by
    :class:`~arancio.core.parsers.tool_results.plugin.PluginToolResultParser`.

    Attributes:
        tool_name: the tool portion of the full name, defaulting to the method name.
        category: the uppercased permission category, defaulting to PLUGIN.
        _method: the declaring function, bound to its plugin instance during loading.
        _description: the description supplied by the decorator, without expansion.
    """

    _result_parser = PluginToolResultParser
    tool_name: ClassVar[str]
    category: ClassVar[str] = "PLUGIN"
    _method: ClassVar[Callable]
    _description: ClassVar[str]

    @classmethod
    def create_class(
        cls,
        plugin_name: str,
        method: Callable,
        *,
        name: str,
        description: str,
        category: str,
    ) -> type["PluginTool"]:
        """Create a new tool subclass for one @tool-decorated method.

        It creates a class only: it builds no tool object and registers
        nothing. The schema is derived from ``method`` right away, so a method that
        cannot be described fails here with the builder's ``ValueError`` instead of
        when the tool is built.

        Args:
            plugin_name: the owning plugin's name, used as the tool-name prefix.
            method: the original function, bound by the manager after construction.
            name: the tool portion of the exposed name.
            description: the tool description, used verbatim.
            category: the uppercased permission category.

        Returns:
            A fresh tool subclass named from ``plugin_name`` and ``name``.
        """
        # validate the signature now: a bad plugin tool is skipped at load time
        # (the caller catches the error), instead of failing when the tool is built
        InputSchemaBuilder.build(method)
        return type(
            f"{plugin_name}-{name}",
            (cls,),
            {
                "tool_name": name,
                "category": category,
                "_method": method,
                "_description": description,
            },
        )

    def _load_description(self) -> str:
        """Return the description stored on the tool class.

        Returns:
            The description declared by the decorator.
        """
        return self._description

    @property
    def _call(self) -> Callable:
        """The plugin method bound to its plugin instance, run as the tool.

        Returns:
            The bound method; its signature and docstring describe the tool's
            arguments.
        """
        return self._method
