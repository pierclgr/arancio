"""Decorator and machinery turning a BasePlugin method into a real tool."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from arancio.core.parsers.tool_result.plugin import PluginToolResultParser
from arancio.core.tools.base import BaseTool

if TYPE_CHECKING:
    from arancio.core.plugins.base import BasePlugin

CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class PluginToolSpec:
    """Metadata a @tool-decorated method carries for the loader to read.

    Attributes:
        name: the tool portion of the exposed name; defaults to the method's
            own name. Registration prefixes it with the plugin's name and a hyphen.
        description: the tool's natural-language description, used verbatim
            — unlike a harness ``description.md``, no dynamic-markdown
            (``<field>``/``<include>``/``<script>``) expansion is applied.
        input_schema: JSON Schema (``properties``/``required``), same shape
            as a harness ``input_schema.yml``.
        category: the permission category name, upper-cased: one of
            ``READ``/``WRITE``/``WEB``/``EXECUTE``/``PLUGIN``/``CUSTOM``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    category: str


def tool(
    *,
    description: str,
    input_schema: dict[str, Any],
    category: str = "PLUGIN",
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Mark a BasePlugin instance method as a tool the model can call.

    Args:
        description: the tool's natural-language description.
        input_schema: JSON Schema for the tool's arguments.
        category: ``"READ"``/``"WRITE"``/``"WEB"``/``"EXECUTE"``/``"PLUGIN"``
            (the default) or ``"CUSTOM"`` — which creates a
            ``PLUGIN:<PluginClassName>`` category dedicated to the owning
            plugin, shared by every ``CUSTOM`` tool it defines, defaulting to
            :attr:`~arancio.core.permissions.types.PermissionLevel.ASK`.
        name: the tool portion of the exposed name; defaults to the method's
            own name. The final name is ``<plugin_name>-<name>``.

    Returns:
        A decorator storing a :class:`PluginToolSpec` on the wrapped method.
    """

    def decorator(func: Callable) -> Callable:
        func.__arancio_tool__ = PluginToolSpec(
            name=name or func.__name__,
            description=description,
            input_schema=input_schema,
            category=category.upper(),
        )
        return func

    return decorator


def find_tool_specs(
    plugin_cls: type["BasePlugin"],
) -> list[tuple[Callable, PluginToolSpec]]:
    """Find every @tool-decorated method a plugin class defines directly.

    Args:
        plugin_cls: the plugin class to scan.

    Returns:
        ``(function, spec)`` pairs, in definition order. Only members
        defined directly on ``plugin_cls`` count, mirroring
        :meth:`~arancio.core.plugins.loader.PluginLoader._find_plugin_class`'s
        own "defined in this module" rule.
    """
    return [
        (func, func.__arancio_tool__)
        for func in vars(plugin_cls).values()
        if callable(func) and hasattr(func, "__arancio_tool__")
    ]


class PluginTool(BaseTool):
    """A tool synthesized from one @tool-decorated plugin method.

    Overrides :meth:`~arancio.core.tools.base.BaseTool._load_prompt` so the
    description and input schema come from the decorator instead of
    ``~/.arancio/harness/tools/<name>/``, since a plugin cannot ship files
    there. A concrete subclass is built per decorated method by
    :func:`build_plugin_tool_class`, with the owning plugin instance, the
    underlying function and the spec bound as class attributes, so it stays
    constructible as ``tool_cls(hook_manager=...)`` like any built-in tool.
    Its exposed name is ``<plugin_name>-<tool_name>``, using the manifest's
    name (defaulting to the folder name) and the method name or decorator override.
    Dictionary results keep their raw content and are displayed as JSON by
    :class:`~arancio.core.parsers.tool_result.plugin.PluginToolResultParser`.

    Attributes:
        _plugin: the plugin instance the underlying method is bound to.
        _method: the undecorated function, called as
            ``method(plugin, **kwargs)``.
        _spec: the tool's metadata from the ``@tool`` decorator.
    """

    _result_parser = PluginToolResultParser
    _plugin: ClassVar["BasePlugin"]
    _method: ClassVar[Callable]
    _spec: ClassVar[PluginToolSpec]

    def _load_prompt(self) -> tuple[str, dict[str, Any]]:
        """Return the description and raw input schema from the bound spec.

        Returns:
            The spec's description and raw input schema.
        """
        return self._spec.description, self._spec.input_schema

    def _call(self, **kwargs: Any) -> Any:
        """Run the underlying plugin method with the tool's arguments.

        Args:
            **kwargs: the tool arguments matching ``input_schema``.

        Returns:
            The plugin method's return value.
        """
        return self._method(self._plugin, **kwargs)


def build_plugin_tool_class(
    plugin: "BasePlugin", method: Callable, spec: PluginToolSpec
) -> type[PluginTool]:
    """Synthesize a PluginTool subclass for one @tool-decorated method.

    Args:
        plugin: the plugin instance the method is bound to.
        method: the undecorated function, called as ``method(plugin, **kwargs)``.
        spec: the tool's metadata from the @tool decorator.

    Returns:
        A fresh :class:`PluginTool` subclass named ``<plugin.name>-<spec.name>``.
    """
    return type(
        f"{plugin.name}-{spec.name}",
        (PluginTool,),
        {
            "_plugin": plugin,
            # wrapped in staticmethod so accessing it via `self._method` on a
            # PluginTool instance returns the plain function unbound, instead
            # of Python's descriptor protocol binding it to that instance
            "_method": staticmethod(method),
            "_spec": spec,
        },
    )
