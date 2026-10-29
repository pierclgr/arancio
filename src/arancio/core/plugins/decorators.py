"""Decorators recording plugin, hook and tool declarations outside their objects."""

from inspect import getfile, isfunction
from pathlib import Path
from typing import Any, Callable

from arancio.core.hooks.base import Hook
from arancio.core.messages.base import ErrorMessage, WarningMessage
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.registry import (
    PENDING_HOOKS,
    PENDING_TOOLS,
    PLUGIN_REGISTRY,
    REGISTRATION_MESSAGES,
)
from arancio.core.plugins.tool import PluginTool


def plugin(**metadata: Any) -> Callable[[type], type]:
    """Group metadata and method declarations without changing the class.

    Invalid metadata reports a startup error and skips only this class.
    Empty declarations register with a warning. No instances are constructed,
    and no attributes are added to the class or its methods.

    Args:
        **metadata: the plugin's name, version, description, author and enabled
            flag. Omitted fields take their defaults from Plugin.

    Returns:
        A decorator returning the original class unchanged.
    """

    def decorate(plugin_cls: type) -> type:
        source_path = Path(getfile(plugin_cls))
        try:
            record = Plugin(
                plugin_cls=plugin_cls,
                source_path=source_path,
                **{"name": plugin_cls.__name__, **metadata},
            )
        except (TypeError, ValueError) as exc:
            REGISTRATION_MESSAGES.append(
                ErrorMessage(
                    content=(
                        f"plugins/{source_path.parent.name}/{source_path.name}:"
                        f"{plugin_cls.__name__}: invalid metadata {metadata!r}: {exc}"
                    )
                )
            )
            return plugin_cls

        for func in vars(plugin_cls).values():
            if not isfunction(func):
                continue
            if func in PENDING_HOOKS:
                record.hooks.append((func, PENDING_HOOKS[func]))
            if func in PENDING_TOOLS:
                record.tools.append(
                    PluginTool.create_class(record.name, func, **PENDING_TOOLS[func])
                )
        PLUGIN_REGISTRY.append(record)
        if not record.hooks and not record.tools:
            REGISTRATION_MESSAGES.append(
                WarningMessage(
                    content=f"{record.location}: declares no hooks and no tools."
                )
            )
        return plugin_cls

    return decorate


def hook(event: Hook) -> Callable[[Callable], Callable]:
    """Bind a plain plugin instance method to one hook.

    Stack the decorator to bind the same method to several hooks. The method
    is called with the hook manager's shared context dict as its only
    argument. A returned message is forwarded through the normal message
    stream; other return values are ignored.

    Args:
        event: the hook the method runs on.

    Returns:
        A decorator recording ``event`` externally and returning the method unchanged.
    """

    def decorator(func: Callable) -> Callable:
        PENDING_HOOKS[func] = (*PENDING_HOOKS.get(func, ()), event)
        return func

    return decorator


def tool(
    *,
    description: str,
    input_schema: dict[str, Any],
    category: str = PluginTool.category,
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Mark a plain plugin instance method as a tool the model can call.

    Args:
        description: the tool's natural-language description.
        input_schema: JSON Schema for the tool's arguments.
        category: ``"READ"``/``"WRITE"``/``"WEB"``/``"EXECUTE"``/``"PLUGIN"``
            (the default) or ``"CUSTOM"`` — which creates a
            ``PLUGIN:<plugin_name>`` category dedicated to the owning
            plugin, shared by every ``CUSTOM`` tool it defines, defaulting to
            :attr:`~arancio.core.permissions.types.PermissionLevel.ASK`.
        name: the tool portion of the exposed name; defaults to the method's
            own name. The final name is ``<plugin_name>-<name>``.

    Returns:
        A decorator recording tool fields externally and returning the method unchanged.
    """

    def decorator(func: Callable) -> Callable:
        PENDING_TOOLS[func] = {
            "name": name or func.__name__,
            "description": description,
            "input_schema": input_schema,
            "category": category.upper(),
        }
        return func

    return decorator
