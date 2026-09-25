"""Decorator binding a Plugin method to one or more hooks."""

from typing import TYPE_CHECKING, Callable

from arancio.core.hooks.types import Hook

if TYPE_CHECKING:
    from arancio.core.plugins.base import Plugin


def hook(event: Hook) -> Callable[[Callable], Callable]:
    """Bind a Plugin instance method to one hook.

    Stack the decorator to bind the same method to several hooks. The method
    is called with the hook's own keyword arguments only.

    Args:
        event: the hook the method runs on.

    Returns:
        A decorator adding ``event`` to the hooks stored on the wrapped method.
    """

    def decorator(func: Callable) -> Callable:
        func.__arancio_hooks__ = (*getattr(func, "__arancio_hooks__", ()), event)
        return func

    return decorator


def find_hook_methods(
    plugin_cls: type["Plugin"],
) -> list[tuple[Callable, tuple[Hook, ...]]]:
    """Find every @hook-decorated method a plugin class defines directly.

    Args:
        plugin_cls: the plugin class to scan.

    Returns:
        ``(function, hooks)`` pairs, in definition order. Only members
        defined directly on ``plugin_cls`` count, like
        :func:`~arancio.core.plugins.tool.find_tool_specs`.
    """
    return [
        (func, func.__arancio_hooks__)
        for func in vars(plugin_cls).values()
        if callable(func) and hasattr(func, "__arancio_hooks__")
    ]
