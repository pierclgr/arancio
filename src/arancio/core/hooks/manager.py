"""Synchronous named-argument injection and dispatch for hooks."""

from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any

from arancio.core.hooks.base import Hook
from arancio.core.messages.base import Message


class HookManager:
    """Inject named services and event arguments into synchronous handlers.

    Handlers run in registration order over a snapshot. Mutable arguments keep their
    identity; event arguments never survive their dispatch. Returned messages are
    collected, other returns ignored, and ordinary handler failures propagate.
    """

    def __init__(self) -> None:
        """Initialize independent registrations and permanent services."""
        self._dependencies: dict[str, Any] = {"hook_manager": self}
        self._handlers: dict[Hook, list[tuple[Callable, tuple[str, ...], bool]]] = {}

    def provide(self, **dependencies: Any) -> None:
        """Provide permanent services, replacing previous values with the same name.

        Args:
            **dependencies: services keyed by their injectable parameter names.
        """
        self._dependencies.update(dependencies)

    def register(self, hook: Hook, handler: Callable) -> None:
        """Register a handler and inspect its explicitly named parameters once.

        Args:
            hook: the hook to attach to.
            handler: callable accepting named arguments. Positional-only parameters
                are unsupported; **kwargs receives all remaining available arguments.
                Duplicate registrations each run independently.
        """
        parameters = signature(handler).parameters
        names = tuple(
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        )
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        self._handlers.setdefault(hook, []).append((handler, names, accepts_kwargs))

    def run(self, hook: Hook, **event: Any) -> list[Message]:
        """Inject services and current event arguments into registered handlers.

        Args:
            hook: the hook to dispatch.
            **event: arguments for this dispatch only, overriding matching services.

        Returns:
            Returned messages in dispatch order.
        """
        available = {**self._dependencies, **event}
        messages: list[Message] = []
        for handler, names, accepts_kwargs in list(self._handlers.get(hook, [])):
            arguments = (
                available
                if accepts_kwargs
                else {name: available[name] for name in names if name in available}
            )
            result = handler(**arguments)
            if isinstance(result, Message):
                messages.append(result)
        return messages
