"""Synchronous handler registration and dispatch for named hooks."""

from collections.abc import Callable
from typing import Any

from arancio.core.hooks.base import Hook
from arancio.core.messages.base import Message


class HookManager:
    """Register handlers against named hooks and run them synchronously.

    Each instance owns its own registrations; there is no global singleton,
    so two managers never share handlers. Dispatch runs every handler
    registered for a hook on the caller's thread, in registration order,
    over a snapshot of the handler list taken when :meth:`run` starts — a
    handler that registers another handler during dispatch affects only
    later calls. Every handler receives the same :attr:`context` dict, so a
    value one handler assigns is seen by the next handler and by the caller.
    Returned messages are collected; other return values are ignored. An
    exception raised by a handler propagates immediately, so any handlers
    after it do not run.

    Dispatched from :class:`~arancio.ui.app.App` for application lifecycle hooks,
    :meth:`~arancio.core.agents.base.Agent.__call__`,
    :meth:`~arancio.core.tools.base.Tool.call` and
    :meth:`~arancio.core.permissions.manager.PermissionManager.validate`; see
    AGENTS.md's Hooks section for the full call-site and keyword-argument
    contract.

    Attributes:
        context: the dict every handler receives. It starts with
            ``hook_manager`` (this manager); the core components built with
            this manager add themselves to it (``permission_manager``,
            ``agent``, ``client``), and each
            :meth:`run` writes its event keys, which stay until a later
            dispatch overwrites them.
        _handlers: registered handlers, keyed by hook.
    """

    def __init__(self) -> None:
        """Initialize the manager with no handlers and itself in the context."""
        self.context: dict[str, Any] = {"hook_manager": self}
        self._handlers: dict[
            Hook, list[Callable[[dict[str, Any]], Message | None]]
        ] = {}

    def register(
        self, hook: Hook, handler: Callable[[dict[str, Any]], Message | None]
    ) -> None:
        """Register a handler to run whenever the given hook is dispatched.

        Args:
            hook: the hook the handler runs for.
            handler: the callable to run, called with :attr:`context`.
                Registering the same callable more than once adds one
                invocation per registration.
        """
        self._handlers.setdefault(hook, []).append(handler)

    def run(self, hook: Hook, **event: Any) -> list[Message]:
        """Write the event into the context and run every handler for the hook.

        Args:
            hook: the hook to dispatch.
            **event: the event's own data, written into :attr:`context` under
                these keys before any handler runs, including object
                references. A hook with no registered handlers still writes
                them.

        Returns:
            Handler messages in dispatch order.
        """
        self.context.update(event)
        messages: list[Message] = []
        for handler in list(self._handlers.get(hook, [])):
            result = handler(self.context)
            if isinstance(result, Message):
                messages.append(result)
        return messages
