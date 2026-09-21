"""Synchronous handler registration and dispatch for named hooks."""

from collections.abc import Callable
from typing import Any

from arancio.core.hooks.types import Hook


class HookManager:
    """Register handlers against named hooks and run them synchronously.

    Each instance owns its own registrations; there is no global singleton,
    so two managers never share handlers. Dispatch runs every handler
    registered for a hook on the caller's thread, in registration order,
    over a snapshot of the handler list taken when :meth:`run` starts — a
    handler that registers another handler during dispatch affects only
    later calls. Handler return values are ignored, and an exception raised
    by a handler propagates immediately, so any handlers after it do not
    run.

    Dispatched from :meth:`~arancio.core.agents.Agent.__call__`,
    :meth:`~arancio.core.agents.Agent._system_prompt`,
    :meth:`~arancio.core.tools.base.BaseTool.call` and
    :meth:`~arancio.core.permissions.manager.PermissionManager.validate`; see
    AGENTS.md's Hooks section for the full call-site and keyword-argument
    contract.

    Attributes:
        _handlers: registered handlers, keyed by hook.
    """

    def __init__(self) -> None:
        """Initialize the manager with no registered handlers."""
        self._handlers: dict[Hook, list[Callable[..., None]]] = {}

    def register(self, hook: Hook, handler: Callable[..., None]) -> None:
        """Register a handler to run whenever the given hook is dispatched.

        Args:
            hook: the hook the handler runs for.
            handler: the callable to run. Registering the same callable more
                than once adds one invocation per registration.
        """
        self._handlers.setdefault(hook, []).append(handler)

    def run(self, hook: Hook, **kwargs: Any) -> None:
        """Run every handler registered for the given hook.

        Args:
            hook: the hook to dispatch.
            **kwargs: forwarded to every handler unchanged, including object
                references. A hook with no registered handlers does
                nothing.
        """
        for handler in list(self._handlers.get(hook, [])):
            handler(**kwargs)
