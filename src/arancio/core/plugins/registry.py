"""Shared declarations and temporary state filled by plugin decorators."""

from typing import Any, Callable
from weakref import WeakKeyDictionary

from arancio.core.hooks.base import Hook
from arancio.core.messages import Message
from arancio.core.plugins.base import Plugin

PLUGIN_REGISTRY: list[Plugin] = []
REGISTRATION_MESSAGES: list[Message] = []

# method decorators run before the owning class exists; keep shared functions reusable
PENDING_HOOKS: WeakKeyDictionary[Callable, tuple[Hook, ...]] = WeakKeyDictionary()
PENDING_TOOLS: WeakKeyDictionary[Callable, dict[str, Any]] = WeakKeyDictionary()
