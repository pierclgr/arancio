"""Tool definitions registered with the tool manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from arancio.core.permissions.types import PermissionCategory
    from arancio.core.tools.base import Tool


@dataclass(frozen=True)
class ToolDefinition:
    """Catalog entry describing a tool that the tool manager can build.

    Attributes:
        name: unique tool identifier, equal to the built tool's ``name``.
        category: the permission category whose grant enables the tool.
        factory: builds the tool when called with the keyword arguments
            ``hook_manager`` and ``tool_session``; a tool class is a valid factory.
    """

    name: str
    category: PermissionCategory
    factory: Callable[..., Tool]
