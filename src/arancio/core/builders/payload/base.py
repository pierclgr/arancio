"""Abstract payload interface."""

from abc import ABC, abstractmethod
from typing import Any

from arancio.core.builders.base import Builder
from arancio.core.messages.base import Message
from arancio.core.messages.tool import ToolResultMessage
from arancio.core.requests.base import Request
from arancio.core.tools.schema import ToolSchema


class PayloadBuilder(Builder, ABC):
    """Build provider payloads from the generic client request model."""

    @classmethod
    @abstractmethod
    def build(cls, request: Request) -> dict[str, Any]:
        """Build a provider-specific payload from the generic request model.

        Args:
            request: the generic request to map.

        Returns:
            The provider-specific HTTP payload.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _build_tool(tool: ToolSchema) -> dict[str, Any]:
        """Convert a tool schema to the provider's function tool item."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    @abstractmethod
    def _build_input_item(cls, message: Message) -> dict[str, Any]:
        """Convert a normalized message to the provider's input item shape."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _render_tool_result(message: ToolResultMessage) -> str:
        """Render a tool result message as a string for the provider."""
        raise NotImplementedError("Subclasses must implement this method")
