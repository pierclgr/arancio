"""Abstract payload interface."""

from abc import ABC, abstractmethod
from typing import Any

from codo.builders.base import Builder
from codo.types.messages import Message, ToolResultMessage
from codo.types.requests import BaseRequest
from codo.types.tools import ToolSchema


class BasePayloadBuilder(Builder, ABC):
    """Build provider payloads from the generic client request model."""

    @classmethod
    @abstractmethod
    def build(cls, request: BaseRequest) -> dict[str, Any]:
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
        """Convert a tool dict to an OpenAI Responses function tool."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    @abstractmethod
    def _build_input_item(cls, message: Message) -> dict[str, Any]:
        """Convert a normalized message to an OpenAI input item."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _render_tool_result(message: ToolResultMessage) -> str:
        """Render a tool result message as a string for the provider."""
        raise NotImplementedError("Subclasses must implement this method")
