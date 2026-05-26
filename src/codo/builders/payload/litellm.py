"""LiteLLM payload implementation."""

import json
from typing import Any

from codo.builders.payload.base import BasePayloadBuilder
from codo.types.messages import (
    AssistantMessage,
    Message,
    ReasoningMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from codo.types.requests import LiteLLMRequest
from codo.types.tools import ToolSchema


class LiteLLMPayloadBuilder(BasePayloadBuilder):
    """Build kwargs payloads for ``litellm.responses`` from generic requests."""

    @classmethod
    def build(cls, request: LiteLLMRequest) -> dict[str, Any]:
        """Build a ``litellm.responses`` kwargs dict from the generic request model.

        Args:
            request: the generic request to map.

        Returns:
            The kwargs dict to forward to ``litellm.responses``. The
            ``api_key`` and any client-level overrides (e.g.
            ``max_output_tokens``) are added by the client itself
            outside the payload-building step.
        """
        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "input": [cls._build_input_item(m) for m in request.message_list],
        }
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt
        if request.tool_list:
            kwargs["tools"] = [cls._build_tool(tool) for tool in request.tool_list]
        if request.thinking_effort is not None:
            reasoning: dict[str, Any] = {"effort": request.thinking_effort}
            if request.thinking_summary is not None:
                reasoning["summary"] = request.thinking_summary
            kwargs["reasoning"] = reasoning
        return kwargs

    @staticmethod
    def _build_tool(tool: ToolSchema) -> dict[str, Any]:
        """Convert a normalized tool schema to a Responses function tool.

        Args:
            tool: the normalized tool definition.

        Returns:
            The Responses-format function tool dict.
        """
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    @classmethod
    def _build_input_item(cls, message: Message) -> dict[str, Any]:
        """Convert a normalized message to a Responses input item.

        ``ReasoningMessage`` items are passed through as their stored
        provider-native dict so the model receives back exactly what it
        emitted on the previous turn.

        Args:
            message: the normalized message to convert.

        Returns:
            The Responses input item dict.

        Raises:
            ValueError: when the message type is not supported.
        """
        if isinstance(message, (UserMessage, AssistantMessage)):
            return {"role": message.role, "content": message.content}
        if isinstance(message, ReasoningMessage):
            return dict(message.item)
        if isinstance(message, ToolCallMessage):
            return {
                "type": "function_call",
                "call_id": message.id,
                "name": message.name,
                "arguments": json.dumps(message.arguments),
            }
        if isinstance(message, ToolResultMessage):
            return {
                "type": "function_call_output",
                "call_id": message.id,
                "output": cls._render_tool_result(message),
            }
        raise ValueError(f"Unsupported message type: {type(message).__name__!r}")

    @staticmethod
    def _render_tool_result(message: ToolResultMessage) -> str:
        """Render a tool result message as a string for the provider.

        Args:
            message: the tool result message to render.

        Returns:
            The rendered tool result string.
        """
        if isinstance(message.output, str):
            return message.output
        return json.dumps(message.output)
