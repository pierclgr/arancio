"""OpenAI payload implementation."""

import json
from typing import Any

from codo.builders.payload.base import BasePayloadBuilder
from codo.types.messages import (
    AssistantMessage,
    Message,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)
from codo.types.requests import OpenAIRequest
from codo.types.tools import ToolSchema


class OpenAIPayloadBuilder(BasePayloadBuilder):
    """Build OpenAI Responses API payloads from generic client requests."""

    @classmethod
    def build(cls, request: OpenAIRequest) -> dict[str, Any]:
        """Build an OpenAI Responses payload from the generic request model.

        Args:
            request: the generic request to map.

        Returns:
            The OpenAI Responses payload.

        Raises:
            ValueError: when the request has an empty message list.
        """
        if not request.message_list:
            raise ValueError("message_list must contain at least one item")

        return {
            "model": request.model_id,
            "instructions": request.system_prompt,
            "tools": [cls._build_tool(tool) for tool in request.tool_list],
            "input": [
                cls._build_input_item(message) for message in request.message_list
            ],
            "reasoning": {
                "effort": request.thinking_effort,
                "summary": request.thinking_summary,
            },
            "include": ["reasoning.encrypted_content"],
            "stream": True,
            "store": False,
        }

    @staticmethod
    def _build_tool(tool: ToolSchema) -> dict[str, Any]:
        """Convert a tool dict to an OpenAI Responses function tool.

        Args:
            tool: the normalized tool definition to convert.

        Returns:
            The OpenAI Responses function tool dict.
        """
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    @classmethod
    def _build_input_item(cls, message: Message) -> dict[str, Any]:
        """Convert a normalized message to an OpenAI input item.

        Args:
            message: the normalized message to convert.

        Returns:
            The OpenAI Responses input item dict.

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
        """Render a tool result message as a string for OpenAI.

        Args:
            message: the tool result message to render.

        Returns:
            The rendered tool result string.
        """
        if isinstance(message, ToolErrorMessage):
            return json.dumps({"is_error": True, "output": message.output})
        if isinstance(message.output, str):
            return message.output
        return json.dumps(message.output)
