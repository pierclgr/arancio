"""OpenAI response parser implementation."""

import json
from collections.abc import Iterator
from typing import Any

import requests

from codo.parsers.response.base import BaseResponseParser
from codo.types.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    Message,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
)


class OpenAIResponseParser(BaseResponseParser):
    """Parse OpenAI Responses API streaming bodies into normalized messages."""

    @classmethod
    def parse(cls, response: requests.Response) -> Iterator[Message]:
        """Parse an OpenAI Responses SSE stream.

        Walks every SSE event from ``response.iter_lines()`` and yields each
        finalized output item (assistant text or function call) emitted
        as a ``response.output_item.done`` event. The codex subscription
        endpoint does not carry the full output array on the
        ``response.completed`` event, so iterating the per-item events
        is the format that works for both endpoints.

        Args:
            response: the HTTP response whose line iterator carries the
                SSE stream from the OpenAI Responses endpoint.

        Yields:
            Each normalized message extracted from the stream in order.
        """
        for event in cls._iter_events(response):
            message = cls._parse_event(event)
            if message is not None:
                yield message

    @staticmethod
    def _iter_events(response: requests.Response) -> Iterator[dict[str, Any]]:
        """Iterate decoded JSON payloads from the SSE ``data:`` lines.

        Args:
            response: the HTTP response whose line iterator carries the
                SSE stream from the OpenAI Responses endpoint.

        Yields:
            Each successfully decoded JSON object from a ``data:`` line.
            Lines that fail to decode are silently skipped.
        """
        prefix = "data: "
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith(prefix):
                continue
            try:
                yield json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                continue

    @classmethod
    def _parse_event(cls, event: dict[str, Any]) -> Message | None:
        """Convert a single OpenAI output item to a normalized message.

        Args:
            event: the SSE event from the OpenAI Responses endpoint.

        Returns:
            The normalized message, or ``None`` when the item type is not
            represented in the conversation history.
        """
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            return AssistantChunkMessage(content=delta) if delta else None
        if event_type == "response.reasoning_summary_text.delta":
            delta = event.get("delta", "")
            return ReasoningChunkMessage(content=delta, item={}) if delta else None
        if event_type == "response.output_item.done":
            item = event.get("item")
            item_type = item.get("type")
            if item_type == "reasoning":
                return ReasoningMessage(
                    item=item,
                    content=cls._render_reasoning_summary(item),
                )
            if item_type == "message":
                text = "".join(
                    part.get("text", "")
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                )
                return AssistantMessage(content=text)
            if item_type == "function_call":
                arguments = item.get("arguments")
                return ToolCallMessage(
                    id=item["call_id"],
                    name=item["name"],
                    arguments=json.loads(arguments) if arguments else {},
                    content=cls._render_tool_call_content(item),
                )
        return None

    @staticmethod
    def _render_reasoning_summary(item: dict[str, Any]) -> str:
        """Render displayable text from an OpenAI reasoning item summary.

        Args:
            item: an OpenAI reasoning output item.

        Returns:
            Summary text joined by newlines, or an empty string when unavailable.
        """
        try:
            return "\n".join(
                part.get("text", "")
                for part in item.get("summary", [])
                if part.get("type") == "summary_text" and part.get("text")
            )
        except Exception:
            return ""

    @staticmethod
    def _render_tool_call_content(item: dict[str, Any]) -> str:
        """Render displayable text from an OpenAI tool call item.

        Args:
            item: an OpenAI function call output item.

        Returns:
            Compact call-like text, or an empty string when unavailable.
        """
        try:
            arguments = item.get("arguments")
            parsed_arguments = json.loads(arguments) if arguments else {}
            return f"{item['name']}({json.dumps(parsed_arguments)})"
        except Exception:
            return ""
