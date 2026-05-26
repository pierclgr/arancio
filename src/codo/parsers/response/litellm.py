"""LiteLLM response parser implementation."""

import json
from collections.abc import Iterable, Iterator
from typing import Any, List

from codo.parsers.base import Parser
from codo.types.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    Message,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
)


class LiteLLMResponseParser(Parser):
    """Parse a ``litellm.responses`` object into normalized messages."""

    @classmethod
    def parse(cls, response: Any) -> Iterator[Message]:
        """Walk the LiteLLM response ``output`` list and yield messages.

        Each output item is mapped to the matching :class:`Message`
        subclass. Items with no normalized representation are skipped.

        Args:
            response: the ``ResponsesAPIResponse`` returned by
                ``litellm.responses``.

        Yields:
            Each assistant, tool-call or reasoning message extracted
            from the response, in output order.
        """
        for item in getattr(response, "output", None) or []:
            item_dict = cls._to_dict(item)
            parsed = cls._parse_output_item(item_dict)
            if parsed is not None:
                yield parsed

    @classmethod
    def parse_stream(cls, stream: Iterable[Any]) -> Iterator[Message]:
        """Walk a LiteLLM streaming event iterator and yield messages.

        Three event types are mapped to normalized messages; the rest
        are skipped:

        - ``response.output_text.delta`` →
          :class:`AssistantChunkMessage` with the text fragment.
        - ``response.reasoning_summary_text.delta`` →
          :class:`ReasoningChunkMessage` with the summary fragment.
        - ``response.output_item.done`` → finalized message for the
          completed output item (reasoning, assistant, or tool call),
          built via :meth:`_parse_output_item`.

        Args:
            stream: the iterator of Responses streaming events returned
                by ``litellm.responses(..., stream=True)``.

        Yields:
            Each chunk and finalized message extracted from the stream,
            in event order.
        """
        for event in stream:
            event_type = cls._event_type(event)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                yield AssistantChunkMessage(content=delta)
            elif event_type == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", "") or ""
                yield ReasoningChunkMessage(content=delta, item={})
            elif event_type == "response.output_item.done":
                item_dict = cls._to_dict(getattr(event, "item", {}))
                parsed = cls._parse_output_item(item_dict)
                if parsed is not None:
                    yield parsed

    @staticmethod
    def _event_type(event: Any) -> str | None:
        """Return the event type string, unwrapping enum values.

        Args:
            event: a Responses streaming event.

        Returns:
            The dotted event type (e.g. ``"response.output_text.delta"``),
            or ``None`` when the event has no recognizable type.
        """
        value = getattr(event, "type", None)
        return getattr(value, "value", value)

    @staticmethod
    def _to_dict(item: Any) -> dict[str, Any]:
        """Coerce a Responses output item to a plain dict.

        Handles both pydantic models (``model_dump``) and dict inputs.

        Args:
            item: the output item, typically a pydantic model.

        Returns:
            The plain dict representation of the item.
        """
        if isinstance(item, dict):
            return item
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            return dump()
        return dict(item)

    @classmethod
    def _parse_output_item(cls, item: dict[str, Any]) -> Message | None:
        """Convert one Responses output item to a normalized message.

        Args:
            item: a single entry from the response ``output`` list.

        Returns:
            The normalized message, or ``None`` when the item type has
            no representation in conversation history.
        """
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
            arguments_raw = item.get("arguments")
            arguments = json.loads(arguments_raw) if arguments_raw else {}
            name = item["name"]
            return ToolCallMessage(
                id=item["call_id"],
                name=name,
                arguments=arguments,
                content=f"{name}({json.dumps(arguments)})",
            )
        return None

    @staticmethod
    def _render_reasoning_summary(item: dict[str, Any]) -> str:
        """Render displayable text from a Responses reasoning item.

        OpenRouter's Responses API preserves the upstream provider's
        native reasoning shape: OpenAI puts text in ``summary[]`` as
        ``summary_text`` parts, Anthropic puts it in ``content[]`` as
        ``reasoning_text`` parts. Walks both arrays and joins every
        text fragment found.

        Args:
            item: a Responses reasoning output item.

        Returns:
            The reasoning text joined by newlines, or an empty string
            when no text parts are present.
        """
        parts: List[str] = []
        for entry in item.get("summary") or []:
            if entry.get("type") == "summary_text" and entry.get("text"):
                parts.append(entry["text"])
        for entry in item.get("content") or []:
            if entry.get("type") == "reasoning_text" and entry.get("text"):
                parts.append(entry["text"])
        return "\n".join(parts)
