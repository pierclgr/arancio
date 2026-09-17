"""JSON-compatible encoding for normalized conversation messages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from arancio.core.messages import (
    AssistantMessage,
    ChunkMessage,
    ErrorMessage,
    Message,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)


def message_to_record(message: Message, visible: bool = True) -> dict[str, Any] | None:
    """Convert one finalized normalized message into a session event.

    The message's own ``in_history`` flag is stored, and ``visible`` is stated
    by the caller asking for the write, so restoration routes the record back
    exactly where it came from. A streaming chunk has no record of its own: the
    schema has no type for it, and it is a fragment of a message that is saved
    once finalized. Dropping it here rather than in the caller also keeps it off
    the branches below, where its finalized parent class would otherwise match
    and save the fragment as a complete message.

    Args:
        message: the finalized message to encode.
        visible: whether the record belongs in the replayed UI log.

    Returns:
        The JSON-compatible message event record, or ``None`` when ``message``
        is a streaming chunk.

    Raises:
        ValueError: when ``message`` is not a supported message type.
    """
    if isinstance(message, ChunkMessage):
        return None

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "content": message.content,
        "display_text": message.display_text,
        "in_history": message.in_history,
        "visible": visible,
    }
    if isinstance(message, ToolErrorMessage):
        record.update(type="tool_error", call_id=message.id)
    elif isinstance(message, ToolResultMessage):
        record.update(type="tool_result", call_id=message.id)
    elif isinstance(message, ToolCallMessage):
        record.update(
            type="tool_call",
            call_id=message.id,
            name=message.name,
            arguments=message.arguments,
        )
    elif isinstance(message, ErrorMessage):
        record.update(type="error")
    elif isinstance(message, ReasoningMessage):
        record.update(type="reasoning", item=message.item)
    elif isinstance(message, AssistantMessage):
        record.update(type="assistant")
    elif isinstance(message, UserMessage):
        record.update(type="user")
    else:
        raise ValueError(f"Unsupported session message: {type(message).__name__}")
    return record


def message_from_record(record: dict[str, Any]) -> Message:
    """Rebuild one normalized message from a session event record.

    Args:
        record: a JSON-compatible message event record.

    Returns:
        The reconstructed finalized message.

    Raises:
        ValueError: when the record does not describe a supported message.
    """
    message_type = record.get("type")
    content = record.get("content")
    display_text = record.get("display_text")
    flags: dict[str, Any] = {"in_history": record.get("in_history") is True}
    if message_type == "user":
        return UserMessage(content=content, display_text=display_text, **flags)
    if message_type == "assistant":
        return AssistantMessage(content=content, display_text=display_text, **flags)
    if message_type == "error":
        return ErrorMessage(content=content, display_text=display_text, **flags)
    if message_type == "reasoning":
        item = record.get("item")
        if not isinstance(item, dict):
            raise ValueError("reasoning item is missing or invalid")
        return ReasoningMessage(
            content=content, display_text=display_text, item=item, **flags
        )
    if message_type == "tool_call":
        arguments = record.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments are missing or invalid")
        return ToolCallMessage(
            content=content,
            display_text=display_text,
            id=_required_str(record, "call_id"),
            name=_required_str(record, "name"),
            arguments=arguments,
            **flags,
        )
    if message_type in {"tool_result", "tool_error"}:
        message_class = (
            ToolErrorMessage if message_type == "tool_error" else ToolResultMessage
        )
        return message_class(
            content=content,
            display_text=display_text,
            id=_required_str(record, "call_id"),
            **flags,
        )
    raise ValueError(f"Unsupported session message type: {message_type!r}")


def is_message_record(record: dict[str, Any]) -> bool:
    """Return whether a record describes a normalized finalized message.

    Args:
        record: the event record to inspect.

    Returns:
        True when the record type is a supported message type.
    """
    return record.get("type") in {
        "user",
        "assistant",
        "reasoning",
        "tool_call",
        "tool_result",
        "tool_error",
        "error",
    }


def _required_str(record: dict[str, Any], field: str) -> str:
    """Return one required string field from an event record.

    Args:
        record: the event record to inspect.
        field: the required field name.

    Returns:
        The non-empty string value.

    Raises:
        ValueError: when the field is absent or is not a string.
    """
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} is missing or invalid")
    return value
