"""Tests for OpenAI response parsing."""

import json
from unittest.mock import Mock

from codo.parsers.response.openai import OpenAIResponseParser
from codo.types.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
)


def _event(item: dict) -> str:
    """Render an OpenAI SSE output item event.

    Args:
        item: the output item carried by the event.

    Returns:
        The serialized SSE data line.
    """
    return "data: " + json.dumps(
        {
            "type": "response.output_item.done",
            "item": item,
        }
    )


def _response(text: str) -> Mock:
    """Build a mocked streaming response from SSE text.

    Args:
        text: the SSE text to expose as response lines.

    Returns:
        A mock response whose ``iter_lines`` method returns the lines.
    """
    response = Mock()
    response.iter_lines.return_value = text.splitlines()
    return response


def _delta_event(event_type: str, delta: str) -> str:
    """Render an OpenAI SSE delta event.

    Args:
        event_type: the OpenAI event type.
        delta: the text delta carried by the event.

    Returns:
        The serialized SSE data line.
    """
    return "data: " + json.dumps({"type": event_type, "delta": delta})


def test_openai_response_parser_keeps_reasoning_items() -> None:
    """Reasoning items are preserved for stateless follow-up requests."""
    reasoning_item = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Need a tool."}],
        "encrypted_content": "encrypted",
    }
    response = _response(_event(reasoning_item))

    messages = list(OpenAIResponseParser.parse(response))

    assert messages == [ReasoningMessage(item=reasoning_item, content="Need a tool.")]


def test_openai_response_parser_keeps_output_item_order() -> None:
    """Reasoning, tool calls, and assistant text are returned in stream order."""
    reasoning_item = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "encrypted",
    }
    response = _response(
        "\n".join(
            [
                _event(reasoning_item),
                _event(
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": json.dumps({"q": "docs"}),
                    }
                ),
                _event(
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "done",
                            }
                        ],
                    }
                ),
            ]
        )
    )

    messages = list(OpenAIResponseParser.parse(response))

    assert messages == [
        ReasoningMessage(item=reasoning_item, content=""),
        ToolCallMessage(
            content='search({"q": "docs"})',
            id="call_1",
            name="search",
            arguments={"q": "docs"},
        ),
        AssistantMessage(content="done"),
    ]


def test_openai_response_parser_yields_text_and_reasoning_deltas() -> None:
    """Text and reasoning deltas are yielded as chunk messages."""
    response = _response(
        "\n".join(
            [
                _delta_event("response.output_text.delta", "hel"),
                _delta_event("response.reasoning_summary_text.delta", "think"),
            ]
        )
    )

    messages = list(OpenAIResponseParser.parse(response))

    assert messages == [
        AssistantChunkMessage(content="hel"),
        ReasoningChunkMessage(content="think", item={}),
    ]
