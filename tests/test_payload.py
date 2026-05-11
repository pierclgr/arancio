"""Tests for payload mapping in ``codo.builders.payload``."""

import json

import pytest

from codo.builders.payload.openai import OpenAIPayloadBuilder
from codo.types.messages import (
    AssistantMessage,
    Message,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)
from codo.types.requests import BaseRequest, OpenAIRequest
from codo.types.tools import ToolSchema


def test_openai_payload_builds_minimal_user_request() -> None:
    """A minimal user request maps to the expected Responses payload."""
    request = OpenAIRequest(message_list=[UserMessage(content="hello")])

    payload = OpenAIPayloadBuilder.build(request)

    assert payload == {
        "model": "gpt-5.5",
        "instructions": "",
        "tools": [],
        "input": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "stream": True,
        "store": False,
    }


def test_openai_payload_maps_tools_and_conversation_items() -> None:
    """Tools, assistant tool calls, and tool results are mapped correctly."""
    reasoning_item = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Need search."}],
        "encrypted_content": "encrypted",
    }
    request = OpenAIRequest(
        model_id="gpt-test",
        system_prompt="be precise",
        tool_list=[
            ToolSchema(
                name="search",
                description="search docs",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            )
        ],
        message_list=[
            UserMessage(content="find docs"),
            ReasoningMessage(item=reasoning_item, content="Need search."),
            AssistantMessage(content="calling search"),
            ToolCallMessage(
                content='search({"q": "docs"})',
                id="call_1",
                name="search",
                arguments={"q": "docs"},
            ),
            ToolErrorMessage(
                content="",
                id="call_1",
                output={"matches": 3},
            ),
        ],
    )

    payload = OpenAIPayloadBuilder.build(request)

    assert payload["model"] == "gpt-test"
    assert payload["instructions"] == "be precise"
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "search",
            "description": "search docs",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
    ]
    assert payload["input"] == [
        {"role": "user", "content": "find docs"},
        reasoning_item,
        {"role": "assistant", "content": "calling search"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search",
            "arguments": json.dumps({"q": "docs"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": json.dumps({"is_error": True, "output": {"matches": 3}}),
        },
    ]


def test_openai_payload_keeps_string_tool_result_as_plain_text() -> None:
    """Successful string tool results are passed through unchanged."""
    request = OpenAIRequest(
        message_list=[
            ToolResultMessage(content="", id="call_1", output="plain text output"),
        ]
    )

    payload = OpenAIPayloadBuilder.build(request)

    assert payload["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "plain text output",
        }
    ]


def test_openai_payload_rejects_empty_message_list() -> None:
    """Payload building fails when there are no conversation items."""
    with pytest.raises(ValueError, match="message_list"):
        OpenAIPayloadBuilder.build(BaseRequest(model_id="gpt-test"))


def test_openai_payload_rejects_unknown_message_type() -> None:
    """Unknown message types raise a clear error."""
    request = BaseRequest(model_id="gpt-test", message_list=[Message(content="")])

    with pytest.raises(ValueError, match="Unsupported message type"):
        OpenAIPayloadBuilder.build(request)
