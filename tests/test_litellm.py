"""Tests for ``arancio.core.clients.litellm.LiteLLMClient``."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import arancio.core.clients.litellm as litellm_module
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.types.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from arancio.core.types.requests import BaseRequest
from arancio.core.types.tools import ToolSchema


def _build_response(output: list) -> SimpleNamespace:
    """Build a fake ``litellm.responses`` response object.

    Args:
        output: the list of fake output items to expose under ``output``.

    Returns:
        A namespace mimicking the Responses API response shape.
    """
    return SimpleNamespace(output=output)


def _patch_responses(monkeypatch: pytest.MonkeyPatch, output: list) -> Mock:
    """Patch ``litellm.responses`` to return a fixed response.

    Args:
        monkeypatch: the pytest fixture used to install the patch.
        output: the fake output list returned by the patched call.

    Returns:
        The mock installed in place of ``litellm.responses``.
    """
    responses = Mock(return_value=_build_response(output))
    monkeypatch.setattr(litellm_module.litellm, "responses", responses)
    return responses


def test_send_request_calls_litellm_with_minimal_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare request forwards only model and input; no credential kwargs."""
    responses = _patch_responses(monkeypatch, [])

    client = LiteLLMClient()
    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="ping")],
    )

    list(client.send_request(request))

    responses.assert_called_once_with(
        model="openai/gpt-4o",
        input=[{"role": "user", "content": "ping"}],
    )
    assert "api_key" not in responses.call_args.kwargs


def test_send_request_passes_instructions_tools_and_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System prompt, tools and reasoning effort are forwarded when set."""
    responses = _patch_responses(monkeypatch, [])

    tool = ToolSchema(
        name="add",
        description="add two ints",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    request = BaseRequest(
        model_id="openai/gpt-4o",
        thinking_effort="high",
        thinking_summary="auto",
        system_prompt="be terse",
        tool_list=[tool],
        message_list=[UserMessage(content="ping")],
    )

    list(LiteLLMClient().send_request(request))

    kwargs = responses.call_args.kwargs
    assert kwargs["instructions"] == "be terse"
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "add",
            "description": "add two ints",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
        }
    ]


def test_send_request_omits_optional_kwargs_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty system prompt, no tools and no reasoning effort drop the kwargs."""
    responses = _patch_responses(monkeypatch, [])

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="ping")],
    )

    list(LiteLLMClient().send_request(request))

    kwargs = responses.call_args.kwargs
    assert "instructions" not in kwargs
    assert "tools" not in kwargs
    assert "reasoning" not in kwargs
    assert "max_output_tokens" not in kwargs


def test_send_request_forwards_max_output_tokens_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor cap is forwarded to litellm.responses."""
    responses = _patch_responses(monkeypatch, [])

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="ping")],
    )

    list(LiteLLMClient(max_output_tokens=256).send_request(request))

    assert responses.call_args.kwargs["max_output_tokens"] == 256


def test_send_request_converts_full_message_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every supported message type maps to its Responses input shape."""
    responses = _patch_responses(monkeypatch, [])

    reasoning_item = {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [{"type": "summary_text", "text": "thinking..."}],
        "encrypted_content": "ec-blob",
    }
    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
            ReasoningMessage(content="thinking...", item=reasoning_item),
            ToolCallMessage(
                id="call-1",
                name="add",
                arguments={"a": 1, "b": 2},
                content="add({...})",
            ),
            ToolResultMessage(
                id="call-1",
                content="3",
            ),
        ],
    )

    list(LiteLLMClient().send_request(request))

    assert responses.call_args.kwargs["input"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        reasoning_item,
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "add",
            "arguments": json.dumps({"a": 1, "b": 2}),
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "3",
        },
    ]


def test_send_request_serializes_non_string_tool_result_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-string tool outputs are JSON-encoded on the wire."""
    responses = _patch_responses(monkeypatch, [])

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[
            ToolResultMessage(id="call-1", content={"result": 3}, display_text="..."),
        ],
    )

    list(LiteLLMClient().send_request(request))

    assert responses.call_args.kwargs["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": json.dumps({"result": 3}),
        }
    ]


def test_send_request_rejects_unsupported_message_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown message subtype aborts the request."""

    class _Unknown:
        content = ""

    responses = Mock()
    monkeypatch.setattr(litellm_module.litellm, "responses", responses)

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[_Unknown()],
    )

    with pytest.raises(ValueError, match="Unsupported message type"):
        list(LiteLLMClient().send_request(request))
    responses.assert_not_called()


def test_parse_response_returns_assistant_tool_call_and_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three normalized message kinds are emitted from one response."""
    reasoning_item = {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [
            {"type": "summary_text", "text": "step one"},
            {"type": "summary_text", "text": "step two"},
        ],
        "encrypted_content": "ec-blob",
    }
    output = [
        reasoning_item,
        {
            "type": "message",
            "content": [
                {"type": "output_text", "text": "the answer is "},
                {"type": "output_text", "text": "3"},
            ],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "add",
            "arguments": json.dumps({"a": 1, "b": 2}),
        },
    ]
    _patch_responses(monkeypatch, output)

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="add 1+2")],
    )

    result = list(LiteLLMClient().send_request(request))

    assert result == [
        ReasoningMessage(content="step one\nstep two", item=reasoning_item),
        AssistantMessage(content="the answer is 3"),
        ToolCallMessage(
            id="call-1",
            name="add",
            arguments={"a": 1, "b": 2},
            content=f"add({json.dumps({'a': 1, 'b': 2})})",
        ),
    ]


def test_parse_response_handles_null_summary_and_content_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning items with ``summary: null`` / ``content: null`` parse cleanly."""
    reasoning_item = {
        "type": "reasoning",
        "summary": None,
        "content": None,
    }
    _patch_responses(monkeypatch, [reasoning_item])

    request = BaseRequest(
        model_id="openrouter/openai/o3-mini",
        message_list=[UserMessage(content="hi")],
    )

    result = list(LiteLLMClient().send_request(request))

    assert result == [ReasoningMessage(content="", item=reasoning_item)]


def test_parse_response_extracts_anthropic_reasoning_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic-shaped reasoning items expose text via ``content[]``."""
    reasoning_item = {
        "type": "reasoning",
        "summary": [],
        "content": [
            {"type": "reasoning_text", "text": "step a"},
            {"type": "reasoning_text", "text": "step b"},
        ],
        "format": "anthropic-claude-v1",
    }
    _patch_responses(monkeypatch, [reasoning_item])

    request = BaseRequest(
        model_id="openrouter/anthropic/claude-sonnet-4.6",
        message_list=[UserMessage(content="hi")],
    )

    result = list(LiteLLMClient().send_request(request))

    assert result == [
        ReasoningMessage(content="step a\nstep b", item=reasoning_item),
    ]


def test_parse_response_skips_unknown_output_item_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output items with no normalized counterpart are silently ignored."""
    output = [
        {"type": "web_search_call", "id": "ws-1"},
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        },
    ]
    _patch_responses(monkeypatch, output)

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="hi")],
    )

    result = list(LiteLLMClient().send_request(request))

    assert result == [AssistantMessage(content="ok")]


def test_parse_response_handles_pydantic_style_output_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic-like output items are coerced via ``model_dump``."""

    class _Item:
        def __init__(self, data: dict) -> None:
            self._data = data

        def model_dump(self) -> dict:
            return self._data

    output = [
        _Item(
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "pong"}],
            }
        )
    ]
    _patch_responses(monkeypatch, output)

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="ping")],
    )

    result = list(LiteLLMClient().send_request(request))

    assert result == [AssistantMessage(content="pong")]


def test_streaming_registers_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming call registers an unknown model and forwards ``stream``."""
    from litellm.utils import supports_native_streaming

    event = SimpleNamespace(type="response.output_text.delta", delta="hi")
    responses = Mock(return_value=[event])
    monkeypatch.setattr(litellm_module.litellm, "responses", responses)

    model = "chatgpt/gpt-unknown-stream-test"
    request = BaseRequest(
        model_id=model,
        message_list=[UserMessage(content="hi")],
    )

    result = list(LiteLLMClient(stream=True).send_request(request))

    assert responses.call_args.kwargs["stream"] is True
    assert result == [AssistantChunkMessage(content="hi")]
    assert supports_native_streaming(model=model, custom_llm_provider="chatgpt")


def test_streaming_does_not_reregister_known_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model already in the registry is not re-registered."""
    responses = Mock(return_value=[])
    monkeypatch.setattr(litellm_module.litellm, "responses", responses)
    register = Mock()
    monkeypatch.setattr(litellm_module.litellm, "register_model", register)

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="hi")],
    )

    list(LiteLLMClient(stream=True).send_request(request))

    register.assert_not_called()
    assert responses.call_args.kwargs["stream"] is True


def test_non_streaming_does_not_register_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-streaming call never touches the model registry."""
    responses = _patch_responses(monkeypatch, [])
    register = Mock()
    monkeypatch.setattr(litellm_module.litellm, "register_model", register)

    request = BaseRequest(
        model_id="chatgpt/gpt-unknown-nonstream-test",
        message_list=[UserMessage(content="hi")],
    )

    list(LiteLLMClient().send_request(request))

    register.assert_not_called()
    assert "stream" not in responses.call_args.kwargs


def test_stream_setter_toggles_request_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``stream`` after construction switches the request path."""
    responses = _patch_responses(monkeypatch, [])

    client = LiteLLMClient(stream=True)
    client.stream = False

    assert client.stream is False

    request = BaseRequest(
        model_id="openai/gpt-4o",
        message_list=[UserMessage(content="hi")],
    )
    list(client.send_request(request))

    assert "stream" not in responses.call_args.kwargs
