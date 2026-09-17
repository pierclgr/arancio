"""Tests for the request/payload/response pipeline and the LiteLLM client.

This is the layer that turns our normalized messages into what a provider expects and
back again. The payload builder's conditional keys matter because sending an empty
``tools`` or a ``reasoning`` block a model does not support is a request error, not a
no-op.
"""

import json
from types import SimpleNamespace
from typing import Any, Iterator, List

import pytest
from fakes import ScriptedController

import arancio.core.clients.litellm as litellm_module
from arancio.core.builders.payload.litellm import LiteLLMPayloadBuilder
from arancio.core.clients.base import BaseClient
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.controllers.requests import ChatGPTLoginRequest
from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from arancio.core.parsers.response.litellm import LiteLLMResponseParser
from arancio.core.requests import BaseRequest
from arancio.core.tools.schema import ToolSchema


def _request(**kwargs: Any) -> BaseRequest:
    """Build a request with everything optional left unset.

    Args:
        **kwargs: fields overriding the minimal request.

    Returns:
        A frozen request for the payload builder.
    """
    return BaseRequest(model_id="openai/gpt-4o", **kwargs)


def test_a_minimal_payload_carries_nothing_it_was_not_given() -> None:
    """An empty tool list or system prompt must not become an empty key."""
    payload = LiteLLMPayloadBuilder.build(_request())

    assert set(payload) == {"model", "input"}
    assert payload["model"] == "openai/gpt-4o"


def test_a_system_prompt_and_tools_are_added_when_present() -> None:
    """Both are optional, so both are conditional."""
    schema = ToolSchema(name="ReadFileTool", description="d", input_schema={"a": 1})

    payload = LiteLLMPayloadBuilder.build(
        _request(system_prompt="be helpful", tool_list=[schema])
    )

    assert payload["instructions"] == "be helpful"
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "ReadFileTool",
            "description": "d",
            "parameters": {"a": 1},
        }
    ]


def test_a_thinking_summary_only_nests_inside_a_thinking_effort() -> None:
    """The summary has no meaning on its own, so it never travels alone."""
    with_summary = LiteLLMPayloadBuilder.build(
        _request(thinking_effort="high", thinking_summary="auto")
    )
    without_summary = LiteLLMPayloadBuilder.build(_request(thinking_effort="high"))
    without_effort = LiteLLMPayloadBuilder.build(_request(thinking_summary="auto"))

    assert with_summary["reasoning"] == {"effort": "high", "summary": "auto"}
    assert without_summary["reasoning"] == {"effort": "high"}
    assert "reasoning" not in without_effort


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (UserMessage(content="hi"), {"role": "user", "content": "hi"}),
        (AssistantMessage(content="yo"), {"role": "assistant", "content": "yo"}),
        (
            ToolCallMessage(content="", id="c1", name="T", arguments={"a": 1}),
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "T",
                "arguments": '{"a": 1}',
            },
        ),
        (
            ToolResultMessage(content="done", id="c1"),
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ),
    ],
    ids=["user", "assistant", "tool-call", "tool-result"],
)
def test_each_message_becomes_its_provider_input_item(
    message: Message, expected: dict
) -> None:
    """The mapping is what the provider sees, so it is pinned per type."""
    assert LiteLLMPayloadBuilder._build_input_item(message) == expected


def test_a_reasoning_item_is_handed_back_exactly_as_it_arrived() -> None:
    """Round-tripping the provider's own item is the point of the Responses API."""
    item = {"id": "rs_1", "type": "reasoning", "encrypted_content": "..."}

    built = LiteLLMPayloadBuilder._build_input_item(
        ReasoningMessage(content="", item=item)
    )

    assert built == item
    assert built is not item


def test_a_structured_tool_result_is_serialized() -> None:
    """A dict payload reaches the provider as JSON, not as a Python repr."""
    built = LiteLLMPayloadBuilder._build_input_item(
        ToolResultMessage(content={"ok": True}, id="c1")
    )

    assert built["output"] == '{"ok": true}'


def test_a_message_the_provider_has_no_slot_for_is_refused() -> None:
    """An error is ours to show, not something to send upstream."""
    with pytest.raises(ValueError):
        LiteLLMPayloadBuilder._build_input_item(ErrorMessage(content="boom"))


def test_a_response_is_split_into_its_output_items() -> None:
    """Reasoning, text and tool calls come back as three normalized messages."""
    response = SimpleNamespace(
        output=[
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hmm"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "ReadFileTool",
                "arguments": '{"file_path": "/a"}',
            },
            {"type": "something_else"},
        ]
    )

    messages = list(LiteLLMResponseParser.parse(response))

    assert isinstance(messages[0], ReasoningMessage)
    assert messages[0].content == "hmm"
    assert messages[1] == AssistantMessage(content="hello")
    call = messages[2]
    assert isinstance(call, ToolCallMessage)
    assert call.arguments == {"file_path": "/a"}
    assert call.content == 'ReadFileTool({"file_path": "/a"})'
    assert len(messages) == 3


def test_a_response_without_output_yields_nothing() -> None:
    """A malformed provider reply is empty, not a crash."""
    assert list(LiteLLMResponseParser.parse(SimpleNamespace())) == []


def test_a_stream_interleaves_fragments_with_finalized_messages() -> None:
    """Both delta kinds become chunks; a completed item becomes the real message."""
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="he"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="th"),
        SimpleNamespace(type="response.irrelevant", delta="ignored"),
        SimpleNamespace(
            type="response.output_item.done",
            item={
                "type": "message",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ),
    ]

    messages = list(LiteLLMResponseParser.parse_stream(events))

    assert messages[0] == AssistantChunkMessage(content="he")
    assert messages[1] == ReasoningChunkMessage(content="th", item={})
    assert messages[2] == AssistantMessage(content="hi")
    assert len(messages) == 3


def test_repeated_arguments_decode_to_the_first_object() -> None:
    """Some bridged providers concatenate the same arguments twice.

    ``json.loads`` rejects that with "Extra data", so the parser decodes the leading
    object instead and ignores the repeat.
    """
    raw = '{"a": 1}{"a": 1}'

    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    assert LiteLLMResponseParser._parse_arguments(raw) == {"a": 1}


def test_absent_arguments_decode_to_an_empty_mapping() -> None:
    """A tool call with no arguments is callable, not undecodable."""
    assert LiteLLMResponseParser._parse_arguments(None) == {}
    assert LiteLLMResponseParser._parse_arguments("") == {}


def test_reasoning_text_is_collected_from_both_provider_shapes() -> None:
    """OpenAI fills ``summary``, Anthropic fills ``content``; both are read."""
    item = {
        "summary": [{"type": "summary_text", "text": "first"}],
        "content": [{"type": "reasoning_text", "text": "second"}],
    }

    assert LiteLLMResponseParser._render_reasoning_summary(item) == "first\nsecond"


def _stub_litellm(monkeypatch: pytest.MonkeyPatch, captured: List[dict]) -> None:
    """Replace the LiteLLM module the client calls with a recorder.

    Args:
        monkeypatch: pytest's patching fixture.
        captured: the list each call's kwargs are appended to.
    """

    def _responses(**kwargs: Any) -> Any:
        """Record the call and return an empty response of the right shape.

        Args:
            **kwargs: the payload the client built.

        Returns:
            An empty event list when streaming, otherwise a response object
            with no output items.
        """
        captured.append(kwargs)
        return [] if kwargs.get("stream") else SimpleNamespace(output=[])

    monkeypatch.setattr(
        litellm_module,
        "litellm",
        SimpleNamespace(responses=_responses, get_model_info=lambda model: {}),
    )


def test_the_client_sends_the_built_payload(
    monkeypatch: pytest.MonkeyPatch, controller: ScriptedController
) -> None:
    """Nothing is added to the payload unless the client was configured for it."""
    captured: List[dict] = []
    _stub_litellm(monkeypatch, captured)
    client = LiteLLMClient(controller=controller, model_id="openai/gpt-4o")

    list(client.send_request(_request()))

    assert captured[0]["model"] == "openai/gpt-4o"
    assert "max_output_tokens" not in captured[0]
    assert "stream" not in captured[0]


def test_an_output_token_cap_is_forwarded_when_configured(
    monkeypatch: pytest.MonkeyPatch, controller: ScriptedController
) -> None:
    """The cap lives on the client, not in the generic request."""
    captured: List[dict] = []
    _stub_litellm(monkeypatch, captured)
    client = LiteLLMClient(
        controller=controller, model_id="openai/gpt-4o", max_output_tokens=256
    )

    list(client.send_request(_request()))

    assert captured[0]["max_output_tokens"] == 256


def test_a_streaming_client_asks_the_provider_to_stream(
    monkeypatch: pytest.MonkeyPatch, controller: ScriptedController
) -> None:
    """Streaming is a client-level choice applied at send time."""
    captured: List[dict] = []
    _stub_litellm(monkeypatch, captured)
    client = LiteLLMClient(controller=controller, model_id="openai/gpt-4o", stream=True)

    list(client.send_request(_request()))

    assert captured[0]["stream"] is True


def test_a_device_code_login_is_shown_through_the_controller(
    controller: ScriptedController,
) -> None:
    """The sign-in prompt is UI-only and never enters the conversation."""
    client = LiteLLMClient(controller=controller, model_id="openai/gpt-4o")

    client._on_chatgpt_device_code("https://verify", "ABCD-1234")

    request = controller.requests[0]
    assert isinstance(request, ChatGPTLoginRequest)
    assert request.verification_url == "https://verify"
    assert request.user_code == "ABCD-1234"


class _RestrictedClient(BaseClient):
    """A client that accepts only one model and one thinking level."""

    _model_options = ["only/model"]
    _thinking_options = ["low"]
    _default_model_id = "unvalidated/default"

    def send_request(self, request: BaseRequest) -> Iterator[Message]:
        """Answer nothing; this client exists to test option validation.

        Args:
            request: the request, unused.

        Returns:
            An empty iterator.
        """
        return iter(())


def test_an_explicit_option_outside_the_allow_list_is_refused() -> None:
    """A subclass that declares options gets them enforced on assignment."""
    with pytest.raises(ValueError):
        _RestrictedClient(model_id="other/model")

    with pytest.raises(ValueError):
        _RestrictedClient(thinking_effort="extreme")


def test_the_class_default_is_trusted_without_validation() -> None:
    """Defaults bypass the setters, so a subclass owns its own consistency."""
    assert _RestrictedClient().model_id == "unvalidated/default"
