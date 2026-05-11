"""Tests for client classes in ``codo.clients``."""

import base64
import json
from unittest.mock import Mock

import pytest

import codo.clients.base as base_module
from codo.clients.openai import OpenAIAPIClient, OpenAISubscriptionClient
from codo.types.messages import AssistantMessage, UserMessage
from codo.types.requests import OpenAIRequest


def _build_request(prompt: str = "hello") -> OpenAIRequest:
    """Build a minimal request for client tests.

    Args:
        prompt: the text content of the single user message.

    Returns:
        A :class:`OpenAIRequest` wrapping a single user message.
    """
    return OpenAIRequest(message_list=[UserMessage(content=prompt)])


def _build_id_token(account_id: str) -> str:
    """Build a minimal unsigned JWT carrying the account id claim.

    Args:
        account_id: the ChatGPT account id to embed in the token claim.

    Returns:
        The assembled unsigned JWT string.
    """
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": account_id,
                    }
                }
            ).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}.sig"


def _build_response_mock() -> Mock:
    """Build a reusable mocked HTTP response.

    Returns:
        A :class:`unittest.mock.Mock` configured to look like a successful
        OpenAI streaming response.
    """
    response = Mock()
    response.status_code = 200
    response.headers = {"Content-Type": "text/event-stream"}
    response.text = "data: " + json.dumps(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "pong",
                    }
                ],
            },
        }
    )
    response.iter_lines.return_value = response.text.splitlines()
    return response


def test_openai_api_client_uses_explicit_api_key() -> None:
    """The explicit API key is stored and used for auth headers."""
    client = OpenAIAPIClient(api_key="key-123")

    assert client._token == "key-123"
    assert client._headers == {"Authorization": "Bearer key-123"}


def test_openai_api_client_reads_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API client falls back to ``OPENAI_API_KEY``."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    client = OpenAIAPIClient()

    assert client._token == "env-key"
    assert client._headers == {"Authorization": "Bearer env-key"}


def test_openai_api_client_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API client rejects missing credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key missing"):
        OpenAIAPIClient()


def test_openai_api_client_send_request_builds_expected_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sending a request uses the OpenAI Responses payload and auth header."""
    response = _build_response_mock()
    post = Mock(return_value=response)
    monkeypatch.setattr(base_module.requests, "post", post)

    client = OpenAIAPIClient(api_key="api-key")
    result = list(client.send_request(_build_request("ping")))

    assert result == [AssistantMessage(content="pong")]
    post.assert_called_once_with(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer api-key"},
        json={
            "model": "gpt-5.5",
            "instructions": "",
            "tools": [],
            "input": [{"role": "user", "content": "ping"}],
            "reasoning": {"effort": "medium", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "stream": True,
            "store": False,
        },
        stream=True,
    )


def test_openai_subscription_client_uses_persisted_token_file(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted subscription tokens load the account id into request headers."""
    token_file = tmp_path / "oauth_token.json"
    token_file.write_text(
        json.dumps(
            {
                "access_token": "subscription-token",
                "extra": {"account_id": "acc-123"},
            }
        )
    )
    monkeypatch.setattr(OpenAISubscriptionClient, "_token_path", token_file)

    client = OpenAISubscriptionClient()

    assert client._token == "subscription-token"
    assert client._extra == {"account_id": "acc-123"}
    assert client._headers == {
        "Authorization": "Bearer subscription-token",
        "chatgpt-account-id": "acc-123",
    }


def test_openai_subscription_client_rejects_missing_account_id(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted OAuth metadata must include a non-empty account id."""
    token_file = tmp_path / "oauth_token.json"
    token_file.write_text(json.dumps({"access_token": "subscription-token"}))
    monkeypatch.setattr(OpenAISubscriptionClient, "_token_path", token_file)

    with pytest.raises(ValueError, match="chatgpt account_id"):
        OpenAISubscriptionClient()


def test_openai_subscription_client_rejects_empty_account_id(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty account ids are rejected before headers are built."""
    token_file = tmp_path / "oauth_token.json"
    token_file.write_text(
        json.dumps(
            {
                "access_token": "subscription-token",
                "extra": {"account_id": ""},
            }
        )
    )
    monkeypatch.setattr(OpenAISubscriptionClient, "_token_path", token_file)

    with pytest.raises(ValueError, match="chatgpt account_id"):
        OpenAISubscriptionClient()


def test_openai_subscription_client_login_branch_persists_token_file(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OAuth login persists the token and extracted account id."""
    token_file = tmp_path / "oauth_token.json"
    token_response = {
        "access_token": "new-token",
        "id_token": _build_id_token("acc-login"),
    }

    monkeypatch.setattr(OpenAISubscriptionClient, "_token_path", token_file)
    monkeypatch.setattr(
        OpenAISubscriptionClient,
        "_SubscriptionClient__browser_login",
        classmethod(lambda cls: token_response),
    )

    client = OpenAISubscriptionClient()

    assert client._token == "new-token"
    assert client._extra == {"account_id": "acc-login"}
    assert client._headers == {
        "Authorization": "Bearer new-token",
        "chatgpt-account-id": "acc-login",
    }
    assert json.loads(token_file.read_text()) == {
        "access_token": "new-token",
        "extra": {"account_id": "acc-login"},
    }


def test_openai_subscription_client_send_request_builds_expected_http_request(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subscription requests include both auth headers and the Responses payload."""
    token_file = tmp_path / "oauth_token.json"
    token_file.write_text(
        json.dumps(
            {
                "access_token": "subscription-token",
                "extra": {"account_id": "acc-123"},
            }
        )
    )
    monkeypatch.setattr(OpenAISubscriptionClient, "_token_path", token_file)

    response = _build_response_mock()
    post = Mock(return_value=response)
    monkeypatch.setattr(base_module.requests, "post", post)

    client = OpenAISubscriptionClient()
    result = list(client.send_request(_build_request("ping")))

    assert result == [AssistantMessage(content="pong")]
    post.assert_called_once_with(
        "https://chatgpt.com/backend-api/codex/responses",
        headers={
            "Authorization": "Bearer subscription-token",
            "chatgpt-account-id": "acc-123",
        },
        json={
            "model": "gpt-5.5",
            "instructions": "",
            "tools": [],
            "input": [{"role": "user", "content": "ping"}],
            "reasoning": {"effort": "medium", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "stream": True,
            "store": False,
        },
        stream=True,
    )


def test_openai_subscription_client_extract_extra_rejects_missing_account_id() -> None:
    """Account extraction fails when the id token does not contain the claim."""
    client = object.__new__(OpenAISubscriptionClient)

    with pytest.raises(ValueError, match="chatgpt_account_id"):
        client._extract_extra({"id_token": ""})
