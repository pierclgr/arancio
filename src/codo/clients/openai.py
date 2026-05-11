"""Concrete OpenAI clients (API key and ChatGPT subscription)."""

import base64
import json
from pathlib import Path
from typing import List, Type

from codo.builders.payload.openai import OpenAIPayloadBuilder
from codo.builders.request.openai import OpenAIRequestBuilder
from codo.clients.base import APIClient, BaseClient, SubscriptionClient
from codo.constants.openai import (
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_DEFAULT_MODEL_ID,
    OPENAI_DEFAULT_THINKING_EFFORT,
    OPENAI_DEFAULT_THINKING_SUMMARY,
    OPENAI_MODEL_OPTIONS,
    OPENAI_OAUTH_CLIENT_ID,
    OPENAI_OAUTH_SCOPE,
    OPENAI_THINKING_OPTIONS,
    OPENAI_THINKING_SUMMARY_OPTIONS,
)
from codo.constants.path.openai import OPENAI_OAUTH_TOKEN_PATH
from codo.constants.url.openai import (
    OPENAI_API_RESPONSES_URL,
    OPENAI_AUTH_CLAIM_KEY,
    OPENAI_CODEX_RESPONSES_URL,
    OPENAI_OAUTH_AUTHORIZE_URL,
    OPENAI_OAUTH_TOKEN_URL,
)
from codo.parsers.response.openai import OpenAIResponseParser


class OpenAIClient(BaseClient):
    """Class defining the format(s) used by OpenAI clients."""

    _payload_builder: Type[OpenAIPayloadBuilder] = OpenAIPayloadBuilder
    _request_builder: Type[OpenAIRequestBuilder] = OpenAIRequestBuilder
    _response_parser: Type[OpenAIResponseParser] = OpenAIResponseParser

    _model_options: List[str] = OPENAI_MODEL_OPTIONS
    _thinking_options: List[str] = OPENAI_THINKING_OPTIONS
    _thinking_summary_options: List[str] = OPENAI_THINKING_SUMMARY_OPTIONS

    _default_model_id: str = OPENAI_DEFAULT_MODEL_ID
    _default_thinking_effort: str = OPENAI_DEFAULT_THINKING_EFFORT
    _default_thinking_summary: str = OPENAI_DEFAULT_THINKING_SUMMARY

    def __init__(self, thinking_summary: str | None = None, *args, **kwargs) -> None:
        """Initialize the client and set the default thinking summary.

        Args:
            thinking_summary: optional reasoning summary tier to set at
                construction.
            *args: positional arguments forwarded to the next class in
                the MRO.
            **kwargs: keyword arguments forwarded to the next class in
                the MRO.
        """
        super().__init__(*args, **kwargs)
        self._thinking_summary: str | None = self._default_thinking_summary
        if thinking_summary is not None:
            self.thinking_summary = thinking_summary

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the client.

        Extends the base representation with the current thinking
        summary tier.

        Returns:
            A compact string with client type and key configuration.
        """
        return (
            super().__repr__()[:-1] + f", thinking_summary={self._thinking_summary!r})"
        )

    @property
    def thinking_summary(self) -> str | None:
        """Return the reasoning summary tier used for outgoing requests.

        Returns:
            The current summary tier, or ``None`` when none is set.
        """
        return self._thinking_summary

    @thinking_summary.setter
    def thinking_summary(self, value: str | None) -> None:
        """Set the reasoning summary tier, validating against options.

        Args:
            value: the summary tier (e.g. ``"auto"``).

        Raises:
            ValueError: when ``_thinking_summary_options`` is set and
                ``value`` is not in it.
        """
        if (
            self._thinking_summary_options
            and value not in self._thinking_summary_options
        ):
            raise ValueError(
                f"Invalid thinking summary level: {value}. Choose between "
                f"{self._thinking_summary_options}"
            )
        self._thinking_summary = value


class OpenAIAPIClient(APIClient, OpenAIClient):
    """OpenAI client using a platform API key (consumption billing).

    Calls the public Responses endpoint.
    """

    _env_var: str = OPENAI_API_KEY_ENV_VAR
    _client_url: str = OPENAI_API_RESPONSES_URL

    def __init__(self, api_key: str | None = None, *args, **kwargs) -> None:
        """Initialize the client.

        Args:
            api_key: the API key; when None, ``OPENAI_API_KEY`` is read
                from the environment.
            *args: positional arguments forwarded to the next class in
                the MRO.
            **kwargs: keyword arguments forwarded to the next class in
                the MRO.
        """
        super().__init__(api_key=api_key, *args, **kwargs)


class OpenAISubscriptionClient(SubscriptionClient, OpenAIClient):
    """OpenAI client using a ChatGPT Plus/Pro subscription via OAuth.

    Calls the Codex backend responses endpoint. Uses the same OAuth public client as the
    official Codex CLI.
    """

    _client_id: str = OPENAI_OAUTH_CLIENT_ID
    _auth_url: str = OPENAI_OAUTH_AUTHORIZE_URL
    _token_url: str = OPENAI_OAUTH_TOKEN_URL
    _scope: str = OPENAI_OAUTH_SCOPE
    _client_url: str = OPENAI_CODEX_RESPONSES_URL
    _token_path: Path = OPENAI_OAUTH_TOKEN_PATH

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the client.

        Args:
            *args: positional arguments forwarded to the next class in
                the MRO.
            **kwargs: keyword arguments forwarded to the next class in
                the MRO.

        Raises:
            ValueError: when the persisted OAuth metadata does not contain
                a non-empty ChatGPT account id.
        """
        super().__init__(*args, **kwargs)
        account_id = self._extra.get("account_id", "")
        if not account_id:
            raise ValueError(
                "OpenAI OAuth token is missing chatgpt account_id. "
                "Delete the token file and login again."
            )

        self._headers["chatgpt-account-id"] = account_id

    def _extract_extra(self, token_response: dict) -> dict:
        """Extract and validate the chatgpt account id from the token response.

        Args:
            token_response: the JSON body of the token endpoint (dict).

        Returns:
            A dict with the ``account_id`` key (dict).

        Raises:
            ValueError: when the id token does not contain a non-empty
                ChatGPT account id.
        """
        account_id = self.__extract_account_id(token_response.get("id_token", ""))
        if not account_id:
            raise ValueError("OpenAI id_token did not contain chatgpt_account_id")
        return {"account_id": account_id}

    @staticmethod
    def __extract_account_id(id_token: str) -> str:
        """Decode the JWT id_token and return the chatgpt_account_id claim.

        Args:
            id_token: raw JWT string returned by the token endpoint (str).

        Returns:
            The chatgpt_account_id claim, or an empty string when absent
            (str).
        """
        if not id_token:
            return ""
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get(OPENAI_AUTH_CLAIM_KEY, {}).get(
            "chatgpt_account_id", ""
        ) or payload.get("chatgpt_account_id", "")
