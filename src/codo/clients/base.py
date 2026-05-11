"""Shared client classes for LLM HTTP APIs."""

import base64
import hashlib
import http.server
import json
import os
import secrets
import urllib.parse
import webbrowser
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import List, Type

import requests

from codo.builders.payload.base import BasePayloadBuilder
from codo.builders.request.base import BaseRequestBuilder
from codo.constants.path.base import OAUTH_TOKEN_PATH
from codo.constants.url.base import OAUTH_REDIRECT_URL
from codo.parsers.response.base import BaseResponseParser
from codo.types.messages import Message
from codo.types.requests import BaseRequest
from codo.types.tools import ToolSchema


class BaseClient:
    """Base LLM client with shared request sending behavior."""

    _client_url: str | None = None
    _payload_builder: Type[BasePayloadBuilder] = BasePayloadBuilder
    _request_builder: Type[BaseRequestBuilder] = BaseRequestBuilder
    _response_parser: Type[BaseResponseParser] = BaseResponseParser

    _model_options: List[str] | None = None
    _thinking_options: List[str] | None = []

    _default_model_id: str | None = None
    _default_thinking_effort: str | None = None

    def __init__(
        self,
        token: str | None = None,
        model_id: str | None = None,
        thinking_effort: str | None = None,
    ) -> None:
        """Initialize the client with a bearer token and optional state.

        Args:
            token: the access token or API key to use for the request.
            model_id: the model identifier to use for outgoing requests;
                when ``None`` the class default ``_default_model_id`` is
                kept.
            thinking_effort: the thinking effort tier to use for
                outgoing requests; when ``None`` the class default
                ``_default_thinking_effort`` is kept.
        """
        self._token = token
        self._headers = {"Authorization": f"Bearer {self._token}"}
        self._model_id: str | None = self._default_model_id
        self._thinking_effort: str | None = self._default_thinking_effort
        if model_id is not None:
            self.model_id = model_id
        if thinking_effort is not None:
            self.thinking_effort = thinking_effort

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the client.

        Returns:
            A compact string with client type and key configuration.
        """
        return (
            f"{type(self).__name__}("
            f"model_id={self._model_id!r}, "
            f"thinking_effort={self._thinking_effort!r}"
            ")"
        )

    @property
    def model_id(self) -> str | None:
        """Return the model id used for outgoing requests.

        Returns:
            The current model id, or ``None`` when no default exists and
            none has been set.
        """
        return self._model_id

    @model_id.setter
    def model_id(self, value: str) -> None:
        """Set the model id, validating against ``_model_options`` when set.

        Args:
            value: the model id to use for subsequent requests.

        Raises:
            ValueError: when ``_model_options`` is set and ``value`` is
                not in it.
        """
        if self._model_options and value not in self._model_options:
            raise ValueError(
                f"Invalid model identifier: {value}. Choose between "
                f"{self._model_options}"
            )
        self._model_id = value

    @property
    def thinking_effort(self) -> str | None:
        """Return the thinking effort used for outgoing requests.

        Returns:
            The current effort string, or ``None`` when none is set.
        """
        return self._thinking_effort

    @thinking_effort.setter
    def thinking_effort(self, value: str | None) -> None:
        """Set the thinking effort, validating against ``_thinking_options``.

        Args:
            value: the thinking effort tier (e.g. ``"low"``).

        Raises:
            ValueError: when ``_thinking_options`` is set and ``value``
                is not in it.
        """
        if self._thinking_options and value not in self._thinking_options:
            raise ValueError(
                f"Invalid thinking effort level: {value}. Choose between "
                f"{self._thinking_options}"
            )
        self._thinking_effort = value

    def build_request(
        self,
        messages: List[Message],
        system_prompt: str = "",
        tools: List[ToolSchema] | None = None,
    ) -> BaseRequest:
        """Build a normalized request bound to the client's current state.

        Delegates to the client's :class:`BaseRequestBuilder` subclass so
        provider-specific fields are populated without callers needing to
        know which fields the underlying request type carries.

        Args:
            messages: the message list to send.
            system_prompt: the system prompt string.
            tools: the tool catalog to expose to the model; when None
                an empty list is used.

        Returns:
            A normalized :class:`BaseRequest` ready for ``send_request``.
        """
        return self._request_builder.build(
            client=self,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
        )

    def send_request(self, request: BaseRequest) -> Iterator[Message]:
        """Send a request and yield parsed messages from the response stream.

        Args:
            request: the generic request to send to the client.

        Yields:
            Each normalized message parsed from the HTTP response stream.

        Raises:
            ValueError: when ``_client_url`` is not set on the concrete
                client.
        """
        if self._client_url is None:
            raise ValueError('"client_url" must be set on the concrete client')

        payload = self._payload_builder.build(request)

        response = requests.post(
            self._client_url,
            headers=self._headers,
            json=payload,
            stream=True,
        )
        response.raise_for_status()

        yield from self._response_parser.parse(response=response)


class APIClient(BaseClient):
    """Client authenticating via an API key as a Bearer token.

    Subclasses set the ``env_var`` class attribute to the name of the environment
    variable consulted when no ``api_key`` is passed to ``__init__``.
    """

    _env_var: str = ""

    def __init__(self, api_key: str | None = None, *args, **kwargs) -> None:
        """Initialize the client and load the API key.

        The key is taken from ``api_key`` when provided, otherwise from the
        environment variable named by ``env_var``.

        Args:
            api_key: the API key; when None, ``env_var`` is read instead.
            *args: positional arguments forwarded to :class:`BaseClient`.
            **kwargs: keyword arguments forwarded to :class:`BaseClient`.

        Raises:
            ValueError: when no key is available from either source.
        """
        api_key = api_key or os.environ.get(self._env_var)
        if not api_key:
            raise ValueError(
                f"API key missing (constructor arg or env {self._env_var!r})"
            )

        super().__init__(token=api_key, *args, **kwargs)


class SubscriptionClient(BaseClient, ABC):
    """Client authenticating via an OAuth 2.0 PKCE browser flow.

    Subclasses set the following class attributes:
        client_id: OAuth public client identifier.
        auth_url: authorization endpoint URL.
        token_url: token exchange endpoint URL.
        scope: space-separated OAuth scopes.
        redirect_uri: local callback URL (defaults to port 1455).
        token_path: filesystem path where the access token is persisted.
    """

    _client_id: str = ""
    _auth_url: str = ""
    _token_url: str = ""
    _scope: str = ""
    _redirect_uri: str = OAUTH_REDIRECT_URL
    _token_path: Path = OAUTH_TOKEN_PATH

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the client and obtain an access token.

        When a persisted token file exists at ``token_path`` the token is
        loaded from disk. Otherwise the OAuth 2.0 PKCE browser flow is
        run: a local callback server is started, the user is redirected
        to the authorization page, the returned code is exchanged for an
        access token, and the result is saved to ``token_path`` with mode
        ``0600``.

        Args:
            *args: positional arguments forwarded to :class:`BaseClient`.
            **kwargs: keyword arguments forwarded to :class:`BaseClient`.
        """
        if self._token_path.exists():
            data = json.loads(self._token_path.read_text())
            token = data["access_token"]
            extra = data.get("extra", {})
        else:
            token_response = self.__browser_login()
            token = token_response["access_token"]
            extra = self._extract_extra(token_response)

            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(
                json.dumps(
                    {
                        "access_token": token,
                        "extra": extra,
                    }
                )
            )
            os.chmod(self._token_path, 0o600)
        self._extra = extra
        super().__init__(token=token, *args, **kwargs)

    @abstractmethod
    def _extract_extra(self, token_response: dict) -> dict:
        """Extract client-specific extra state from the token response.

        Subclasses override this hook to pull provider-specific claims
        (e.g. account ids) out of the OAuth token response.

        Args:
            token_response: the JSON body of the token endpoint (dict).

        Returns:
            Extra state to persist alongside the access token (dict).
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def __browser_login(cls) -> dict:
        """Run the OAuth PKCE browser flow and return the token response.

        Returns:
            The token response from the token endpoint (dict).

        Raises:
            RuntimeError: when the callback state does not match or the
                authorization code is missing from the callback.
        """
        code_verifier = cls.__b64url(secrets.token_bytes(32))
        code_challenge = cls.__b64url(hashlib.sha256(code_verifier.encode()).digest())
        state = cls.__b64url(secrets.token_bytes(16))

        params = {
            "response_type": "code",
            "client_id": cls._client_id,
            "redirect_uri": cls._redirect_uri,
            "scope": cls._scope,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        authorize_url = f"{cls._auth_url}?{urllib.parse.urlencode(params)}"

        callback: dict = {}

        class __Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                qs = urllib.parse.urlparse(self.path).query
                callback.update(urllib.parse.parse_qs(qs))
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Login complete. You can close this tab.")

            def log_message(self, format, *args) -> None:
                pass

        host, port = cls.__host_port(cls._redirect_uri)
        with http.server.HTTPServer((host, port), __Handler) as httpd:
            webbrowser.open(authorize_url)
            httpd.handle_request()

        if callback.get("state", [""])[0] != state:
            raise RuntimeError("OAuth state mismatch")
        code = callback.get("code", [""])[0]
        if not code:
            raise RuntimeError("OAuth code missing from callback")

        token_response = requests.post(
            cls._token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": cls._redirect_uri,
                "client_id": cls._client_id,
                "code_verifier": code_verifier,
            },
        )
        token_response.raise_for_status()
        token_response = token_response.json()

        return token_response

    @staticmethod
    def __b64url(data: bytes) -> str:
        """Encode bytes as url-safe base64 without padding.

        Args:
            data: the raw bytes to encode.

        Returns:
            The url-safe base64-encoded string without padding.
        """
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def __host_port(url: str) -> tuple[str, int]:
        """Extract (host, port) from a URL.

        Args:
            url: the URL to parse.

        Returns:
            A ``(host, port)`` tuple, defaulting to ``("localhost", 80)``
            when either component is missing from the URL.
        """
        parsed = urllib.parse.urlparse(url)
        return parsed.hostname or "localhost", parsed.port or 80
