"""Web fetch tool: fetch a page and answer a query against it via an LLM."""

from datetime import datetime, timezone
from typing import Annotated, Type

import trafilatura
from pydantic import Field
from trafilatura.settings import use_config

from arancio.core.clients.base import Client
from arancio.core.hooks.manager import HookManager
from arancio.core.messages.assistant import AssistantMessage
from arancio.core.messages.base import ChunkMessage
from arancio.core.messages.user import UserMessage
from arancio.core.parsers.tool_results.web.fetch import FetchWebToolResultParser
from arancio.core.tools.base import Tool
from arancio.core.tools.session import ToolSession
from arancio.core.tools.web.constants import (
    FETCH_DEFAULT_TIMEOUT,
    FETCH_MAX_CONTENT_CHARS,
    FETCH_MAX_TIMEOUT,
    FETCH_SYSTEM_PROMPT,
)


class FetchWebTool(Tool):
    """Fetch a web page and answer a query against it with an injected LLM client.

    The tool downloads the URL and extracts its main readable content with trafilatura,
    caps the content length, then asks an injected summarization client to answer the
    caller's query using only that content. The client is built and configured by the
    caller (model, thinking effort, streaming, …) outside the tool, kept separate from
    the agent's client so summarization can run on a cheap, dedicated model. This is a
    read-only, network-only tool: it does not interact with ``ToolSession``.
    """

    _default_timeout: int = FETCH_DEFAULT_TIMEOUT
    _max_timeout: int = FETCH_MAX_TIMEOUT
    _max_content_chars: int = FETCH_MAX_CONTENT_CHARS
    _system_prompt: str = FETCH_SYSTEM_PROMPT
    _result_parser: Type[FetchWebToolResultParser] = FetchWebToolResultParser

    def __init__(
        self,
        client: Client,
        hook_manager: HookManager,
        tool_session: ToolSession,
    ) -> None:
        """Initialize the tool with an externally built summarization client.

        Args:
            client: the LLM client used to answer the query against the fetched
                page content. Built and configured by the caller so
                summarization can run on a cheap model separate from the
                agent's client.
            hook_manager: forwarded to :meth:`Tool.__init__`.
            tool_session: forwarded to :meth:`Tool.__init__`.
        """
        super().__init__(hook_manager=hook_manager, tool_session=tool_session)
        self._client = client

    @property
    def client(self) -> Client:
        """The summarization client.

        Returns:
            The client used for the summarization request.
        """
        return self._client

    @client.setter
    def client(self, value: Client) -> None:
        """Set the summarization client.

        Args:
            value: the new client to use for summarization requests.
        """
        self._client = value

    def __repr__(self) -> str:
        """Return a developer-friendly representation including the client.

        Returns:
            The tool's class name and the repr of its summarization client.
        """
        return f"{type(self).__name__}(client={self._client!r})"

    def _call(
        self,
        url: str,
        query: str,
        timeout: Annotated[int, Field(ge=1, le=FETCH_MAX_TIMEOUT)] | None = None,
    ) -> dict:
        """Fetch a page and answer ``query`` against its content.

        Args:
            url: the http(s) URL of the page to fetch (e.g.
                "https://docs.pytorch.org").
            query: instruction describing what to extract, summarize, or answer from
                the page.
            timeout: fetch timeout in seconds. Defaults to 60. Hard cap 300.

        Returns:
            A dict with keys ``url`` (str, final/canonical URL), ``query``
            (str), ``title`` (str | None), ``content_type`` (None),
            ``retrieved_at`` (str, UTC ISO timestamp), ``answer`` (str, the LLM
            response) and ``truncated`` (bool, whether content was capped).

        Raises:
            ValueError: when ``url`` is not an ``http(s)`` URL.
            RuntimeError: when the download fails, no readable content can be
                extracted, or the summarization produces no answer.
        """
        if not isinstance(url, str) or not url.lower().startswith(
            ("http://", "https://")
        ):
            raise ValueError(f"url must be an http(s) URL: {url!r}")

        t = timeout if timeout is not None else self._default_timeout
        t = max(1, min(self._max_timeout, t))

        cfg = use_config()
        cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(t))

        downloaded = trafilatura.fetch_url(url, config=cfg)
        if downloaded is None:
            raise RuntimeError(f"failed to fetch {url}")

        content = trafilatura.extract(downloaded, output_format="markdown", config=cfg)
        if not content:
            raise RuntimeError(f"no readable content extracted from {url}")

        metadata = trafilatura.extract_metadata(downloaded, default_url=url)
        title = getattr(metadata, "title", None) if metadata else None
        final_url = (getattr(metadata, "url", None) if metadata else None) or url

        truncated = len(content) > self._max_content_chars
        if truncated:
            content = content[: self._max_content_chars]

        answer = self._summarize(final_url, title, content, query)

        return {
            "url": final_url,
            "query": query,
            "title": title,
            "content_type": None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "answer": answer,
            "truncated": truncated,
        }

    def _summarize(
        self,
        url: str,
        title: str | None,
        content: str,
        query: str,
    ) -> str:
        """Answer ``query`` over the page content via the summarization client.

        Args:
            url: source URL of the fetched page.
            title: page title, when available.
            content: normalized (markdown) page content.
            query: the caller's instruction.

        Returns:
            The finalized assistant answer text.

        Raises:
            RuntimeError: when the summarization yields no answer.
        """
        user = (
            f"Source URL: {url}\n"
            f"Page title: {title or '(unknown)'}\n\n"
            f"Page content:\n{content}\n\n"
            f"Task: {query}"
        )
        request = self._client.build_request(
            messages=[UserMessage(content=user)],
            system_prompt=FetchWebTool._system_prompt,
            tools=[],
        )
        parts = [
            m.content
            for m in self._client.send_request(request)
            if isinstance(m, AssistantMessage) and not isinstance(m, ChunkMessage)
        ]
        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("summarization produced no answer")
        return answer
