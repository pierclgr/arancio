"""Web search tool exposing DuckDuckGo search to LLM clients."""

from typing import Annotated, Type

from ddgs import DDGS
from ddgs.exceptions import TimeoutException
from pydantic import Field

from arancio.core.parsers.tool_results.web.search import SearchWebToolResultParser
from arancio.core.tools.base import Tool
from arancio.core.tools.web.constants import (
    SEARCH_DEFAULT_NUM_RESULTS,
    SEARCH_DEFAULT_TIMEOUT,
    SEARCH_MAX_NUM_RESULTS,
    SEARCH_MAX_TIMEOUT,
)


class SearchWebTool(Tool):
    """Run a web search via DuckDuckGo (through the ``ddgs`` library).

    The tool dispatches the query to ``ddgs.DDGS().text(...)`` with a configurable
    result cap and timeout, then normalizes each hit to ``{url, title, excerpt}``. The
    excerpt is whatever the search backend itself returned -- the tool does not fetch
    the URL. This is a read-only, network-only tool: it does not interact with
    ``ToolSession``.
    """

    _default_num_results: int = SEARCH_DEFAULT_NUM_RESULTS
    _max_num_results: int = SEARCH_MAX_NUM_RESULTS
    _default_timeout: int = SEARCH_DEFAULT_TIMEOUT
    _max_timeout: int = SEARCH_MAX_TIMEOUT
    _result_parser: Type[SearchWebToolResultParser] = SearchWebToolResultParser

    def _execute(
        self,
        query: str,
        num_results: Annotated[int, Field(ge=1, le=SEARCH_MAX_NUM_RESULTS)]
        | None = None,
        timeout: Annotated[int, Field(ge=1, le=SEARCH_MAX_TIMEOUT)] | None = None,
    ) -> dict:
        """Run a DuckDuckGo text search and return structured results.

        Args:
            query: the free-form search query (e.g. "pytorch transformer api
                2026").
            num_results: maximum number of results to return. Defaults to 10. Hard
                cap 20.
            timeout: per-search timeout in seconds. Defaults to 60. Hard cap 300.

        Returns:
            A dict with keys ``query`` (str), ``results`` (list of dicts
            with ``url``, ``title`` and ``excerpt``) and ``timed_out``
            (bool). ``timed_out`` is true only when ddgs raised
            ``TimeoutException``, which happens when zero results were
            aggregated before the deadline. Non-timeout ddgs failures
            (rate limits, network errors, …) propagate as
            ``DDGSException`` and are wrapped into a ToolErrorMessage
            string by ``Tool.call``.
        """
        n = num_results if num_results is not None else self._default_num_results
        t = timeout if timeout is not None else self._default_timeout
        n = max(1, min(self._max_num_results, n))
        t = max(1, min(self._max_timeout, t))

        try:
            with DDGS(timeout=t) as ddgs:
                raw = ddgs.text(query, max_results=n) or []
        except TimeoutException:
            return {"query": query, "results": [], "timed_out": True}

        results = [
            {
                "url": r.get("href", ""),
                "title": r.get("title", ""),
                "excerpt": r.get("body", ""),
            }
            for r in raw
        ]
        return {"query": query, "results": results, "timed_out": False}
