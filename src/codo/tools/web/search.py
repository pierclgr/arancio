"""Web search tool exposing DuckDuckGo search to LLM clients."""

from typing import Type

from ddgs import DDGS
from ddgs.exceptions import TimeoutException

from codo.parsers.tool_result.web.search import SearchWebToolResultParser
from codo.tools.base import BaseTool


class SearchWebTool(BaseTool):
    """Run a web search via DuckDuckGo (through the ``ddgs`` library).

    The tool dispatches the query to ``ddgs.DDGS().text(...)`` with a configurable
    result cap and timeout, then normalizes each hit to ``{url, title, excerpt}``. The
    excerpt is whatever the search backend itself returned -- the tool does not fetch
    the URL. This is a read-only, network-only tool: it does not interact with
    ``ToolSession``.
    """

    _default_num_results: int = 10
    _max_num_results: int = 20
    _default_timeout: int = 60
    _max_timeout: int = 300
    _result_parser: Type[SearchWebToolResultParser] = SearchWebToolResultParser

    def _call(
        self,
        query: str,
        num_results: int | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Run a DuckDuckGo text search and return structured results.

        Args:
            query: free-form search query string.
            num_results: maximum number of results to return. Defaults to
                :attr:`_default_num_results`. Clamped to
                ``[1, _max_num_results]``.
            timeout: per-search timeout in seconds. Defaults to
                :attr:`_default_timeout`. Clamped to ``[1, _max_timeout]``.

        Returns:
            A dict with keys ``query`` (str), ``results`` (list of dicts
            with ``url``, ``title`` and ``excerpt``) and ``timed_out``
            (bool). ``timed_out`` is true only when ddgs raised
            ``TimeoutException``, which happens when zero results were
            aggregated before the deadline. Non-timeout ddgs failures
            (rate limits, network errors, …) propagate as
            ``DDGSException`` and are wrapped into a ToolErrorMessage
            string by ``BaseTool.call``.
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
