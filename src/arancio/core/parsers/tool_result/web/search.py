"""Web search tool result parser."""

from arancio.core.parsers.tool_result.base import BaseToolResultParser


class SearchWebToolResultParser(BaseToolResultParser):
    """Parse web search tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render the result list, appending a timeout marker when needed.

        Args:
            output: the structured web search tool output.

        Returns:
            The formatted result block, with a ``[timed out]`` suffix on
            timeout.
        """
        display_text = cls._format_content(output)
        if output.get("timed_out"):
            display_text = (
                f"{display_text}\n[timed out]" if display_text else "[timed out]"
            )
        return display_text

    @classmethod
    def _failed(cls, output: dict) -> bool:
        """Report failure when the search timed out.

        Args:
            output: the structured web search tool output.

        Returns:
            True when the search timed out.
        """
        return bool(output.get("timed_out"))

    @classmethod
    def _format_content(cls, output: dict) -> str:
        """Format the result list into a human-readable block with footer.

        Args:
            output: the web search tool result dict.

        Returns:
            A formatted string with numbered results and a summary footer.
        """
        query = output.get("query") or ""
        results: list = output.get("results") or []
        total = len(results)

        if total == 0:
            return "[no results]"

        blocks: list[str] = []
        for i, r in enumerate(results, start=1):
            if isinstance(r, dict):
                title = r.get("title", "")
                url = r.get("url", "")
                excerpt = r.get("excerpt", "")
                blocks.append(f"{i}. {title}\n   {url}\n   {excerpt}")
            else:
                blocks.append(f"{i}. {r}")

        noun = "result" if total == 1 else "results"
        footer = f'[{total} {noun} for "{query}"]'
        return "\n\n".join(blocks) + "\n\n" + footer
