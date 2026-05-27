"""Web search tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class SearchWebToolResultParser(BaseToolResultParser):
    """Parse web search tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse web search tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw web search tool output.
            is_error: whether the tool execution failed.

        Returns:
            A parsed tool result message.
        """
        if not isinstance(output, dict):
            if is_error:
                return ToolErrorMessage(
                    content=str(output),
                    id=call_id,
                    output=output,
                )
            return ToolResultMessage(
                content=str(output),
                id=call_id,
                output=output,
            )

        timed_out = output.get("timed_out")
        if timed_out or is_error:
            content = cls._format_content(output)
            if timed_out:
                content = f"{content}\n[timed out]" if content else "[timed out]"
            return ToolErrorMessage(
                content=content,
                id=call_id,
                output=output,
            )

        content = cls._format_content(output)
        return ToolResultMessage(
            content=content,
            id=call_id,
            output=output,
        )

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
