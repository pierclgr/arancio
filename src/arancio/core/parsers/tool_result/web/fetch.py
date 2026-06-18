"""Web fetch tool result parser."""

from arancio.core.parsers.tool_result.base import BaseToolResultParser


class FetchWebToolResultParser(BaseToolResultParser):
    """Parse web fetch tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Format the answer with a source footer.

        Args:
            output: the web fetch tool result dict.

        Returns:
            A string with the LLM answer followed by a source footer.
        """
        answer = output.get("answer") or "[no answer]"

        footer_bits = [bit for bit in (output.get("title"), output.get("url")) if bit]
        tail = f"[{' — '.join(footer_bits)}]" if footer_bits else ""
        if output.get("truncated"):
            tail = f"{tail}\n[content truncated]" if tail else "[content truncated]"

        return f"{answer}\n\n{tail}" if tail else answer
