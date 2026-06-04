"""Web fetch tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class FetchWebToolResultParser(BaseToolResultParser):
    """Parse web fetch tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse web fetch tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw web fetch tool output.
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

        content = cls._format_content(output)
        if is_error:
            return ToolErrorMessage(
                content=content,
                id=call_id,
                output=output,
            )
        return ToolResultMessage(
            content=content,
            id=call_id,
            output=output,
        )

    @classmethod
    def _format_content(cls, output: dict) -> str:
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
