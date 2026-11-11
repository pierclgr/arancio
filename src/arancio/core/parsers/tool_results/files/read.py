"""Read file tool result parser."""

from arancio.core.parsers.tool_results.base import ToolResultParser


class ReadFileToolResultParser(ToolResultParser):
    """Parse read file tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render read file output into a content block with a status footer.

        Args:
            output: the structured read file tool output.

        Returns:
            The file content followed by line-range and truncation annotations.
        """
        file_path = output.get("file_path") or ""
        content_block = output.get("content") or ""
        total_lines = output.get("total_lines") or 0
        start_line = output.get("start_line") or 0
        end_line = output.get("end_line") or 0
        truncated_lines = output.get("truncated_lines") or 0

        parts = []
        if file_path:
            parts.append(file_path)
        if content_block:
            parts.append(content_block)
        if "source_total_lines" in output:
            source_total = output["source_total_lines"]
            if source_total == 0:
                parts.append("[empty file]")
            elif output["source_start_line"] == 0:
                parts.append(
                    f"[no source lines returned, file has {source_total} lines]"
                )
            else:
                parts.append(
                    f"[source lines {output['source_start_line']}-"
                    f"{output['source_end_line']} of {source_total}]"
                )
                if total_lines == 0:
                    parts.append("[expanded output is empty]")
                else:
                    parts.append(
                        f"[expanded lines {start_line}-{end_line} of {total_lines}]"
                    )
        elif total_lines == 0:
            parts.append("[empty file]")
        elif start_line == 0:
            parts.append(f"[no lines returned, file has {total_lines} lines]")
        else:
            parts.append(f"[lines {start_line}-{end_line} of {total_lines}]")
        if truncated_lines:
            parts.append(f"[{truncated_lines} long lines truncated]")

        return "\n".join(parts)
