"""Result rendering for plugin-defined tools."""

import json

from arancio.core.parsers.tool_result.base import ToolResultParser


class PluginToolResultParser(ToolResultParser):
    """Keep plugin result content and render dictionaries as JSON."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render a JSON-compatible dictionary for display.

        Args:
            output: the dictionary returned by a plugin tool.

        Returns:
            JSON text with non-ASCII characters kept readable.
        """
        return json.dumps(output, ensure_ascii=False)
