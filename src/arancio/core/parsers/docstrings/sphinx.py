"""Parser for Sphinx-style docstrings."""

import re

from arancio.core.parsers.docstrings.base import DocstringParser
from arancio.core.parsers.docstrings.patterns import SPHINX_PARAMETER_ENTRY_PATTERN


class SphinxDocstringParser(DocstringParser):
    """Read the ``:param name:`` fields of a Sphinx-style docstring.

    Fields are ``
    :param name: text`` or ``
    :param type name: text``; indented lines continue the previous description, and any
        other field or a blank line ends it.
    """

    _parameter_entry_pattern: re.Pattern[str] = SPHINX_PARAMETER_ENTRY_PATTERN

    @classmethod
    def parse(cls, docstring: str) -> dict[str, str]:
        """Read the parameter descriptions of a Sphinx-style docstring.

        Args:
            docstring: the dedented docstring.

        Returns:
            Each documented parameter mapped to its description on one line.
        """
        descriptions: dict[str, str] = {}
        current: str | None = None
        for line in docstring.splitlines():
            match: re.Match[str] | None = cls._parameter_entry_pattern.match(line)
            if match is not None:
                current = match.group(1)
                descriptions[current] = match.group(2)
            elif line.startswith(":") or not line.strip():
                current = None
            elif current is not None and line[0].isspace():
                descriptions[current] += f" {line.strip()}"
        return descriptions
