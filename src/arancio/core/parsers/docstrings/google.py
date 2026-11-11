"""Parser for Google-style docstrings."""

import re

from arancio.core.parsers.docstrings.base import DocstringParser
from arancio.core.parsers.docstrings.patterns import (
    GOOGLE_PARAMETER_ENTRY_PATTERN,
    GOOGLE_PARAMETERS_SECTION_PATTERN,
)


class GoogleDocstringParser(DocstringParser):
    """Read the ``Args:`` section of a Google-style docstring.

    The section is the ``Args:`` line and the indented lines after it. Entries are
    ``name: text`` or ``name (type): text`` indented four spaces; more indented lines
    continue the previous description.
    """

    _parameters_section_pattern: re.Pattern[str] = GOOGLE_PARAMETERS_SECTION_PATTERN
    _parameter_entry_pattern: re.Pattern[str] = GOOGLE_PARAMETER_ENTRY_PATTERN

    @classmethod
    def parse(cls, docstring: str) -> dict[str, str]:
        """Read the parameter descriptions of a Google-style docstring.

        Args:
            docstring: the dedented docstring.

        Returns:
            Each documented parameter mapped to its description on one line.
        """
        section: re.Match[str] | None = cls._parameters_section_pattern.search(
            docstring
        )
        if section is None:
            return {}
        descriptions: dict[str, str] = {}
        current: str | None = None
        for line in section.group(1).splitlines():
            match: re.Match[str] | None = cls._parameter_entry_pattern.match(line)
            if match is not None:
                current = match.group(1)
                descriptions[current] = match.group(2)
            elif current is not None and line.strip():
                descriptions[current] += f" {line.strip()}"
        return descriptions
