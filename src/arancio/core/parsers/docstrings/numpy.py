"""Parser for NumPy-style docstrings."""

import re

from arancio.core.parsers.docstrings.base import DocstringParser
from arancio.core.parsers.docstrings.patterns import (
    NUMPY_PARAMETER_ENTRY_PATTERN,
    NUMPY_PARAMETERS_SECTION_PATTERN,
)


class NumpyDocstringParser(DocstringParser):
    """Read the ``Parameters`` section of a NumPy-style docstring.

    The section is a ``Parameters`` heading underlined with dashes, up to the next
    heading. Each entry is a ``name`` or ``name : type`` line at column 0, with its
    description indented on the following lines.
    """

    _parameters_section_pattern: re.Pattern[str] = NUMPY_PARAMETERS_SECTION_PATTERN
    _parameter_entry_pattern: re.Pattern[str] = NUMPY_PARAMETER_ENTRY_PATTERN

    @classmethod
    def parse(cls, docstring: str) -> dict[str, str]:
        """Read the parameter descriptions of a NumPy-style docstring.

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
            if not line.strip():
                continue
            if line[0].isspace():
                if current is not None:
                    descriptions[current] = f"{descriptions[current]} {line.strip()}"
                continue
            match: re.Match[str] | None = cls._parameter_entry_pattern.match(line)
            current = match.group(1) if match is not None else None
            if current is not None:
                descriptions[current] = ""
        return {name: text.strip() for name, text in descriptions.items()}
