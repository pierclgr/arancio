"""Abstract docstring parser interface."""

import re
from abc import ABC, abstractmethod

from arancio.core.parsers.base import Parser


class DocstringParser(Parser, ABC):
    """Read the parameter descriptions out of a docstring in one style.

    Each concrete parser understands one docstring convention and returns an empty
    mapping when the docstring does not follow it.

    Attributes:
        _parameters_section_pattern: the regex matching the whole parameter section,
            whose first group is the section body; ``None`` for styles whose entries
            stand alone (Sphinx).
        _parameter_entry_pattern: the regex matching one parameter entry; its first
            group is the parameter name.
    """

    _parameters_section_pattern: re.Pattern[str] | None = None
    _parameter_entry_pattern: re.Pattern[str]

    @classmethod
    @abstractmethod
    def parse(cls, docstring: str) -> dict[str, str]:
        """Read the parameter descriptions of a docstring.

        Args:
            docstring: the dedented docstring.

        Returns:
            Each documented parameter mapped to its description on one line; empty
            when the docstring has no parameter section in this style.
        """
        raise NotImplementedError("Subclasses must implement this method")
