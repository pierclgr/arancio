"""Shared parser interface."""

from abc import ABC, abstractmethod
from typing import Any


class Parser(ABC):
    """Abstract base class for parsers.

    A parser is a component that processes and transforms input data according to a
    specific format or schema. Subclasses must implement the parse method to define the
    specific parsing logic.
    """

    @classmethod
    @abstractmethod
    def parse(cls, **kwargs) -> Any:
        """Parse input data and return the processed result.

        Args:
            **kwargs: Keyword arguments containing the input data to be parsed
                and any additional parsing parameters.

        Returns:
            Any: The parsed and processed data.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
