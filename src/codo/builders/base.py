"""Shared builder interface."""

from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    """Abstract base class for builders.

    A builder is a component that constructs or assembles something according to a
    specific pattern or configuration. Subclasses must implement the build method to
    define the specific construction logic.
    """

    @classmethod
    @abstractmethod
    def build(cls, **kwargs) -> Any:
        """Builds and returns the constructed object.

        Args:
            **kwargs: Keyword arguments specific to the builder implementation.

        Returns:
            Any: The constructed object or result of the build process.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
