"""Abstract runnable command interface."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseCommand(ABC):
    """Base class for slash commands parsed from the prompt.

    Subclasses set :attr:`name` and :attr:`description` and implement
    :meth:`execute` with its own named parameters. The concrete :meth:`run`
    coerces the prompt words to :meth:`execute`'s parameter types via
    :meth:`_validate_args` (rejecting a wrong number of arguments or a value that
    cannot be coerced), then forwards them positionally.

    Attributes:
        name: the command name typed after the leading slash.
        description: a short natural-language summary of the command.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    def run(cls, *args) -> Any:
        """Validate and coerce the arguments, then run :meth:`execute`.

        Args:
            *args: the prompt words, forwarded to :meth:`execute`.

        Returns:
            Whatever :meth:`execute` returns.
        """
        arguments = cls._validate_args(*args)
        return cls.execute(*arguments)

    @classmethod
    def _validate_args(cls, *args) -> list[Any]:
        """Bind and coerce the prompt words to :meth:`execute`'s parameters.

        Each word is bound to a parameter and converted to its type annotation
        (e.g. ``"3"`` to ``3`` for an ``int`` parameter). A missing mandatory
        parameter, a surplus argument, or a value that cannot be converted to
        the annotated type raises :class:`TypeError`, which propagates to the
        caller.

        Args:
            *args: the prompt words parsed from the prompt.

        Returns:
            The coerced arguments, in the order :meth:`execute` expects.

        Raises:
            TypeError: when ``args`` do not match :meth:`execute`'s signature
                in arity, or a value cannot be coerced to its annotated type.
        """
        signature = inspect.signature(cls.execute)
        bound = signature.bind(*args)
        arguments = []
        for parameter_name, value in bound.arguments.items():
            annotation = signature.parameters[parameter_name].annotation
            if annotation is not inspect.Parameter.empty and isinstance(
                annotation, type
            ):
                try:
                    value = annotation(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"argument '{parameter_name}' must be "
                        f"{annotation.__name__}, got {value!r}"
                    ) from exc
            arguments.append(value)
        return arguments

    @classmethod
    @abstractmethod
    def execute(cls, *args) -> Any:
        """Execute the command action and return its result.

        Args:
            *args: the prompt words bound to the command's parameters.

        Returns:
            The command result.

        Raises:
            NotImplementedError: when the subclass does not implement it.
        """
        raise NotImplementedError("Subclasses must implement this method.")
