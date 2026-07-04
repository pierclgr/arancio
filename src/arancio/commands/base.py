"""Abstract runnable command interface."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseCommand(ABC):
    """Base class for slash commands parsed from the prompt.

    Subclasses set :attr:`name` and :attr:`description` and implement
    :meth:`execute` with its own named parameters. A command that needs to act on
    the running application declares a parameter named ``application``, which
    :class:`arancio.prompt.actions.executor.ActionExecutor` always supplies;
    commands that don't need it declare a ``**kwargs`` catch-all to absorb it (and
    any other argument they don't care about) instead of listing it explicitly.
    The concrete :meth:`run` receives the arguments as keywords, coerces them to
    :meth:`execute`'s parameter types via :meth:`_validate_args` (rejecting a
    missing mandatory argument or a value that cannot be coerced), then forwards
    them all to :meth:`execute`.

    Attributes:
        name: the command name typed after the leading slash.
        description: a short natural-language summary of the command.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    def run(cls, **kwargs) -> Any:
        """Validate and coerce the arguments, then run :meth:`execute`.

        Args:
            **kwargs: the arguments keyed by :meth:`execute`'s parameter names,
                including ``application``.

        Returns:
            Whatever :meth:`execute` returns.
        """
        arguments = cls._validate_args(**kwargs)
        return cls.execute(**arguments)

    @classmethod
    def _validate_args(cls, **kwargs) -> dict[str, Any]:
        """Coerce the keyword arguments to :meth:`execute`'s parameter types.

        ``application`` is forwarded untouched; each argument bound to a named,
        annotated parameter of :meth:`execute` is converted to its type
        annotation (e.g. ``"3"`` to ``3`` for an ``int`` parameter). An argument
        absorbed by :meth:`execute`'s ``**kwargs`` catch-all, if it has one, is
        also forwarded untouched. A value that cannot be converted to its
        annotated type raises :class:`TypeError`; a missing mandatory argument or
        an unexpected keyword argument raises :class:`TypeError` when
        :meth:`execute` is actually called with the result.

        Args:
            **kwargs: the arguments keyed by :meth:`execute`'s parameter names.

        Returns:
            The keyword arguments to pass to :meth:`execute`.

        Raises:
            TypeError: when a value cannot be coerced to its annotated type.
        """
        signature = inspect.signature(cls.execute)
        arguments = {}
        for parameter_name, value in kwargs.items():
            parameter = signature.parameters.get(parameter_name)
            annotation = parameter.annotation if parameter else inspect.Parameter.empty
            if (
                parameter_name != "application"
                and annotation is not inspect.Parameter.empty
                and isinstance(annotation, type)
            ):
                try:
                    value = annotation(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"argument '{parameter_name}' must be "
                        f"{annotation.__name__}, got {value!r}"
                    ) from exc
            arguments[parameter_name] = value
        return arguments

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> Any:
        """Execute the command action and return its result.

        Args:
            **kwargs: the arguments keyed by the command's parameter names,
                including ``application``.

        Returns:
            The command result.

        Raises:
            NotImplementedError: when the subclass does not implement it.
        """
        raise NotImplementedError("Subclasses must implement this method.")
