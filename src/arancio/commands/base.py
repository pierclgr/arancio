"""Abstract runnable command interface."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from arancio.prompt.actions.constants import INJECTABLE_COMMAND_PARAMETERS


class BaseCommand(ABC):
    """Base class for slash commands parsed from the prompt.

    Subclasses set :attr:`name` and :attr:`description` and implement
    :meth:`execute` with its own named parameters. A command declares a
    parameter named ``application``, ``settings_manager`` and/or ``agent`` to
    have :class:`arancio.prompt.actions.executor.ActionExecutor` supply the
    running application, the settings manager and/or the agent; a command that
    needs none of them declares a ``**kwargs`` catch-all to absorb them (and
    any other argument it doesn't care about) instead of listing them
    explicitly. The concrete
    :meth:`run` receives the raw prompt words plus the injectable objects,
    binds the words to :meth:`execute`'s parameters and coerces them to their
    types via :meth:`_validate_args` (rejecting a missing mandatory argument or
    a value that cannot be coerced), then forwards them all to :meth:`execute`.

    Attributes:
        name: the command name typed after the leading slash.
        description: a short natural-language summary of the command.
        joins_arguments: when true, the prompt words beyond the command's
            parameters are joined with a space into its last parameter instead
            of being dropped.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    joins_arguments: ClassVar[bool] = False

    @classmethod
    def run(cls, args: list[str], **kwargs) -> Any:
        """Bind and coerce the arguments, then run :meth:`execute`.

        Args:
            args: the raw prompt words.
            **kwargs: the injectable objects (``application``, ``agent``, …),
                keyed by their parameter names.

        Returns:
            Whatever :meth:`execute` returns.
        """
        arguments = cls._validate_args(args, **kwargs)
        return cls.execute(**arguments)

    @classmethod
    def _validate_args(cls, args: list[str], **kwargs) -> dict[str, Any]:
        """Bind the prompt words to :meth:`execute`'s parameters and coerce them.

        Each word is matched, in order, to the next prompt parameter
        :meth:`execute` declares (excluding the injectable ones); a word beyond
        the number of declared parameters is dropped rather than rejected,
        unless :attr:`joins_arguments` is set, in which case the last parameter
        receives every remaining word joined with a space. Each bound word is
        converted to its parameter's type annotation (e.g. ``"3"`` to ``3`` for
        an ``int`` parameter). An injectable object is forwarded untouched, and
        only when :meth:`execute` declares a parameter for it. A value that
        cannot be converted to its annotated type raises :class:`TypeError`; a
        missing mandatory argument raises :class:`TypeError` when
        :meth:`execute` is actually called with the result.

        Args:
            args: the raw prompt words.
            **kwargs: the injectable objects, keyed by their parameter names.

        Returns:
            The keyword arguments to pass to :meth:`execute`.

        Raises:
            TypeError: when a value cannot be coerced to its annotated type.
        """
        signature = inspect.signature(cls.execute)
        prompt_parameter_names = [
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name not in INJECTABLE_COMMAND_PARAMETERS
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
        if cls.joins_arguments and len(args) > len(prompt_parameter_names):
            last = len(prompt_parameter_names) - 1
            args = [*args[:last], " ".join(args[last:])]
        arguments = {}
        for parameter_name, value in zip(prompt_parameter_names, args):
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
            arguments[parameter_name] = value
        for name, value in kwargs.items():
            if name in signature.parameters:
                arguments[name] = value
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
