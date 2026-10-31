"""Abstract runnable command interface."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Command(ABC):
    """Bind typed words and run a command with Application as its only service.

    Concrete commands adapt input to Application operations. Defaults, extra-word
    behavior and joined arguments retain their existing rules. This module has no
    dependency on prompt routing, UI or live backend managers.

    Attributes:
        name: the registry name typed after the slash.
        description: the short command description.
        joins_arguments: whether trailing words form the last prompt argument.
        INJECTABLE_COMMAND_PARAMETERS: names supplied by the executor.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    joins_arguments: ClassVar[bool] = False
    INJECTABLE_COMMAND_PARAMETERS: ClassVar[frozenset[str]] = frozenset({"application"})

    @classmethod
    def run(cls, args: list[str], **kwargs) -> Any:
        """Bind and coerce the arguments, then run :meth:`execute`.

        Args:
            args: the raw prompt words.
            **kwargs: the Application facade,
                keyed by their parameter names.

        Returns:
            Whatever :meth:`execute` returns.
        """
        arguments = cls.__validate_args(args, **kwargs)
        return cls.execute(**arguments)

    @classmethod
    def __validate_args(cls, args: list[str], **kwargs) -> dict[str, Any]:
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
            **kwargs: the Application facade, keyed as application.

        Returns:
            The keyword arguments to pass to :meth:`execute`.

        Raises:
            TypeError: when a value cannot be coerced to its annotated type.
        """
        signature = inspect.signature(cls.execute)
        prompt_parameter_names = [
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name not in cls.INJECTABLE_COMMAND_PARAMETERS
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
