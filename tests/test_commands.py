"""Tests for slash commands."""

from typing import Any

import pytest

from arancio.commands.base import BaseCommand
from arancio.commands.exit import ExitCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.core.messages import AssistantMessage


class _RecordingApplication:
    """Application stub recording whether it was asked to exit."""

    def __init__(self) -> None:
        """Start with no recorded exit."""
        self.exit_called = False

    def exit(self) -> None:
        """Record that an exit was requested."""
        self.exit_called = True


class _IntArgCommand(BaseCommand):
    """Test command whose single argument must be an int."""

    name = "int-arg"
    description = "Echo an integer argument."

    @classmethod
    def execute(cls, application: Any, count: int) -> int:
        """Return the received count unchanged.

        Args:
            application: the running application (unused).
            count: the integer argument bound to the prompt word.

        Returns:
            The received count.
        """
        return count


def test_hello_world_greets_the_name_the_given_number_of_times() -> None:
    """The int argument is coerced from its keyword and applied."""
    result = HelloWorldCommand.run(name="Sam", times="2")

    assert result == AssistantMessage(content="Hello World, Sam\nHello World, Sam")


def test_hello_world_defaults_times_when_omitted() -> None:
    """The optional times argument defaults to a single greeting."""
    result = HelloWorldCommand.run(name="Sam")

    assert result == AssistantMessage(content="Hello World, Sam")


def test_hello_world_ignores_unneeded_application_argument() -> None:
    """An application argument the command does not declare is absorbed by kwargs."""
    result = HelloWorldCommand.run(application=_RecordingApplication(), name="Sam")

    assert result == AssistantMessage(content="Hello World, Sam")


def test_hello_world_requires_the_mandatory_argument() -> None:
    """A missing mandatory argument fails."""
    with pytest.raises(TypeError):
        HelloWorldCommand.run()


def test_run_coerces_argument_to_annotated_type() -> None:
    """A numeric string is coerced to the parameter's int annotation."""
    assert _IntArgCommand.run(application=_RecordingApplication(), count="5") == 5


def test_run_rejects_argument_that_cannot_be_coerced() -> None:
    """A value that cannot convert to the annotated type raises TypeError."""
    with pytest.raises(TypeError):
        _IntArgCommand.run(application=_RecordingApplication(), count="abc")


def test_int_arg_command_rejects_unexpected_keyword_argument() -> None:
    """A command without a kwargs catch-all rejects an unknown keyword argument."""
    with pytest.raises(TypeError):
        _IntArgCommand.run(application=_RecordingApplication(), count="5", extra="oops")


def test_exit_command_quits_the_application() -> None:
    """The exit command exits the application and returns nothing."""
    application = _RecordingApplication()

    result = ExitCommand.run(application=application)

    assert result is None
    assert application.exit_called is True


def test_base_command_cannot_be_instantiated() -> None:
    """BaseCommand is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCommand()
