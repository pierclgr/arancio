"""Tests for slash commands."""

import pytest

from arancio.commands.base import BaseCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.core.messages import AssistantMessage


class _IntArgCommand(BaseCommand):
    """Test command whose single argument must be an int."""

    name = "int-arg"
    description = "Echo an integer argument."

    @classmethod
    def execute(cls, count: int) -> int:
        """Return the received count unchanged."""
        return count


def test_hello_world_greets_the_name_the_given_number_of_times() -> None:
    """The int argument is coerced from its prompt word and applied."""
    result = HelloWorldCommand.run("Sam", "2")

    assert result == AssistantMessage(content="Hello World, Sam\nHello World, Sam")


def test_hello_world_rejects_extra_arguments() -> None:
    """A surplus argument fails against the two-parameter signature."""
    with pytest.raises(TypeError):
        HelloWorldCommand.run("Sam", "2", "extra")


def test_hello_world_requires_all_arguments() -> None:
    """A missing mandatory argument fails."""
    with pytest.raises(TypeError):
        HelloWorldCommand.run("Sam")


def test_run_coerces_argument_to_annotated_type() -> None:
    """A numeric string is coerced to the parameter's int annotation."""
    assert _IntArgCommand.run("5") == 5


def test_run_rejects_argument_that_cannot_be_coerced() -> None:
    """A value that cannot convert to the annotated type raises TypeError."""
    with pytest.raises(TypeError):
        _IntArgCommand.run("abc")


def test_base_command_cannot_be_instantiated() -> None:
    """BaseCommand is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCommand()
