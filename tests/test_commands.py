"""Tests for slash commands."""

from typing import Any

import pytest

from arancio.commands.base import BaseCommand
from arancio.commands.exit import ExitCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.commands.model import ModelCommand


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

    assert result == "Hello World, Sam\nHello World, Sam"


def test_hello_world_defaults_times_when_omitted() -> None:
    """The optional times argument defaults to a single greeting."""
    result = HelloWorldCommand.run(name="Sam")

    assert result == "Hello World, Sam"


def test_hello_world_ignores_unneeded_application_argument() -> None:
    """An application argument the command does not declare is absorbed by kwargs."""
    result = HelloWorldCommand.run(application=_RecordingApplication(), name="Sam")

    assert result == "Hello World, Sam"


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


class _ModelApplication:
    """Application stub recording the model id it was asked to display."""

    def __init__(self) -> None:
        """Start with no displayed model id."""
        self.displayed_model_id: str | None = None

    def set_displayed_model_id(self, model_id: str) -> None:
        """Record the model id the toolbar was asked to display.

        Args:
            model_id: the model id to display.
        """
        self.displayed_model_id = model_id


class _FakeModelSettings:
    """Minimal settings stub exposing the fields the model command reads/writes."""

    def __init__(self, provider: str | None, model_name: str | None) -> None:
        """Store the initial provider and model name.

        Args:
            provider: the initial provider, or ``None`` to leave it unset.
            model_name: the initial model name, or ``None`` to leave it unset.
        """
        self.provider = provider
        self.model_name = model_name

    @property
    def model_id(self) -> str:
        """Build the model id from the current provider and model name.

        Returns:
            The joined ``provider/model_name`` model id.

        Raises:
            ValueError: when the provider is not configured.
            ValueError: when the model name is not configured.
        """
        if not self.provider:
            raise ValueError("No provider configured.")
        if not self.model_name:
            raise ValueError("No model name configured.")
        return f"{self.provider}/{self.model_name}"


class _FakeSettingsManager:
    """Settings-manager stub recording ``apply``/``save`` calls without disk I/O."""

    def __init__(
        self, provider: str | None = "openai", model_name: str | None = "gpt-4o"
    ) -> None:
        """Build a settings stub with the given provider and model name.

        Args:
            provider: the initial provider, or ``None`` to leave it unset.
            model_name: the initial model name, or ``None`` to leave it unset.
        """
        self.settings = _FakeModelSettings(provider, model_name)
        self.applied = False
        self.saved = False

    def apply(self) -> None:
        """Record that the settings were applied to the live objects."""
        self.applied = True

    def save(self) -> None:
        """Record that the settings were persisted to disk."""
        self.saved = True


def test_model_command_sets_model_name_and_confirms() -> None:
    """The model command applies the new name and confirms the full model id."""
    application = _ModelApplication()
    settings_manager = _FakeSettingsManager()

    result = ModelCommand.run(
        application=application, settings_manager=settings_manager, model_name="gpt-5"
    )

    assert settings_manager.settings.model_name == "gpt-5"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_model_id == "openai/gpt-5"
    assert result == "Model set to openai/gpt-5"


def test_model_command_requires_a_configured_provider() -> None:
    """Setting the model name without a provider configured raises, unpersisted."""
    settings_manager = _FakeSettingsManager(provider=None, model_name="gpt-4o")

    with pytest.raises(ValueError):
        ModelCommand.run(
            application=_ModelApplication(),
            settings_manager=settings_manager,
            model_name="gpt-5",
        )

    assert settings_manager.applied is False
    assert settings_manager.saved is False
    # model_name is restored to its previous value rather than left dangling
    assert settings_manager.settings.model_name == "gpt-4o"


def test_base_command_cannot_be_instantiated() -> None:
    """BaseCommand is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCommand()
