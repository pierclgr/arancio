"""Tests for slash commands."""

from pathlib import Path
from typing import Any

import pytest

from arancio.commands.base import BaseCommand
from arancio.commands.cd import CdCommand
from arancio.commands.clear import ClearCommand
from arancio.commands.effort import EffortCommand
from arancio.commands.exit import ExitCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.commands.model import ModelCommand
from arancio.commands.permissions import PermissionsCommand
from arancio.commands.provider import ProviderCommand
from arancio.core.constants.litellm import LITELLM_PROVIDER_NAMES
from arancio.core.permissions.types import PermissionCategory, PermissionLevel


class _RecordingApplication:
    """Application stub recording whether it was asked to exit or clear the log."""

    def __init__(self) -> None:
        """Start with no recorded exit or log clear."""
        self.exit_called = False
        self.clear_log_called = False

    def exit(self) -> None:
        """Record that an exit was requested."""
        self.exit_called = True

    def clear_log(self) -> None:
        """Record that the log was asked to be cleared."""
        self.clear_log_called = True


class _RecordingAgent:
    """Agent stub recording whether its history was asked to be cleared."""

    def __init__(self) -> None:
        """Start with no recorded history clear."""
        self.clear_history_called = False

    def clear_history(self) -> None:
        """Record that the history was asked to be cleared."""
        self.clear_history_called = True


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


def test_clear_command_clears_history_and_log() -> None:
    """The clear command empties the agent's history and the app's log silently."""
    agent = _RecordingAgent()
    application = _RecordingApplication()

    result = ClearCommand.run(agent=agent, application=application)

    assert agent.clear_history_called is True
    assert application.clear_log_called is True
    assert result is None


class _CdApplication:
    """Application stub applying the real working-directory resolution rules."""

    def __init__(self, working_directory: Path) -> None:
        """Start held at ``working_directory``.

        Args:
            working_directory: the directory the stub starts in.
        """
        self.working_directory = working_directory

    def set_working_directory(self, path: Path | str) -> None:
        """Resolve and validate ``path``, mirroring ``App.set_working_directory``.

        Args:
            path: the requested directory, absolute or relative.

        Raises:
            ValueError: when the resolved path does not exist, or exists but is
                not a directory.
        """
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.working_directory / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            raise ValueError(f"{resolved} does not exist")
        if not resolved.is_dir():
            raise ValueError(
                f"{resolved} not a directory: did you mean {resolved.parent}?"
            )
        self.working_directory = resolved


def test_cd_command_moves_to_an_absolute_path(tmp_path: Path) -> None:
    """An absolute path is used as-is."""
    target = tmp_path / "target"
    target.mkdir()
    application = _CdApplication(working_directory=tmp_path)

    result = CdCommand.run(application=application, path=str(target))

    assert application.working_directory == target.resolve()
    assert result == f"Working directory set to {target.resolve()}"


def test_cd_command_resolves_a_relative_path_against_the_current_directory(
    tmp_path: Path,
) -> None:
    """A relative path is resolved against the working directory currently set."""
    nested = tmp_path / "outer" / "inner"
    nested.mkdir(parents=True)
    application = _CdApplication(working_directory=tmp_path / "outer")

    result = CdCommand.run(application=application, path="inner")

    assert application.working_directory == nested.resolve()
    assert result == f"Working directory set to {nested.resolve()}"


def test_cd_command_moves_to_a_path_containing_spaces(tmp_path: Path) -> None:
    """A path with spaces works once the prompt splitter has kept it in one word."""
    target = tmp_path / "test" / "of path"
    target.mkdir(parents=True)
    application = _CdApplication(working_directory=tmp_path)

    result = CdCommand.run(application=application, path="test/of path")

    assert application.working_directory == target.resolve()
    assert result == f"Working directory set to {target.resolve()}"


def test_cd_command_reports_a_missing_path_as_not_existing(tmp_path: Path) -> None:
    """A path that does not exist reports that it does not exist."""
    application = _CdApplication(working_directory=tmp_path)

    with pytest.raises(ValueError, match="does not exist"):
        CdCommand.run(application=application, path="missing")

    assert application.working_directory == tmp_path


def test_cd_command_reports_a_file_path_as_not_a_directory(tmp_path: Path) -> None:
    """A path pointing at an existing file keeps the not-a-directory wording."""
    (tmp_path / "afile").write_text("x")
    application = _CdApplication(working_directory=tmp_path)

    with pytest.raises(ValueError, match="not a directory"):
        CdCommand.run(application=application, path="afile")

    assert application.working_directory == tmp_path


class _ModelApplication:
    """Application stub recording the model id or effort it was asked to display."""

    def __init__(self) -> None:
        """Start with no displayed model id or effort."""
        self.displayed_model_id: str | None = None
        self.displayed_effort: str | None = None

    def set_displayed_effort(self, effort: str | None) -> None:
        """Record the effort the toolbar was asked to display.

        Args:
            effort: the effort to display, or ``None`` when thinking is
                disabled.
        """
        self.displayed_effort = effort

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
    def provider(self) -> str | None:
        """Return the currently configured provider.

        Returns:
            The lowercased provider prefix, or ``None`` when not yet
            configured.
        """
        return self._provider

    @provider.setter
    def provider(self, value: str | None) -> None:
        """Validate and set the provider, matched case-insensitively.

        Mirrors the real :attr:`Settings.provider` setter.

        Args:
            value: the provider name, in any case, or ``None`` to unset it.

        Raises:
            ValueError: when ``value`` is not a valid LiteLLM provider name.
        """
        if value is None:
            self._provider = None
            return
        lowered = value.lower()
        if lowered not in LITELLM_PROVIDER_NAMES:
            raise ValueError(f"Unknown provider: {value!r}.")
        self._provider = lowered

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


def test_provider_command_sets_provider_and_confirms() -> None:
    """Setting the provider applies, persists and refreshes the toolbar's model id."""
    application = _ModelApplication()
    settings_manager = _FakeSettingsManager(provider="openai", model_name="gpt-4o")

    result = ProviderCommand.run(
        application=application, settings_manager=settings_manager, provider="anthropic"
    )

    assert settings_manager.settings.provider == "anthropic"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_model_id == "anthropic/gpt-4o"
    assert result == "Provider set to anthropic"


def test_provider_command_without_model_name_skips_toolbar_update() -> None:
    """Setting the provider alone still applies/persists, without a model id to show."""
    application = _ModelApplication()
    settings_manager = _FakeSettingsManager(provider=None, model_name=None)

    result = ProviderCommand.run(
        application=application, settings_manager=settings_manager, provider="openai"
    )

    assert settings_manager.settings.provider == "openai"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_model_id is None
    assert result == "Provider set to openai"


def test_provider_command_keeps_model_name_unchanged() -> None:
    """Setting the provider does not touch the current model name."""
    settings_manager = _FakeSettingsManager(provider="openai", model_name="gpt-4o")

    ProviderCommand.run(
        application=_ModelApplication(),
        settings_manager=settings_manager,
        provider="anthropic",
    )

    assert settings_manager.settings.model_name == "gpt-4o"


def test_provider_command_normalizes_case_in_confirmation() -> None:
    """The confirmation and toolbar use the resolved, lowercased provider."""
    application = _ModelApplication()
    settings_manager = _FakeSettingsManager(provider="openai", model_name="gpt-4o")

    result = ProviderCommand.run(
        application=application, settings_manager=settings_manager, provider="OpenAI"
    )

    assert settings_manager.settings.provider == "openai"
    assert application.displayed_model_id == "openai/gpt-4o"
    assert result == "Provider set to openai"


def test_provider_command_rejects_invalid_provider() -> None:
    """An unrecognized provider raises, unpersisted, leaving the old value intact."""
    application = _ModelApplication()
    settings_manager = _FakeSettingsManager(provider="openai", model_name="gpt-4o")

    with pytest.raises(ValueError):
        ProviderCommand.run(
            application=application,
            settings_manager=settings_manager,
            provider="not-a-real-provider",
        )

    assert settings_manager.applied is False
    assert settings_manager.saved is False
    assert settings_manager.settings.provider == "openai"
    assert application.displayed_model_id is None


class _FakeEffortSettings:
    """Minimal settings stub exposing the fields the effort command reads/writes."""

    def __init__(
        self,
        thinking_effort: str = "medium",
        provider: str | None = "openai",
        model_name: str | None = "gpt-4o",
    ) -> None:
        """Store the initial thinking effort, provider and model name.

        Args:
            thinking_effort: the initial thinking effort.
            provider: the initial provider, or ``None`` to leave it unset.
            model_name: the initial model name, or ``None`` to leave it unset.
        """
        self.thinking_effort = thinking_effort
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


class _FakeEffortSettingsManager:
    """Settings-manager stub recording ``apply``/``save`` calls without disk I/O."""

    def __init__(
        self,
        thinking_effort: str = "medium",
        provider: str | None = "openai",
        model_name: str | None = "gpt-4o",
    ) -> None:
        """Build a settings stub with the given thinking effort, provider and name.

        Args:
            thinking_effort: the initial thinking effort.
            provider: the initial provider, or ``None`` to leave it unset.
            model_name: the initial model name, or ``None`` to leave it unset.
        """
        self.settings = _FakeEffortSettings(thinking_effort, provider, model_name)
        self.applied = False
        self.saved = False

    def apply(self) -> None:
        """Record that the settings were applied to the live objects."""
        self.applied = True

    def save(self) -> None:
        """Record that the settings were persisted to disk."""
        self.saved = True


def test_effort_command_sets_thinking_effort_and_confirms() -> None:
    """The effort command applies the new value, refreshes the toolbar and confirms."""
    application = _ModelApplication()
    settings_manager = _FakeEffortSettingsManager()

    result = EffortCommand.run(
        application=application, settings_manager=settings_manager, level="high"
    )

    assert settings_manager.settings.thinking_effort == "high"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_effort == "high"
    assert result == "Thinking effort set to high"


def test_effort_command_accepts_any_free_form_value() -> None:
    """Any text is accepted as the effort level, with no fixed set enforced."""
    settings_manager = _FakeEffortSettingsManager()

    result = EffortCommand.run(
        application=_ModelApplication(),
        settings_manager=settings_manager,
        level="ultra-mega",
    )

    assert settings_manager.settings.thinking_effort == "ultra-mega"
    assert result == "Thinking effort set to ultra-mega"


@pytest.mark.parametrize("keyword", ["null", "NULL", "Null"])
def test_effort_command_null_disables_thinking(keyword: str) -> None:
    """The word "null", in any case, sets the thinking effort to None."""
    application = _ModelApplication()
    settings_manager = _FakeEffortSettingsManager()

    result = EffortCommand.run(
        application=application, settings_manager=settings_manager, level=keyword
    )

    assert settings_manager.settings.thinking_effort is None
    assert application.displayed_effort is None
    assert result == "Thinking effort set to null"


def test_effort_command_requires_a_configured_model() -> None:
    """Setting the effort without a configured model raises, unpersisted."""
    application = _ModelApplication()
    settings_manager = _FakeEffortSettingsManager(provider=None, model_name=None)

    with pytest.raises(ValueError):
        EffortCommand.run(
            application=application, settings_manager=settings_manager, level="high"
        )

    assert settings_manager.applied is False
    assert settings_manager.saved is False
    assert settings_manager.settings.thinking_effort == "medium"
    assert application.displayed_effort is None


class _FakePermissionsSettings:
    """Minimal settings stub exposing the field the permissions command reads/writes."""

    def __init__(
        self, permissions: dict[PermissionCategory, PermissionLevel] | None = None
    ) -> None:
        """Store the initial permission grants.

        Args:
            permissions: the initial category-to-level mapping, or ``None``
                to default every category to :attr:`PermissionLevel.ASK`.
        """
        self.permissions = permissions or {
            category: PermissionLevel.ASK for category in PermissionCategory
        }


class _FakePermissionsSettingsManager:
    """Settings-manager stub recording ``apply``/``save`` calls without disk I/O."""

    def __init__(
        self, permissions: dict[PermissionCategory, PermissionLevel] | None = None
    ) -> None:
        """Build a settings stub with the given permission grants.

        Args:
            permissions: the initial category-to-level mapping, or ``None``
                to default every category to :attr:`PermissionLevel.ASK`.
        """
        self.settings = _FakePermissionsSettings(permissions)
        self.applied = False
        self.saved = False

    def apply(self) -> None:
        """Record that the settings were applied to the live objects."""
        self.applied = True

    def save(self) -> None:
        """Record that the settings were persisted to disk."""
        self.saved = True


def test_permissions_command_reports_current_level_without_a_level_argument() -> None:
    """Omitting the level reports the category's currently set level, unpersisted."""
    settings_manager = _FakePermissionsSettingsManager()

    result = PermissionsCommand.run(category="read", settings_manager=settings_manager)

    assert result == "read permission level: ask"
    assert settings_manager.applied is False
    assert settings_manager.saved is False


def test_permissions_command_sets_level_and_confirms() -> None:
    """Giving both arguments replaces the level, applies and persists it."""
    settings_manager = _FakePermissionsSettingsManager()

    result = PermissionsCommand.run(
        category="read", level="auto", settings_manager=settings_manager
    )

    assert result == "read permission level set to auto"
    assert (
        settings_manager.settings.permissions[PermissionCategory.READ]
        == PermissionLevel.AUTO
    )
    assert settings_manager.applied is True
    assert settings_manager.saved is True


@pytest.mark.parametrize("category", ["READ", "Read", "read"])
def test_permissions_command_category_name_is_case_insensitive(
    category: str,
) -> None:
    """The permission category name is matched case-insensitively."""
    settings_manager = _FakePermissionsSettingsManager()

    result = PermissionsCommand.run(
        category=category, settings_manager=settings_manager
    )

    assert result == "read permission level: ask"


@pytest.mark.parametrize("level", ["AUTO", "Auto", "auto"])
def test_permissions_command_level_is_case_insensitive(level: str) -> None:
    """The permission level is matched case-insensitively."""
    settings_manager = _FakePermissionsSettingsManager()

    result = PermissionsCommand.run(
        category="read", level=level, settings_manager=settings_manager
    )

    assert result == "read permission level set to auto"


def test_permissions_command_rejects_unknown_permission() -> None:
    """An unknown permission category raises, unpersisted."""
    settings_manager = _FakePermissionsSettingsManager()

    with pytest.raises(ValueError):
        PermissionsCommand.run(category="nope", settings_manager=settings_manager)

    assert settings_manager.applied is False
    assert settings_manager.saved is False


def test_permissions_command_rejects_unknown_level() -> None:
    """An unknown permission level raises, unpersisted."""
    settings_manager = _FakePermissionsSettingsManager()

    with pytest.raises(ValueError, match="null, ask, auto"):
        PermissionsCommand.run(
            category="read", level="nope", settings_manager=settings_manager
        )

    assert settings_manager.applied is False
    assert settings_manager.saved is False
    assert (
        settings_manager.settings.permissions[PermissionCategory.READ]
        == PermissionLevel.ASK
    )


@pytest.mark.parametrize("keyword", ["null", "NULL", "Null"])
def test_permissions_command_null_removes_the_grant(keyword: str) -> None:
    """The word "null", in any case, removes the category's grant."""
    settings_manager = _FakePermissionsSettingsManager()

    result = PermissionsCommand.run(
        category="read", level=keyword, settings_manager=settings_manager
    )

    assert result == "read permission removed"
    assert (
        settings_manager.settings.permissions[PermissionCategory.READ]
        is PermissionLevel.NONE
    )
    assert settings_manager.applied is True
    assert settings_manager.saved is True


def test_permissions_command_null_is_idempotent_when_already_none() -> None:
    """Removing an already-ungranted category succeeds without raising."""
    settings_manager = _FakePermissionsSettingsManager(
        permissions={
            PermissionCategory.READ: PermissionLevel.NONE,
            PermissionCategory.WRITE: PermissionLevel.ASK,
            PermissionCategory.WEB: PermissionLevel.ASK,
            PermissionCategory.EXECUTE: PermissionLevel.ASK,
        }
    )

    result = PermissionsCommand.run(
        category="read", level="null", settings_manager=settings_manager
    )

    assert result == "read permission removed"
    assert (
        settings_manager.settings.permissions[PermissionCategory.READ]
        is PermissionLevel.NONE
    )


def test_permissions_command_reports_unset_for_an_ungranted_category() -> None:
    """Reading a category at NONE reports that, not an error."""
    settings_manager = _FakePermissionsSettingsManager(
        permissions={
            PermissionCategory.READ: PermissionLevel.NONE,
            PermissionCategory.WRITE: PermissionLevel.ASK,
            PermissionCategory.WEB: PermissionLevel.ASK,
            PermissionCategory.EXECUTE: PermissionLevel.ASK,
        }
    )

    result = PermissionsCommand.run(category="read", settings_manager=settings_manager)

    assert result == "No read permission set"


def test_base_command_cannot_be_instantiated() -> None:
    """BaseCommand is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCommand()
