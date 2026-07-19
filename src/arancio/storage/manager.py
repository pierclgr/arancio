"""On-disk storage management for the arancio working directory."""

import shutil
from pathlib import Path
from typing import Any

import yaml

from arancio.core.constants.path.base import (
    ARANCIO_DEFAULT_DIR,
    ARANCIO_SETTINGS_FILE,
    LITELLM_CONFIG_DIR,
)
from arancio.core.messages import ErrorMessage, Message, WarningMessage
from arancio.settings.settings import Settings


class StorageManager:
    """Manage arancio's persistent on-disk storage.

    Handles generic directory creation and the dedicated arancio working
    directory (default ``~/.arancio``), under which arancio keeps everything it
    persists across runs (login information, harness, future settings, …).

    Attributes:
        _root: the arancio working directory all dedicated paths hang off.
    """

    def __init__(self, root: Path = ARANCIO_DEFAULT_DIR) -> None:
        """Initialize the manager with the working directory root.

        Args:
            root: the arancio working directory. Defaults to
                :data:`ARANCIO_DEFAULT_DIR` (``~/.arancio``); tests pass a
                temporary path for isolation.
        """
        self._root = root

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the manager.

        Returns:
            The manager's class name followed by its working directory root.
        """
        return f"{type(self).__name__}({self._root})"

    @property
    def root(self) -> Path:
        """Return the arancio working directory root.

        Returns:
            The working directory all dedicated paths hang off.
        """
        return self._root

    def make_dir(self, *parts: str) -> Path:
        """Create and return a directory under the working directory root.

        Args:
            *parts: path components appended to the root.

        Returns:
            The created (or already existing) directory.
        """
        path = self._root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def bind_litellm_login_dir(
        self, litellm_config_dir: Path = LITELLM_CONFIG_DIR
    ) -> Path:
        """Redirect LiteLLM's login storage into the working directory.

        LiteLLM hardcodes its config directory (``~/.config/litellm``) and
        offers no override, so everything it writes there (per-provider login
        state, e.g. ChatGPT OAuth tokens) is redirected by symlinking that
        directory onto ``<root>/litellm``. Any pre-existing real directory is
        migrated (its entries moved into the target) before being replaced by
        the symlink, so existing logins are preserved. An already-symlinked
        config directory is left untouched.

        Args:
            litellm_config_dir: the LiteLLM config directory to redirect.
                Defaults to :data:`LITELLM_CONFIG_DIR` (``~/.config/litellm``);
                tests pass a temporary path for isolation.

        Returns:
            The working-directory subdirectory LiteLLM's storage now resolves to.
        """
        target = self.make_dir("litellm")

        if litellm_config_dir.is_symlink():
            return target

        if litellm_config_dir.exists():
            for entry in litellm_config_dir.iterdir():
                shutil.move(str(entry), str(target / entry.name))
            litellm_config_dir.rmdir()

        litellm_config_dir.parent.mkdir(parents=True, exist_ok=True)
        litellm_config_dir.symlink_to(target, target_is_directory=True)
        return target

    @staticmethod
    def save_settings(settings: Settings) -> Path:
        """Write the settings to the settings file.

        Args:
            settings: the settings to persist.

        Returns:
            The path the settings were written to.
        """
        ARANCIO_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARANCIO_SETTINGS_FILE.write_text(
            yaml.safe_dump(settings.to_dict(), sort_keys=False)
        )
        return ARANCIO_SETTINGS_FILE

    @classmethod
    def load_settings(cls) -> tuple[dict[str, Any] | None, list[Message]]:
        """Read the settings file, signaling when it is missing or unusable.

        When the settings file is absent, the registered default settings are
        built and written to disk (so a first run leaves a populated file
        behind), reported as a warning; ``None`` is returned since there is no
        file content to hand off. When the file exists but its content is
        empty or otherwise not a YAML mapping (a syntax error, or e.g. a list
        at the top level), ``None`` is returned without touching the file,
        reported as an error. In both cases, deciding that ``None`` means
        "use the default settings" and building them is the settings
        manager's responsibility, not this method's. Per-field validation of
        an otherwise well-formed dictionary is likewise the settings
        manager's responsibility (delegated to
        :class:`~arancio.settings.validator.SettingsValidator`).

        Returns:
            A ``(data, messages)`` pair: the settings dictionary parsed from
            disk, or ``None`` when the file is missing or unusable, and the
            messages to surface in the UI describing that condition.
        """
        if not ARANCIO_SETTINGS_FILE.exists():
            cls.save_settings(Settings.default())
            return None, [
                WarningMessage(
                    content="settings.yml not found; creating it with default settings."
                )
            ]

        try:
            data = yaml.safe_load(ARANCIO_SETTINGS_FILE.read_text())
        except yaml.YAMLError:
            data = None

        if not isinstance(data, dict):
            return None, [
                ErrorMessage(
                    content=(
                        "settings.yml is empty or not a valid YAML mapping; "
                        "using default settings."
                    )
                )
            ]

        return data, []
