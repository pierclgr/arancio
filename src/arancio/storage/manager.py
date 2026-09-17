"""On-disk storage management for the arancio working directory."""

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from arancio.core.constants.path import (
    ARANCIO_DEFAULT_DIR,
)
from arancio.core.messages import ErrorMessage, Message, WarningMessage
from arancio.settings.constants import ARANCIO_SETTINGS_FILE
from arancio.settings.settings import Settings
from arancio.storage.constants import ARANCIO_LITELLM_DIR, LITELLM_CONFIG_DIR


class StorageManager:
    """Manage arancio's persistent on-disk storage.

    Handles generic directory creation and file operations for arancio's
    dedicated paths, which are defined by path constants.

    Attributes:
        _root: the arancio working directory all dedicated paths hang off.
    """

    def __init__(self, root: Path = ARANCIO_DEFAULT_DIR) -> None:
        """Initialize the manager with the working directory root.

        Args:
            root: the base directory used by generic directory creation.
                Defaults to :data:`ARANCIO_DEFAULT_DIR` (``~/.arancio``).
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
        """Return the base directory used by generic directory creation.

        Returns:
            The configured base directory.
        """
        return self._root

    def make_dir(self, path: str | Path) -> Path:
        """Create and return a directory at a relative or full path.

        Args:
            path: the directory path. Relative paths are resolved under the
                configured root.

        Returns:
            The created (or already existing) directory.
        """
        directory = Path(path)
        if not directory.is_absolute():
            directory = self._root / directory
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def find_files(self, directory: Path, pattern: str) -> list[Path]:
        """Return files below a directory matching a glob pattern.

        Args:
            directory: the directory tree to search.
            pattern: the filename glob pattern to match.

        Returns:
            Matching paths in deterministic path order.
        """
        return sorted(directory.rglob(pattern))

    @staticmethod
    def read_lines(path: Path) -> list[str]:
        """Read a UTF-8 text file as raw lines.

        Args:
            path: the text file to read.

        Returns:
            The file's lines without their terminating newlines.
        """
        return path.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def file_size(path: Path) -> int:
        """Return the current byte length of a file.

        Args:
            path: the file to inspect.

        Returns:
            The file's current size in bytes, or zero when it does not exist.
        """
        return path.stat().st_size if path.exists() else 0

    @staticmethod
    def append_line(path: Path, line: str) -> None:
        """Append and sync one UTF-8 text line.

        Args:
            path: the text file to append.
            line: text without a trailing newline.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as file:
            file.write(line.encode("utf-8") + b"\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def create_file(path: Path, line: str | None = None) -> None:
        """Create and sync a new UTF-8 text file, optionally with its first line.

        Args:
            path: the new file path, which must not already exist.
            line: optional text without a trailing newline.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            if line is not None:
                file.write(line.encode("utf-8") + b"\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def truncate_file(path: Path, offset: int) -> None:
        """Discard every byte in a file after ``offset``.

        Args:
            path: the file to truncate.
            offset: the retained byte length from the beginning of the file.
        """
        with path.open("r+b") as file:
            file.truncate(offset)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def remove_file(path: Path) -> None:
        """Remove a file when it exists.

        Args:
            path: the file to remove.
        """
        if path.exists():
            path.unlink()

    @staticmethod
    def write_file(path: Path, content: str) -> None:
        """Create parent directories and replace a UTF-8 text file's content.

        Args:
            path: the text file to create or replace.
            content: the complete UTF-8 text content to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def save_settings(settings: Settings) -> None:
        """Serialize and save settings to the global settings file.

        Args:
            settings: the settings snapshot to persist.
        """
        StorageManager.write_file(
            ARANCIO_SETTINGS_FILE,
            yaml.safe_dump(settings.to_dict(), sort_keys=False),
        )

    def bind_litellm_login_dir(
        self, litellm_config_dir: Path = LITELLM_CONFIG_DIR
    ) -> Path:
        """Redirect LiteLLM's login storage into the working directory.

        LiteLLM hardcodes its config directory (``~/.config/litellm``) and
        offers no override, so everything it writes there (per-provider login
        state, e.g. ChatGPT OAuth tokens) is redirected by symlinking that
        directory onto :data:`ARANCIO_LITELLM_DIR`. Any pre-existing real directory is
        migrated (its entries moved into the target) before being replaced by
        the symlink, so existing logins are preserved. An already-symlinked
        config directory is left untouched.

        Args:
            litellm_config_dir: the LiteLLM config directory to redirect.
                Defaults to :data:`LITELLM_CONFIG_DIR` (``~/.config/litellm``);
                tests pass a temporary path for isolation.

        Returns:
            The directory LiteLLM's storage now resolves to.
        """
        target = self.make_dir(ARANCIO_LITELLM_DIR)

        if litellm_config_dir.is_symlink():
            return target

        if litellm_config_dir.exists():
            for entry in litellm_config_dir.iterdir():
                shutil.move(str(entry), str(target / entry.name))
            litellm_config_dir.rmdir()

        litellm_config_dir.parent.mkdir(parents=True, exist_ok=True)
        litellm_config_dir.symlink_to(target, target_is_directory=True)
        return target

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
