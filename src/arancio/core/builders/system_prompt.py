"""System prompt builder loading the prompt from a harness markdown file."""

from datetime import date
from pathlib import Path

from arancio.core.builders.base import Builder
from arancio.storage.manager import StorageManager


class SystemPromptBuilder(Builder):
    """Builder that assembles the system prompt for an agent.

    ``SYSTEM_PROMPT.md`` is read and parsed once, when the builder is instantiated, and
    cached on the instance for subsequent :meth:`build` calls; :meth:`reload` refreshes
    that cache from disk. Runtime context that must stay current (the date and the
    working directory) is appended by :meth:`build` instead of being part of the cached
    text, so it survives a session that outlives the day it started on or that moves to
    another directory.
    """

    def __init__(self) -> None:
        """Load and cache the system prompt from the harness file.

        Loads ``SYSTEM_PROMPT.md`` via
        :meth:`~arancio.storage.manager.StorageManager.load_system_prompt`
        (which seeds it with a default when missing and parses it as
        dynamic markdown), mirroring how tool descriptions are loaded once
        in :class:`~arancio.core.tools.base.BaseTool`.
        """
        self._system_prompt: str = StorageManager.load_system_prompt().content

    def reload(self) -> None:
        """Refresh the cached system prompt from disk.

        Goes through
        :meth:`~arancio.storage.manager.StorageManager.load_system_prompt`
        rather than the parsed file's own ``reload``, so a harness file
        deleted mid-session is seeded with the default again instead of
        raising.
        """
        self._system_prompt = StorageManager.load_system_prompt().content

    def build(self) -> str:
        """Return the cached system prompt with the current runtime context appended.

        Both the date and the working directory are resolved on every call,
        so a long-running session reports the new day after midnight and the
        new directory after a move.

        Returns:
            The rendered system prompt string.
        """
        # Path.cwd() tracks the app's working directory because
        # App.set_working_directory mirrors every move to os.chdir
        return (
            f"{self._system_prompt}\n\n"
            f"Today: {date.today().isoformat()}\n"
            f"Current working directory: {Path.cwd()}"
        )
