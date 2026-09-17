"""System prompt builder loading the prompt from a harness markdown file."""

from datetime import date
from pathlib import Path

from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.builders.base import Builder
from arancio.core.constants.agent import AGENT_DEFAULT_SYSTEM_PROMPT
from arancio.core.constants.path import SYSTEM_PROMPT_HARNESS_PATH


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

        Seeds ``SYSTEM_PROMPT.md`` with the default when missing, then parses it as
        dynamic markdown. This mirrors how tool descriptions are loaded once in
        :class:`~arancio.core.tools.base.BaseTool`.
        """
        self._system_prompt = self._load_system_prompt()

    def reload(self) -> None:
        """Refresh the cached system prompt from disk.

        A harness file deleted mid-session is seeded with the default again instead of
        raising.
        """
        self._system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        """Seed and parse the system-prompt harness file.

        Returns:
            The expanded system prompt content.
        """
        if not SYSTEM_PROMPT_HARNESS_PATH.is_file():
            SYSTEM_PROMPT_HARNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SYSTEM_PROMPT_HARNESS_PATH.write_text(
                AGENT_DEFAULT_SYSTEM_PROMPT, encoding="utf-8"
            )

        return DynamicMarkdownFile(SYSTEM_PROMPT_HARNESS_PATH).content

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
