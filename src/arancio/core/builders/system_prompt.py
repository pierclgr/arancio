"""System prompt builder loading the prompt from a harness markdown file."""

from arancio.core.builders.base import Builder
from arancio.storage.manager import StorageManager


class SystemPromptBuilder(Builder):
    """Builder that assembles the system prompt for an agent.

    ``SYSTEM_PROMPT.md`` is read and parsed once, when the builder is instantiated, and
    cached on the instance for subsequent :meth:`build` calls.
    """

    def __init__(self) -> None:
        """Load and cache the system prompt from the harness file.

        Loads ``SYSTEM_PROMPT.md`` via
        :meth:`~arancio.storage.manager.StorageManager.load_system_prompt`
        (which seeds it with a default when missing and parses it as
        dynamic markdown), mirroring how tool descriptions are loaded once
        in :class:`~arancio.core.tools.base.BaseTool`.
        """
        self._prompt: str = StorageManager.load_system_prompt().content

    def build(self) -> str:
        """Return the cached system prompt.

        Returns:
            The rendered system prompt string.
        """
        return self._prompt
