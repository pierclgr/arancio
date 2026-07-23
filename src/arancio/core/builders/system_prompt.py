"""System prompt builder loading the prompt from a harness markdown file."""

from typing import ClassVar

from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.builders.base import Builder
from arancio.core.constants.path.base import SYSTEM_PROMPT_HARNESS_PATH


class SystemPromptBuilder(Builder):
    """Builder that assembles the system prompt for an agent.

    ``SYSTEM_PROMPT.md`` is read and parsed once, on the first :meth:`build` call, then
    cached for the rest of the process's lifetime.
    """

    _prompt: ClassVar[str | None] = None

    @classmethod
    def build(cls) -> str:
        """Build and return the system prompt.

        Loads ``SYSTEM_PROMPT.md`` from the harness directory and parses
        it as dynamic markdown the first time it is called, mirroring how
        tool descriptions are loaded once in
        :class:`~arancio.core.tools.base.BaseTool`; subsequent calls
        return the cached result.

        Returns:
            The rendered system prompt string.

        Raises:
            FileNotFoundError: when the system prompt harness file does
                not exist.
        """
        if cls._prompt is None:
            if not SYSTEM_PROMPT_HARNESS_PATH.is_file():
                raise FileNotFoundError(
                    f"System prompt file {SYSTEM_PROMPT_HARNESS_PATH} not found."
                )

            prompt_file = DynamicMarkdownFile(SYSTEM_PROMPT_HARNESS_PATH)
            prompt_file.parse()
            cls._prompt = prompt_file.content

        return cls._prompt
