"""Action types produced by the prompt manager."""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path


class Action(ABC):
    """Base class for a resolved decision about how to handle a prompt.

    :class:`arancio.prompt.manager.PromptManager` turns a raw prompt into a
    concrete action, and :class:`arancio.prompt.actions.executor.ActionExecutor`
    carries it out. Actions are plain data: they hold what the executor needs and
    do not execute themselves.
    """


@dataclass
class CommandAction(Action):
    """Run the named slash command with its parsed arguments.

    Attributes:
        name: the command name parsed from the prompt.
        args: the command arguments parsed from the prompt.
        raw_input: the exact text the user submitted before parsing.
    """

    name: str
    args: list[str]
    raw_input: str


@dataclass
class ShellCommandAction(Action):
    """Run user-provided text as a shell command.

    Attributes:
        command: the exact command text following the shell prefix.
        add_to_history: whether to store the generated tool call and result
            in the agent's conversation history.
        raw_input: the exact text the user submitted before parsing.
    """

    command: str
    add_to_history: bool
    raw_input: str


@dataclass
class FileMention:
    """An existing file or directory and optional inclusive source-line bounds.

    Attributes:
        path: the resolved absolute target path.
        start_line: the first requested source line, or None for a whole-file read.
        end_line: the last requested source line, or None for a whole-file read.
        raw_markdown: whether to return raw Markdown source. True only
            for ``.md`` file mentions with a trailing ``:raw`` modifier.
    """

    path: Path
    start_line: int | None = None
    end_line: int | None = None
    raw_markdown: bool = False


@dataclass
class PromptAction(Action):
    """Send the raw prompt text to the model.

    Attributes:
        prompt: the prompt to send to the model, with every resolving
            @mention already rewritten to its resolved absolute path.
        raw_input: the exact text the user submitted before parsing.
        mentions: resolved targets and source-line bounds from @mention tokens,
            in their original order. Empty when the prompt had none.
    """

    prompt: str
    raw_input: str
    mentions: list[FileMention] = field(default_factory=list)
