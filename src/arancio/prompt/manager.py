"""Prompt parsing that directly constructs typed Action data."""

import shlex
from pathlib import Path

from arancio.prompt.actions.types import (
    Action,
    CommandAction,
    PromptAction,
    ShellCommandAction,
)
from arancio.prompt.patterns import (
    COMMAND_PATTERN,
    HIDDEN_SHELL_COMMAND_PATTERN,
    MENTION_PATTERN,
    SHELL_COMMAND_PATTERN,
)


class PromptManager:
    """Resolve input directly to Actions, preserving prefixes and mention rules.

    Shell prefixes keep their entire remainder, commands retain non-POSIX quote
    handling, and ordinary prompts resolve existing file/directory mentions. Every
    Action retains the original raw_input for display and command recording.
    """

    @classmethod
    def resolve(cls, prompt: str) -> Action:
        """Build the Action matching one original input line.

        Args:
            prompt: the original input text.

        Returns:
            A shell, command or model-prompt Action with raw_input preserved.

        Raises:
            ValueError: when a shell command is empty or command quotes are unbalanced.
        """
        for pattern, add_to_history in (
            (HIDDEN_SHELL_COMMAND_PATTERN, False),
            (SHELL_COMMAND_PATTERN, True),
        ):
            shell_match = pattern.match(prompt)
            if shell_match is not None:
                shell_command = shell_match.group(1)
                if not shell_command.strip():
                    raise ValueError("shell command is empty")
                return ShellCommandAction(
                    command=shell_command,
                    add_to_history=add_to_history,
                    raw_input=prompt,
                )

        match = COMMAND_PATTERN.match(prompt)
        if match is None:
            rewritten_prompt, mentions = cls.__resolve_mentions(prompt)
            return PromptAction(
                prompt=rewritten_prompt, raw_input=prompt, mentions=mentions
            )

        command_name = match.group(1)
        command_args = cls.__split_command_args(match.group(2) or "")
        return CommandAction(name=command_name, args=command_args, raw_input=prompt)

    @staticmethod
    def __split_command_args(raw: str) -> list[str]:
        r"""Split a command's argument text into words, honouring quotes.

        Splitting is non-POSIX so a backslash stays literal and a Windows path
        such as ``C:\\Users\\me`` survives intact; a matched pair of surrounding
        quotes is then stripped from each word, so ``"my folder"`` arrives as a
        single ``my folder`` argument. A quote only opens a quoted run when it
        starts a word, leaving an apostrophe inside one (``don't``) literal and
        needing no closing.

        Args:
            raw: the text following the command name, possibly empty.

        Returns:
            The argument words, each with a surrounding matched quote pair
            removed.

        Raises:
            ValueError: when a quote opening a word is never closed, reported
                against the argument text rather than surfacing as a confusing
                failure further down.
        """
        try:
            words = shlex.split(raw, posix=False)
        except ValueError as exc:
            raise ValueError(f"unbalanced quote in command arguments: {raw}") from exc

        return [
            word[1:-1]
            if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'"
            else word
            for word in words
        ]

    @staticmethod
    def __resolve_mentions(prompt: str) -> tuple[str, list[Path]]:
        """Rewrite resolving @mentions to their absolute path and collect the targets.

        Args:
            prompt: the raw user prompt.

        Returns:
            A ``(rewritten_prompt, resolved_paths)`` pair. ``rewritten_prompt`` has
            every mention whose target is an existing file or directory rewritten
            to its resolved absolute path, quoted with ``@"..."`` when that path
            contains a space and as bare ``@...`` otherwise; a mention that does not
            resolve to either is left byte-for-byte unchanged. ``resolved_paths``
            holds the resolved absolute :class:`~pathlib.Path` for each resolving
            mention, in the order its mention first appears in the prompt.
        """
        resolved_paths: list[Path] = []
        rewritten_prompt = prompt

        for match in MENTION_PATTERN.finditer(prompt):
            raw_path = match.group(1) if match.group(1) is not None else match.group(2)
            target = (Path.cwd() / raw_path).resolve()
            if not (target.is_file() or target.is_dir()):
                continue
            resolved_paths.append(target)
            replacement = f'@"{target}"' if " " in str(target) else f"@{target}"
            rewritten_prompt = rewritten_prompt.replace(match.group(0), replacement)

        return rewritten_prompt, resolved_paths
