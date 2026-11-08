"""Prompt parsing that directly constructs typed Action data."""

import re
import shlex
from pathlib import Path

from arancio.prompt.actions.types import (
    Action,
    CommandAction,
    FileMention,
    PromptAction,
    ShellCommandAction,
)
from arancio.prompt.patterns import (
    COMMAND_PATTERN,
    HIDDEN_SHELL_COMMAND_PATTERN,
    LINE_RANGE_PATTERN,
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

    @classmethod
    def __resolve_mentions(cls, prompt: str) -> tuple[str, list[FileMention]]:
        """Rewrite existing mention targets and parse their source-line selectors.

        Args:
            prompt: the raw user prompt.

        Returns:
            The rewritten prompt and mentions in occurrence order. Existing literal
            paths take precedence over selectors; otherwise the longest existing
            path prefix is used. Missing targets stay unchanged. Malformed selectors
            and directory selectors carry no line bounds.
        """
        mentions: list[FileMention] = []
        rewritten_prompt: str = MENTION_PATTERN.sub(
            lambda match: cls.__replace_mention(match, mentions), prompt
        )
        return rewritten_prompt, mentions

    @staticmethod
    def __replace_mention(match: re.Match[str], mentions: list[FileMention]) -> str:
        """Resolve one token and append its target when it exists.

        Args:
            match: the complete mention token and its optional quoted suffix.
            mentions: resolved mentions, appended in prompt order.

        Returns:
            The canonical mention, or the original token if unresolved.
        """
        quoted: str | None = match.group("quoted")
        raw_path: str = (
            quoted + (match.group("suffix") or "")
            if quoted is not None
            else match.group("bare")
        )
        minimum_boundary: int = max(1, len(Path(raw_path).drive))
        boundaries: list[int] = [len(raw_path)]
        boundaries.extend(
            index
            for index in range(len(raw_path) - 1, minimum_boundary - 1, -1)
            if raw_path[index] == ":"
        )
        for boundary in boundaries:
            try:
                target: Path = (Path.cwd() / raw_path[:boundary]).resolve()
                if not (target.is_file() or target.is_dir()):
                    continue
            except (OSError, ValueError):
                # invalid suffixes can exceed filesystem filename limits
                continue

            mention: FileMention = FileMention(path=target)
            suffix: str = raw_path[boundary:]
            selector: re.Match[str] | None = LINE_RANGE_PATTERN.fullmatch(suffix[1:])
            if target.is_file() and selector is not None:
                try:
                    start: int = int(selector.group(1))
                    end: int = int(selector.group(2) or selector.group(1))
                except ValueError:
                    # excessively large integer strings are invalid selectors
                    pass
                else:
                    mention.start_line = start
                    mention.end_line = end
            mentions.append(mention)
            replacement: str = f'@"{target}"' if " " in str(target) else f"@{target}"
            return replacement + suffix
        return match.group(0)
