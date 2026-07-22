"""Prompt manager turning a raw prompt into action arguments."""

from pathlib import Path
from typing import Any

from arancio.prompt.patterns import COMMAND_PATTERN, MENTION_PATTERN


class PromptManager:
    """Resolves a raw prompt into the keyword arguments for its action.

    A prompt starting with ``/`` is parsed as ``/<name> <args...>``: the first word is
    the command name and the remaining whitespace-separated words are its arguments,
    producing the ``command_name``/``command_args`` keyword arguments for a
    :class:`arancio.prompt.actions.types.CommandAction`; ``@`` mentions in the
    remainder are left untouched, never mention-resolved.

    Any other prompt is scanned for ``@path``/``@"path with spaces"`` mentions: a
    mention whose path resolves to an existing file or directory (relative paths
    against the process working directory, absolute paths as-is) is rewritten to
    its resolved absolute path; a mention that does not resolve to either is left
    unchanged. This produces the ``prompt`` (rewritten) and ``mentions`` (resolved
    absolute paths, in appearance order) keyword arguments for a
    :class:`arancio.prompt.actions.types.PromptAction`.
    :class:`arancio.prompt.actions.executor.ActionExecutor` builds and runs the
    action from these arguments, reading a file mention with ``ReadFileTool`` and
    listing a directory mention's contents with ``ShellCommandTool``.
    """

    @classmethod
    def resolve_prompt(cls, prompt: str) -> dict[str, Any]:
        """Resolve the prompt into the keyword arguments for its action.

        Args:
            prompt: the raw user prompt.

        Returns:
            The ``command_name``/``command_args`` keyword arguments for a command
            action when the prompt is a slash command, otherwise the ``prompt``
            (with resolving ``@`` mentions rewritten to absolute paths) and
            ``mentions`` (their resolved absolute paths, in order) keyword
            arguments for a prompt action.
        """
        match = COMMAND_PATTERN.match(prompt)
        if match is None:
            rewritten_prompt, mentions = cls._resolve_mentions(prompt)
            return {"prompt": rewritten_prompt, "mentions": mentions}

        command_name = match.group(1)
        command_args = match.group(2).split() if match.group(2) else []
        return {"command_name": command_name, "command_args": command_args}

    @staticmethod
    def _resolve_mentions(prompt: str) -> tuple[str, list[Path]]:
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
