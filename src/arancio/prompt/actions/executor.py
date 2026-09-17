"""Executor running the action built from the prompt manager's arguments."""

from __future__ import annotations

import inspect
import json
import platform
import shlex
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

from arancio.commands.base import BaseCommand
from arancio.commands.registry import COMMAND_REGISTRY
from arancio.core.agents import Agent
from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    UserMessage,
)
from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.prompt.actions.constants import INJECTABLE_COMMAND_PARAMETERS
from arancio.prompt.actions.types import (
    BaseAction,
    CommandAction,
    PromptAction,
    ShellCommandAction,
)
from arancio.sessions.manager import SessionManager
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager

if TYPE_CHECKING:
    from arancio.ui.app import App


class ActionExecutor:
    """Carries out an action built from the prompt manager's arguments.

    A :class:`arancio.prompt.actions.types.ShellCommandAction` runs locally
    through ``ShellCommandTool``. Its explicit prefix bypasses the normal
    tool-permission gate. A :class:`arancio.prompt.actions.types.CommandAction`
    is resolved against
    :data:`arancio.commands.registry.COMMAND_REGISTRY`: its raw prompt words are
    bound to the command's parameter names and run locally. A
    :class:`arancio.prompt.actions.types.PromptAction` is sent to the model
    through the agent. Both produce the same message stream so callers render
    them the same way.
    """

    # shared across every instance so resolving @mentions doesn't reload the
    # harness files on every prompt
    _read_file_tool: ReadFileTool = ReadFileTool()
    _shell_command_tool: ShellCommandTool = ShellCommandTool()

    def __init__(
        self,
        agent: Agent,
        application: App,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
    ) -> None:
        """Store the objects used to carry out actions.

        Args:
            agent: the agent whose run loop handles a prompt action.
            application: the running app a command acts on.
            settings_manager: the settings manager a command can use to apply
                and persist settings changes.
            session_manager: the manager owning the active chat; every write
                goes through its ``session_recorder``, and ``/clear`` receives the
                manager itself.
        """
        self._agent: Agent = agent
        self._application: App = application
        self._settings_manager: SettingsManager = settings_manager
        self._session_manager: SessionManager = session_manager

    def execute(self, action: BaseAction) -> Iterator[Message]:
        """Execute the action, producing its output messages.

        Args:
            action: the action to execute.

        Returns:
            An iterator over the messages the action produces.

        Raises:
            ValueError: when the action is of an unknown type.
        """
        if isinstance(action, ShellCommandAction):
            return self._execute_shell_command(action)
        if isinstance(action, CommandAction):
            return self._execute_command(action)
        elif isinstance(action, PromptAction):
            return self._execute_prompt(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def _execute_shell_command(self, action: ShellCommandAction) -> Iterator[Message]:
        """Run a user-provided command through ``ShellCommandTool``.

        The user's explicit shell prefix is sufficient consent, so this path
        bypasses ``PermissionManager`` and does not require a configured model.
        Both forms yield the same paired messages; only ``!`` appends them to
        agent history, preceded by a user-attribution message.

        Args:
            action: the shell command action to execute.

        Yields:
            A synthetic tool call followed by its paired result or error.
        """
        call_id = f"shell_{uuid.uuid4().hex}"
        arguments = {"command": action.command}
        call = ToolCallMessage(
            content=f"ShellCommandTool({json.dumps(arguments)})",
            id=call_id,
            name="ShellCommandTool",
            arguments=arguments,
            in_history=action.add_to_history,
        )
        command_error = self._session_manager.session_recorder.command(
            raw_input=action.raw_input,
            name="shell",
            args=[action.command],
        )
        attribution_error = None
        if action.add_to_history:
            attribution = UserMessage(
                content="User explicitly ran the following command:"
            )
            self._agent.add_message_to_history(attribution)
            self._agent.add_message_to_history(call)
            # the user never saw this line: it exists so the model does not
            # read the synthetic call as its own decision
            attribution_error = self._session_manager.session_recorder.message(
                attribution, visible=False
            )
        call_error = self._session_manager.session_recorder.message(call)
        yield call
        yield from self._yield_notices(command_error, attribution_error, call_error)

        result = self._shell_command_tool.call(call_id=call_id, **arguments)
        result.in_history = action.add_to_history
        if action.add_to_history:
            self._agent.add_message_to_history(result)
        result_error = self._session_manager.session_recorder.message(result)
        yield result
        yield from self._yield_notices(result_error)

    @staticmethod
    def _yield_notices(*notices: ErrorMessage | None) -> Iterator[Message]:
        """Yield each persistence notice a recorder write actually produced.

        Args:
            notices: the results of recorder writes, ``None`` when the write
                landed on disk.

        Yields:
            Every notice that is not ``None``, in the order given.
        """
        for notice in notices:
            if notice:
                yield notice

    def _yield_error(self, content: str) -> Iterator[Message]:
        """Record an error the executor itself reports, then yield it.

        Args:
            content: the error text shown to the user and saved in the session.

        Yields:
            The error, followed by the persistence notice when saving it failed.
        """
        error = ErrorMessage(content=content)
        save_error = self._session_manager.session_recorder.message(error)
        yield error
        yield from self._yield_notices(save_error)

    def _execute_command(self, action: CommandAction) -> Iterator[Message]:
        """Bind the action's words to the command's parameters and run it.

        A plain-string result is wrapped into an :class:`AssistantMessage` so
        the executor always yields messages, regardless of the action type.

        Args:
            action: the command action to run.

        Yields:
            The command's result message, or an :class:`ErrorMessage` when the
            command is unknown or its arguments do not match its signature.
        """
        command = COMMAND_REGISTRY.get(action.name)
        if command is None:
            yield from self._yield_error(f"Command not found: {action.name}")
            return

        try:
            kwargs = self._build_command_kwargs(command, action.args)
        except Exception as exc:
            yield from self._yield_error(
                f"Error while executing command {action.name}: {exc}"
            )
            return

        command_error = (
            self._session_manager.session_recorder.command(
                raw_input=action.raw_input,
                name=action.name,
                args=action.args,
            )
            if action.name == "clear"
            else None
        )
        try:
            result = command.run(**kwargs)
        except Exception as exc:
            yield from self._yield_error(
                f"Error while executing command {action.name}: {exc}"
            )
            yield from self._yield_notices(command_error)
            return

        if command_error is None:
            command_error = self._session_manager.session_recorder.command(
                raw_input=action.raw_input,
                name=action.name,
                args=action.args,
            )

        state_error = self._session_effect_of(action)

        if result is None:
            yield from self._yield_notices(command_error, state_error)
            return
        message = (
            AssistantMessage(content=result) if isinstance(result, str) else result
        )
        if isinstance(message, AssistantMessage):
            message.in_history = False
            result_error = self._session_manager.session_recorder.message(message)
        else:
            result_error = None
        yield message
        yield from self._yield_notices(command_error, state_error, result_error)

    def _build_command_kwargs(
        self, command: Type[BaseCommand], args: list[str]
    ) -> dict[str, Any]:
        """Match the prompt words to the command's parameters and inject dependencies.

        Each word is matched, in order, to the next prompt parameter the command
        declares (excluding the injected ones below); a word beyond the number of
        declared parameters is dropped rather than rejected.

        Args:
            command: the command whose ``execute`` parameters to match against.
            args: the raw prompt words.

        Returns:
            The keyword arguments for :meth:`command.run`: the prompt words keyed
            by their matching parameter name, plus the running application, the
            settings manager and/or the agent for any the command declares a
            parameter for.
        """
        # each injectable parameter name maps to a same-named private attribute
        # on the executor (``application`` -> ``self._application``, etc.)
        injectable = {
            name: getattr(self, f"_{name}") for name in INJECTABLE_COMMAND_PARAMETERS
        }
        signature = inspect.signature(command.execute)
        prompt_parameter_names = [
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name not in INJECTABLE_COMMAND_PARAMETERS
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
        kwargs = dict(zip(prompt_parameter_names, args))
        for name, value in injectable.items():
            if name in signature.parameters:
                kwargs[name] = value
        return kwargs

    def _execute_prompt(self, action: PromptAction) -> Iterator[Message]:
        """Send the action's text to the model through the agent.

        Args:
            action: the prompt action to send.

        Yields:
            A synthesized ReadFileTool (file mention) or ShellCommandTool
            (directory mention, listed with ``ls -la``) call/result pair
            per resolved @mention, in prompt order, then the agent's
            message stream for the prompt, or an :class:`ErrorMessage` when
            no provider or model name is configured yet.
        """
        try:
            self._settings_manager.settings.model_id
        except ValueError as exc:
            yield from self._yield_error(f"Error sending message: {exc}")
            return
        prelude = self._resolve_mentions(action.mentions)
        message = UserMessage(
            content=action.prompt,
            display_text=action.raw_input,
        )
        # the agent appends this to model history but never yields it back, so
        # the caller that built it is the one that records it
        save_error = self._session_manager.session_recorder.message(message)
        yield from self._yield_notices(save_error)
        yield from self._session_manager.session_recorder.record_stream(
            self._agent.run(message, prelude=prelude)
        )

    def _session_effect_of(self, action: CommandAction) -> ErrorMessage | None:
        """Persist whatever session state a successful local command changed.

        Only the executor knows which command changes what, so the mapping lives
        here; the write itself is a direct recorder call.

        Args:
            action: the successfully executed local command.

        Returns:
            A persistence error notice, or ``None`` when no session state
            changed or the write succeeded.
        """
        if action.name == "cd":
            return self._session_manager.session_recorder.working_directory(
                self._application.working_directory
            )
        changes_configuration = action.name in {"provider", "model", "effort"}
        changes_permission = action.name == "permissions" and len(action.args) > 1
        if not (changes_configuration or changes_permission):
            return None
        return self._session_manager.session_recorder.configuration(
            SessionConfiguration.from_settings(self._settings_manager.settings)
        )

    @classmethod
    def _resolve_mentions(cls, mentions: list[Path]) -> list[Message]:
        """Synthesize a tool call/result pair for each resolved @mention.

        Bypasses :class:`arancio.core.permissions.manager.PermissionManager`
        entirely: the user's own explicit mention is itself sufficient
        consent, regardless of the granted READ permission level. A file
        target is read via a real :class:`ReadFileTool` instance, reusing
        its disk read (or, for a ``.md`` target, its dynamic-markdown
        expansion), line-numbering, truncation and session bookkeeping
        unchanged. A directory target is listed via a real
        :class:`ShellCommandTool` instance running ``ls -la`` (or the
        PowerShell equivalent on Windows) against the directory.

        Args:
            mentions: resolved absolute paths for the prompt's @mentions, in
                order.

        Returns:
            A flat list alternating a ``ToolCallMessage`` named
            ``"ReadFileTool"`` or ``"ShellCommandTool"`` and its paired
            ``ToolResultMessage``/``ToolErrorMessage``, one pair per
            mention, in the same order as ``mentions``.
        """
        if not mentions:
            return []

        messages: list[Message] = []
        for target in mentions:
            call_id = f"mention_{uuid.uuid4().hex}"
            if target.is_dir():
                name = "ShellCommandTool"
                if platform.system() == "Windows":
                    command = f'Get-ChildItem -Force -LiteralPath "{target}"'
                else:
                    command = f"ls -la -- {shlex.quote(str(target))}"
                arguments = {"command": command}
                result = cls._shell_command_tool.call(call_id=call_id, **arguments)
            else:
                name = "ReadFileTool"
                arguments = {"file_path": str(target)}
                result = cls._read_file_tool.call(call_id=call_id, **arguments)
            messages.append(
                ToolCallMessage(
                    content=f"{name}({json.dumps(arguments)})",
                    id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )
            messages.append(result)
        return messages
