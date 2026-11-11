"""Typed-input routing, explicit tools and the answer-carrying Core driver."""

from __future__ import annotations

import json
import platform
import shlex
import uuid
from collections.abc import Iterator
from inspect import GEN_RUNNING, getgeneratorstate
from threading import get_ident
from typing import TYPE_CHECKING

from arancio.application.prompt.actions.types import (
    CommandAction,
    FileMention,
    PromptAction,
    ShellCommandAction,
)
from arancio.application.prompt.manager import PromptManager
from arancio.commands.base import Command
from arancio.commands.fork import ForkCommand
from arancio.commands.registry import COMMAND_REGISTRY, SESSION_DISCARDING_COMMANDS
from arancio.core.agents.base import Agent
from arancio.core.events import (
    Decision,
    Execution,
    PermissionRequest,
    PermissionResponse,
)
from arancio.core.messages.assistant import AssistantMessage
from arancio.core.messages.base import ErrorMessage, Message
from arancio.core.messages.tool import ToolCallMessage
from arancio.core.messages.user import UserMessage
from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.session import ToolSession
from arancio.sessions.manager import SessionManager
from arancio.settings.manager import SettingsManager
from arancio.ui.controllers.base import UIController

if TYPE_CHECKING:
    from arancio.application.base import Application


class ActionExecutor:
    """Parse input, run adapters and drive Core without a concrete frontend.

    Attributes:
        __application: the facade supplied to commands.
        __agent: the Core execution and history owner.
        __settings_manager: the active configuration used for model checks.
        __session_manager: the public persistence facade.
        __tool_session: the read guard supplying recording snapshots.
        __frontend_controller: the abstract permission-question interface.
        __read_file_tool: the injected direct tool for file mentions.
        __shell_command_tool: the injected direct tool for shell/directory input.
        __execution: the currently driven Core generator.
        __owner_thread_id: the thread driving that generator.
        __close_requested: whether shutdown has requested an end to execution.
    """

    def __init__(
        self,
        *,
        application: Application,
        agent: Agent,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
        tool_session: ToolSession,
        frontend_controller: UIController,
        read_file_tool: ReadFileTool,
        shell_command_tool: ShellCommandTool,
    ) -> None:
        """Store the shared graph and direct tools without constructing UI objects.

        Args:
            application: the facade passed to commands.
            agent: the Core generator and history owner.
            settings_manager: the active model configuration.
            session_manager: the recording facade.
            tool_session: the file-read guard shared with all tools.
            frontend_controller: the abstract frontend question contract.
            read_file_tool: the tool used for explicit file mentions.
            shell_command_tool: the tool used for shell input and directories.
        """
        self.__application: Application = application
        self.__agent: Agent = agent
        self.__settings_manager: SettingsManager = settings_manager
        self.__session_manager: SessionManager = session_manager
        self.__tool_session: ToolSession = tool_session
        self.__frontend_controller: UIController = frontend_controller
        self.__read_file_tool: ReadFileTool = read_file_tool
        self.__shell_command_tool: ShellCommandTool = shell_command_tool
        self.__execution: Execution | None = None
        self.__owner_thread_id: int | None = None
        self.__close_requested: bool = False

    @property
    def is_running(self) -> bool:
        """Whether this executor owns an active Core execution."""
        return self.__execution is not None

    def close(self) -> None:
        """Request closure and close suspended Core only on its owning thread.

        A different thread, or a callback during running Core code, only sets the
        request flag. The driver closes Core when control next returns to it.
        """
        self.__close_requested = True
        if (
            self.__execution is not None
            and self.__owner_thread_id == get_ident()
            and getgeneratorstate(self.__execution) != GEN_RUNNING
        ):
            self.__execution.close()

    def execute(self, prompt: str) -> Iterator[Message]:
        """Parse and route one input line on the consuming thread.

        Args:
            prompt: the original input text.

        Yields:
            Display messages and temporary persistence notices in order.

        Raises:
            ValueError: when the parser returns an unsupported Action type.
        """
        if self.__close_requested:
            return
        try:
            action = PromptManager.resolve(prompt)
        except ValueError as exc:
            yield from self.__yield_error(f"Invalid prompt: {exc}")
            return
        if isinstance(action, ShellCommandAction):
            yield from self.__execute_shell_command(action)
        elif isinstance(action, CommandAction):
            yield from self.__execute_command(action)
        elif isinstance(action, PromptAction):
            yield from self.__execute_prompt(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def __writes_nothing(self, command: type[Command] | None) -> bool:
        """Keep session-discarding commands from creating an unused chat log.

        Args:
            command: the resolved command class, or None for an unknown command.

        Returns:
            Whether writes for this command must be skipped in the current chat.
        """
        return (
            command in SESSION_DISCARDING_COMMANDS
            and not self.__session_manager.current_state().created_on_disk
        )

    def __execute_shell_command(self, action: ShellCommandAction) -> Iterator[Message]:
        """Run explicit shell input outside the normal permission gate.

        Args:
            action: the parsed shell invocation and its model-history policy.

        Yields:
            The call, hook messages, result and save notices in existing order.
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
        try:
            command_error = self.__session_manager.record_command(
                raw_input=action.raw_input, name="shell", args=[action.command]
            )
        except OSError as exc:
            command_error = ErrorMessage(content=f"Could not save session: {exc}")
        attribution_error = None
        if action.add_to_history:
            attribution = UserMessage(
                content="User explicitly ran the following command:"
            )
            self.__agent.add_message_to_history(attribution)
            self.__agent.add_message_to_history(call)
            attribution_error = self.__record_message(attribution, visible=False)
        call_error = self.__record_message(call)
        yield call
        yield from self.__yield_notices(command_error, attribution_error, call_error)

        result, hook_messages = self.__shell_command_tool.call(call)
        for message in hook_messages:
            yield message
            yield from self.__yield_notices(self.__record_message(message))
        result.in_history = action.add_to_history
        if action.add_to_history:
            self.__agent.add_message_to_history(result)
        result_error = self.__record_message(result)
        yield result
        yield from self.__yield_notices(result_error)

    def __execute_command(self, action: CommandAction) -> Iterator[Message]:
        """Run a command directly with Application as its only injected service.

        Strings are saved once. Returned Messages and the existing fork tuple
        already describe recorded output or save notices and are only displayed.

        Args:
            action: the parsed command invocation.

        Yields:
            Command output and temporary notices in existing attribution order.
        """
        command = COMMAND_REGISTRY.get(action.name)
        if command is None:
            yield from self.__yield_error(f"Command not found: {action.name}")
            return
        discards_session = command in SESSION_DISCARDING_COMMANDS
        command_error = (
            self.__record_command(action, command) if discards_session else None
        )
        try:
            result = command.run(action.args, application=self.__application)
        except Exception as exc:
            if command is ForkCommand and isinstance(exc, OSError):
                # the checksum write failed; never record through that same path
                yield ErrorMessage(content=f"Could not save session: {exc}")
            else:
                yield from self.__yield_error(
                    f"Error while executing command {action.name}: {exc}", command
                )
            yield from self.__yield_notices(command_error)
            return

        if not discards_session:
            command_error = self.__record_command(action, command)
        if result is None:
            yield from self.__yield_notices(command_error)
            return
        if isinstance(result, tuple):
            _, confirmation, notices = result
            yield confirmation
            yield from self.__yield_notices(command_error, *notices)
            return
        if isinstance(result, Message):
            yield result
            yield from self.__yield_notices(command_error)
            return

        message = AssistantMessage(content=result, in_history=False)
        result_error = (
            None if self.__writes_nothing(command) else self.__record_message(message)
        )
        yield message
        yield from self.__yield_notices(command_error, result_error)

    def __execute_prompt(self, action: PromptAction) -> Iterator[Message]:
        """Record the user prompt and drive Core with its mention prelude.

        Args:
            action: the rewritten model prompt and its original input.

        Yields:
            Model/prelude messages and save notices, or an unconfigured-model error.
        """
        try:
            self.__settings_manager.settings.model.check_configured()
        except ValueError as exc:
            yield from self.__yield_error(f"Error sending message: {exc}")
            return
        prelude = self.__resolve_mentions(action.mentions)
        message = UserMessage(content=action.prompt, display_text=action.raw_input)
        yield from self.__yield_notices(self.__record_message(message))
        yield from self.__handle_agent(self.__agent(message, prelude=prelude))

    def __handle_agent(self, execution: Execution) -> Iterator[Message]:
        """Advance Core with matching answers and close it on the consuming thread.

        Args:
            execution: the Core generator created for this prompt.

        Yields:
            Each Message followed by its optional save notice.

        Raises:
            RuntimeError: when another Core execution is already active.
        """
        if self.__execution is not None:
            execution.close()
            raise RuntimeError("Core execution is active.")
        self.__execution = execution
        self.__owner_thread_id = get_ident()
        answer = None
        try:
            while not self.__close_requested:
                try:
                    event = self.__execution.send(answer)
                except StopIteration:
                    return
                answer = None
                if self.__close_requested:
                    return
                if isinstance(event, PermissionRequest):
                    answer = self.__ask_permission(event)
                else:
                    yield event
                    yield from self.__yield_notices(self.__record_message(event))
        finally:
            try:
                self.__execution.close()
            finally:
                self.__execution = None
                self.__owner_thread_id = None

    def __ask_permission(self, request: PermissionRequest) -> PermissionResponse:
        """Map the frontend answer using the current question and decision rules.

        Args:
            request: the Core request whose identity the response must match.

        Returns:
            ALLOW for yes, DENY for no/None, or DENY with the free-text note.
        """
        answer = self.__frontend_controller.ask(
            f"Allow {request.call.name} with arguments {request.call.arguments}?",
            ["yes", "no"],
        )
        return PermissionResponse(
            request_id=request.id,
            decision=Decision.ALLOW if answer == "yes" else Decision.DENY,
            message=answer if answer not in ("yes", "no", None) else None,
        )

    def __record_message(
        self, message: Message, visible: bool = True
    ) -> ErrorMessage | None:
        """Record through Sessions with a fresh read-state snapshot.

        Args:
            message: the message to record.
            visible: whether it belongs in visible replay.

        Returns:
            A log/checksum-save notice, or None when saved or skipped.
        """
        try:
            return self.__session_manager.record_message(
                message, file_states=self.__tool_session.snapshot(), visible=visible
            )
        except OSError as exc:
            return ErrorMessage(content=f"Could not save session: {exc}")

    def __record_command(
        self, action: CommandAction, command: type[Command]
    ) -> ErrorMessage | None:
        """Record an invocation in the correct source session when it has a log.

        Args:
            action: the parsed command line.
            command: the resolved class determining discard policy.

        Returns:
            A temporary save notice, or None when saved or deliberately skipped.
        """
        if self.__writes_nothing(command):
            return None
        try:
            return self.__session_manager.record_command(
                raw_input=action.raw_input, name=action.name, args=action.args
            )
        except OSError as exc:
            return ErrorMessage(content=f"Could not save session: {exc}")

    def __yield_error(
        self, content: str, command: type[Command] | None = None
    ) -> Iterator[Message]:
        """Report an ordinary error and retain its existing recording policy.

        Args:
            content: the current user-facing error text.
            command: the resolved command, when it determines discard policy.

        Yields:
            The error followed by a save notice when recording it failed.
        """
        error = ErrorMessage(content=content)
        save_error = (
            None if self.__writes_nothing(command) else self.__record_message(error)
        )
        yield error
        yield from self.__yield_notices(save_error)

    @staticmethod
    def __yield_notices(*notices: ErrorMessage | None) -> Iterator[Message]:
        """Display save notices without recording them or answering Core with them.

        Args:
            notices: save results, with None representing success or a skipped write.

        Yields:
            Each notice in the supplied order.
        """
        for notice in notices:
            if notice is not None:
                yield notice

    def __resolve_mentions(self, mentions: list[FileMention]) -> list[Message]:
        """Synthesize a tool call/result pair for each resolved @mention.

        Bypasses :class:`arancio.core.permissions.manager.PermissionManager`
        entirely: the user's own explicit mention is itself sufficient
        consent, regardless of the granted READ permission level. A file
        target is read via a real :class:`ReadFileTool` instance, reusing
        its disk read (or, for a ``.md`` target without ``:raw``, its dynamic-markdown
        expansion), line-numbering, truncation and session bookkeeping
        unchanged. A directory target is listed via a real
        :class:`ShellCommandTool` instance running ``ls -la`` (or the
        PowerShell equivalent on Windows) against the directory.

        Args:
            mentions: resolved paths and inclusive source bounds, in prompt order.

        Returns:
            A flat list containing a ``ToolCallMessage`` named
            ``"ReadFileTool"`` or ``"ShellCommandTool"`` and its paired
            ``ToolResultMessage``/``ToolErrorMessage``, one pair per
            mention, in the same order as ``mentions``. Hook messages appear
            between each call and its result.
        """
        if not mentions:
            return []

        messages: list[Message] = []
        for mention in mentions:
            target = mention.path
            call_id = f"mention_{uuid.uuid4().hex}"
            if target.is_dir():
                name = "ShellCommandTool"
                if platform.system() == "Windows":
                    command = f'Get-ChildItem -Force -LiteralPath "{target}"'
                else:
                    command = f"ls -la -- {shlex.quote(str(target))}"
                arguments = {"command": command}
                tool = self.__shell_command_tool
            else:
                name = "ReadFileTool"
                arguments = {"file_path": str(target)}
                if mention.raw_markdown:
                    arguments["raw_markdown"] = True
                start_line, end_line = mention.start_line, mention.end_line
                if (
                    start_line is not None
                    and end_line is not None
                    and 1 <= start_line <= end_line
                ):
                    arguments["offset"] = start_line
                    arguments["limit"] = end_line - start_line + 1
                tool = self.__read_file_tool
            call = ToolCallMessage(
                content=f"{name}({json.dumps(arguments)})",
                id=call_id,
                name=name,
                arguments=arguments,
            )
            result, hook_messages = tool.call(call)
            messages.append(call)
            messages.extend(hook_messages)
            messages.append(result)
        return messages
