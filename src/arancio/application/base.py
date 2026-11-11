"""Application operations coordinating Core, Settings and Sessions."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from arancio.application.prompt.actions.executor import ActionExecutor
from arancio.application.status import ApplicationStatus
from arancio.core.agents.base import Agent
from arancio.core.clients.base import Client
from arancio.core.clients.constants import LITELLM_PROVIDER_NAMES
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.clients.types import Reasoning
from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager
from arancio.core.messages.assistant import AssistantMessage
from arancio.core.messages.base import ErrorMessage, Message
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.plugins.constants import PLUGIN_PERMISSION_CATEGORY_PREFIX
from arancio.core.plugins.manager import PluginManager
from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.manager import ToolManager
from arancio.core.tools.session import ToolSession
from arancio.sessions.configurations import SessionConfiguration
from arancio.sessions.manager import SessionManager
from arancio.sessions.state import SessionState
from arancio.settings.base import Settings
from arancio.settings.constants import (
    MODEL_NAME_FIELD,
    PERMISSIONS_FIELD,
    PROVIDER_FIELD,
    REASONING_EFFORT_FIELD,
)
from arancio.settings.manager import SettingsManager
from arancio.storage.manager import StorageManager
from arancio.ui.controllers.base import UIController


class Application:
    """Coordinate backend operations while exposing only data and Messages.

    A future frontend supplies UIController. This layer owns no widgets,
    frontend threads or event loop. Only one caller drives submit at a time.

    Attributes:
        __agent: the Core execution and history owner.
        __client: the main model client.
        __summary_client: the selected model with reasoning disabled.
        __tool_session: the shared file-read guard.
        __hook_manager: the lifecycle hook dispatcher.
        __settings_manager: the active/global settings owner.
        __session_manager: the conversation persistence facade.
        __frontend_controller: the abstract frontend contract.
        __action_executor: the input router and Core generator driver.
        __startup_messages: notices awaiting start.
        __working_directory: the current runtime directory.
        __started: whether STARTUP was dispatched.
        __closed: whether shutdown was requested.
        __setting_fields: the dotted paths of the editable fields, in dependency
            order.
        __setting_messages: confirmation formatters for checked changes, keyed by
            dotted path.
    """

    __setting_fields: ClassVar[tuple[str, ...]] = (
        PROVIDER_FIELD,
        MODEL_NAME_FIELD,
        REASONING_EFFORT_FIELD,
        PERMISSIONS_FIELD,
    )
    __setting_messages: ClassVar[dict[str, Callable[[Settings, Any], str]]] = {
        PROVIDER_FIELD: lambda settings, value: (
            f"Provider set to {settings.model.provider}"
        ),
        MODEL_NAME_FIELD: lambda settings, value: f"Model set to {settings.model.name}",
        REASONING_EFFORT_FIELD: lambda settings, value: (
            f"Reasoning effort set to {value if value is not None else 'null'}"
        ),
        PERMISSIONS_FIELD: lambda settings, value: (
            Application.__format_permission_changes(settings, value)
        ),
    }

    @staticmethod
    def build(controller: UIController) -> Application:
        """Build and wire the backend using the supplied frontend contract.

        This loads settings, then the plugins they enable, and creates the
        backend. It does not create a frontend or dispatch lifecycle hooks.

        Args:
            controller: the frontend contract implementation supplied by the caller.

        Returns:
            The configured Application and its shared backend dependencies.
        """
        storage_manager = StorageManager()

        # build the shared core services
        hook_manager = HookManager()
        tool_session = ToolSession()
        client = LiteLLMClient(on_login=controller.show_login, stream=True)
        summary_client = LiteLLMClient(on_login=controller.show_login, stream=False)
        summary_client.reasoning = Reasoning()
        tool_manager = ToolManager(
            web_summary_client=summary_client,
            hook_manager=hook_manager,
            tool_session=tool_session,
        )

        # settings decide which plugins load; plugins create their categories
        # before permissions are granted
        settings_manager = SettingsManager(storage_manager, LITELLM_PROVIDER_NAMES)
        settings, settings_messages = settings_manager.load()
        plugin_manager = PluginManager(hook_manager, tool_manager)
        plugin_messages = plugin_manager.load(
            {name: plugin.enabled for name, plugin in settings.plugins.items()}
        )
        permission_manager = PermissionManager(
            hook_manager=hook_manager, permissions=Application.__grants(settings)
        )
        agent = Agent(
            client=client,
            permission_manager=permission_manager,
            tool_manager=tool_manager,
            hook_manager=hook_manager,
        )
        session_manager = SessionManager(
            storage_manager, Application.__build_session_configuration(settings)
        )

        # direct tools share the same hooks and file-read guard as catalog tools
        read_file_tool = ReadFileTool(
            hook_manager=hook_manager, tool_session=tool_session
        )
        shell_command_tool = ShellCommandTool(
            hook_manager=hook_manager, tool_session=tool_session
        )
        return Application(
            agent=agent,
            client=client,
            summary_client=summary_client,
            tool_session=tool_session,
            hook_manager=hook_manager,
            settings_manager=settings_manager,
            session_manager=session_manager,
            frontend_controller=controller,
            read_file_tool=read_file_tool,
            shell_command_tool=shell_command_tool,
            startup_messages=settings_messages + plugin_messages,
        )

    def __init__(
        self,
        *,
        agent: Agent,
        client: Client,
        summary_client: Client,
        tool_session: ToolSession,
        hook_manager: HookManager,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
        frontend_controller: UIController,
        read_file_tool: ReadFileTool,
        shell_command_tool: ShellCommandTool,
        startup_messages: list[Message],
    ) -> None:
        """Store the shared graph, construct the executor and apply loaded settings.

        Args:
            agent: the Core execution and history owner.
            client: the main client to configure.
            summary_client: the summary client, whose reasoning stays disabled.
            tool_session: the read guard shared by catalog and direct tools.
            hook_manager: the manager dispatching lifecycle hooks.
            settings_manager: the already-loaded settings manager.
            session_manager: the manager holding the opening conversation.
            frontend_controller: the future frontend's abstract contract implementation.
            read_file_tool: the direct tool for explicit file mentions.
            shell_command_tool: the direct tool for shell input and directories.
            startup_messages: settings/plugin notices to show without recording.
        """
        self.__agent: Agent = agent
        self.__client: Client = client
        self.__summary_client: Client = summary_client
        self.__tool_session: ToolSession = tool_session
        self.__hook_manager: HookManager = hook_manager
        self.__settings_manager: SettingsManager = settings_manager
        self.__session_manager: SessionManager = session_manager
        self.__frontend_controller: UIController = frontend_controller
        self.__startup_messages: list[Message] = list(startup_messages)
        self.__working_directory: Path = (
            session_manager.current_state().working_directory
        )
        self.__started: bool = False
        self.__closed: bool = False
        self.__action_executor: ActionExecutor = ActionExecutor(
            application=self,
            agent=agent,
            settings_manager=settings_manager,
            session_manager=session_manager,
            tool_session=tool_session,
            frontend_controller=frontend_controller,
            read_file_tool=read_file_tool,
            shell_command_tool=shell_command_tool,
        )
        self.__apply_settings(settings_manager.settings)

    def start(self) -> list[Message]:
        """Return startup notices and STARTUP hook messages exactly once.

        These messages are neither persisted nor passed to Core history.

        Returns:
            The ordered startup messages, or an empty list after startup/shutdown.
        """
        if self.__started or self.__closed:
            return []
        self.__started = True
        messages = self.__startup_messages
        self.__startup_messages = []
        return messages + self.__hook_manager.run(Hook.STARTUP)

    def shutdown(self) -> None:
        """Cancel questions, request safe execution closure and dispatch SHUTDOWN once.

        Active blocking work cannot be interrupted. Its owner closes Core when control
        returns. No thread waits for another thread's execution to finish. Hook messages
        are cleanup output and are not displayed or recorded.
        """
        if self.__closed:
            return
        self.__closed = True
        try:
            self.__frontend_controller.cancel_questions()
            self.__action_executor.close()
        finally:
            self.__hook_manager.run(Hook.SHUTDOWN)

    def submit(self, text: str) -> Iterator[Message]:
        """Run one nonempty input line on the calling thread.

        Consumers stopping early must close the returned generator on that same
        thread so the Core driver can finish cleanup.

        Args:
            text: the original input passed unchanged to prompt parsing.

        Returns:
            The iterator yielding display Messages and temporary save notices.

        Raises:
            ValueError: when the input is empty or whitespace only.
        """
        self.__require_idle()
        if not text.strip():
            raise ValueError("Input must not be empty.")
        return self.__action_executor.execute(text)

    def status(self) -> ApplicationStatus:
        """Return current display values without exposing managed services.

        Returns:
            A fresh immutable model/effort/directory snapshot.
        """
        return ApplicationStatus(
            model=self.__settings_manager.settings.model,
            working_directory=self.__working_directory,
        )

    def update_settings(self, **changes: Any) -> str | ErrorMessage | None:
        """Apply supplied settings in dependency order, preserving completed changes.

        Changes follow the settings file layout, for example
        ``model={"reasoning": {"effort": "low"}}``. Unknown keys raise ValueError
        before any change. Values are checked by SettingsManager one field at a
        time. A later failure does not roll back earlier fields. Omitted fields
        and empty permission mappings are unchanged.

        Args:
            **changes: a ``model`` mapping with ``provider``, ``name`` and/or
                ``reasoning.effort``, and/or typed ``permissions``.

        Returns:
            Ordered confirmation text, a save notice, or None for no changes.
        """
        self.__require_idle()
        flat_changes = self.__flatten(changes)
        confirmations: list[str] = []
        for field in self.__setting_fields:
            if field not in flat_changes:
                continue
            value = flat_changes[field]
            if field == PERMISSIONS_FIELD and isinstance(value, Mapping):
                value = dict(value)
                if not value:
                    continue
            notice = self.__change_setting(field, value)
            if notice is not None:
                return notice
            confirmations.append(
                self.__setting_messages[field](self.__settings_manager.settings, value)
            )
        return "  \n".join(confirmations) if confirmations else None

    def permission_levels(self) -> dict[PermissionCategory, PermissionLevel]:
        """Return the independent category-to-level mapping used by input adapters.

        Returns:
            A copied grant map of default and plugin categories, preserving
            category identities.
        """
        return self.__grants(self.__settings_manager.settings)

    def change_directory(self, path: str | Path) -> str | ErrorMessage:
        """Change the process directory and record the successful runtime value.

        Args:
            path: an absolute or relative path, with optional home expansion.

        Returns:
            Existing confirmation text or a temporary session-save notice.
        """
        self.__require_idle()
        self.__set_working_directory(path)
        notice = self.__update_session_state(working_directory=self.__working_directory)
        return (
            notice
            if notice is not None
            else (f"Working directory set to {self.__working_directory}")
        )

    def rename(self, name: str | None) -> str | ErrorMessage:
        """Change or clear the session's explicit name.

        Args:
            name: the chosen name, or None for the session-ID fallback.

        Returns:
            Existing confirmation text or a temporary session-save notice.
        """
        self.__require_idle()
        notice = self.__update_session_state(explicit_name=name)
        return notice if notice is not None else f"Session renamed to {name!r}"

    def new_session(self) -> None:
        """Reset settings, history and read safety, then select an unsaved chat."""
        self.__require_idle()
        self.__settings_manager.reset_global()
        settings = self.__settings_manager.settings
        self.__apply_settings(settings)
        self.__agent.clear_history()
        self.__tool_session.clear()
        self.__session_manager.create(self.__build_session_configuration(settings))
        self.__frontend_controller.reset_conversation([])

    def resume(self, query: str) -> str | Message | None:
        """Select and restore a saved conversation without changing global defaults.

        A checksum failure after selection still completes runtime restoration.
        Repair/fallback save notices are returned for display only.

        Args:
            query: an exact identity or case-insensitive name substring.

        Returns:
            Disambiguation/fallback text, a save notice, or None on ordinary success.

        Raises:
            ValueError: when no match exists or the selected session is current.
        """
        self.__require_idle()
        matches = self.__session_manager.find(query)
        if not matches:
            raise ValueError(f"No session matches {query!r}.")
        if len(matches) > 1:
            listing = ", ".join(f"{entry.name} ({entry.id})" for entry in matches)
            return f"Multiple sessions match {query!r}: {listing}"
        entry = matches[0]
        if self.__session_manager.current_state().id == entry.id:
            raise ValueError(f"Session {query} is the current session.")

        fallback = self.__working_directory
        notices: list[ErrorMessage] = []
        try:
            state, repair_notice = self.__session_manager.load(entry.id)
        except OSError as exc:
            state = self.__session_manager.current_state()
            if state.id != entry.id:
                return ErrorMessage(content=f"Could not load session: {exc}")
            repair_notice = ErrorMessage(content=f"Could not save session: {exc}")
        if repair_notice is not None:
            notices.append(repair_notice)

        configuration = state.configuration
        self.__settings_manager.restore_active(
            model=configuration.model,
            permissions=configuration.permissions,
        )
        self.__apply_settings(self.__settings_manager.settings)
        self.__agent.restore_history(self.__session_manager.model_history)
        self.__tool_session.clear()
        for path, mtime in state.file_states.items():
            target = Path(path)
            try:
                if target.is_file() and target.stat().st_mtime == mtime:
                    self.__tool_session.record_read(path, mtime)
            except OSError:
                # unreadable or disappearing files must remain unread to Core
                continue

        warning = None
        directory = state.working_directory
        if not directory.is_dir():
            warning = (
                f"Saved working directory no longer exists: {directory}; "
                f"using {fallback}."
            )
            directory = fallback
        self.__set_working_directory(directory)
        if warning is not None:
            notice = self.__update_session_state(
                working_directory=self.__working_directory
            )
            if notice is not None:
                notices.append(notice)
        self.__frontend_controller.reset_conversation(
            self.__session_manager.visible_messages
        )
        if notices:
            parts = [warning] if warning is not None else []
            parts.extend(str(notice.content) for notice in notices)
            return ErrorMessage(content="\n".join(parts))
        return warning

    def fork(self) -> tuple[SessionState, AssistantMessage, list[ErrorMessage]]:
        """Return Sessions' existing fork state, confirmation and notices unchanged.

        Returns:
            The detached child state, saved/display-only confirmation and notices.
        """
        self.__require_idle()
        return self.__session_manager.fork()

    def exit(self) -> None:
        """Request frontend closure; its lifecycle must call shutdown."""
        self.__frontend_controller.exit()

    def __require_idle(self) -> None:
        """Reject new work after shutdown or while a Core execution is active.

        Raises:
            RuntimeError: when Application is closed or Core is active.
        """
        if self.__closed:
            raise RuntimeError("Application is shut down.")
        if self.__action_executor.is_running:
            raise RuntimeError("Core execution is active.")

    def __apply_settings(self, settings: Settings) -> None:
        """Apply a checked snapshot through Core's public configuration seams.

        Args:
            settings: the detached active settings to apply.
        """
        # each client formats the id its provider expects
        provider, name = settings.model.provider, settings.model.name
        self.__client.model_id = self.__client.get_model_id(provider, name)
        self.__client.reasoning = settings.model.reasoning
        self.__summary_client.model_id = self.__summary_client.get_model_id(
            provider, name
        )
        self.__summary_client.reasoning = Reasoning()
        self.__agent.turn = settings.turn
        self.__agent.set(self.__grants(settings))

    @classmethod
    def __flatten(cls, changes: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        """Turn nested setting changes into values keyed by their dotted paths.

        An editable field is a leaf, so a permissions mapping is kept whole.

        Args:
            changes: the nested changes, following the settings file layout.
            prefix: the dotted path of ``changes``, ending with a dot; empty at
                the top level.

        Returns:
            Each changed value keyed by the dotted path of its field.

        Raises:
            ValueError: when a key is not an editable field or section, or a
                section is not a mapping.
        """
        flat: dict[str, Any] = {}
        for key, value in changes.items():
            path = f"{prefix}{key}"
            if path in cls.__setting_fields:
                flat[path] = value
            elif any(field.startswith(f"{path}.") for field in cls.__setting_fields):
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Application setting {path!r} must be a mapping: {value!r}"
                    )
                flat.update(cls.__flatten(value, f"{path}."))
            else:
                raise ValueError(
                    f"Unknown editable application setting {path!r}: {value!r}"
                )
        return flat

    def __change_setting(self, field: str, value: Any) -> ErrorMessage | None:
        """Complete one edit/apply/save/record sequence without cross-layer rollback.

        Permission changes of a loaded plugin's category are applied and saved as
        that plugin's ``permission_level``; the others as default categories.

        Args:
            field: the dotted path of the setting selected by update_settings.
            value: the proposed typed value or permission changes.

        Returns:
            A settings/session-save notice, or None on success.
        """
        plugin_levels: dict[str, PermissionLevel] = {}
        if field == PERMISSIONS_FIELD and isinstance(value, Mapping):
            plugin_categories = self.__plugin_categories(
                self.__settings_manager.settings
            )
            plugin_levels = {
                plugin_categories[category]: level
                for category, level in value.items()
                if category in plugin_categories
            }
            value = {
                category: level
                for category, level in value.items()
                if category not in plugin_categories
            }
        self.__settings_manager.set_field(field, value)
        if plugin_levels:
            self.__settings_manager.set_plugin_permission_levels(plugin_levels)
        settings = self.__settings_manager.settings
        self.__apply_settings(settings)
        try:
            if field == PERMISSIONS_FIELD:
                for category in value:
                    self.__settings_manager.save_field(field, category)
                for name in plugin_levels:
                    self.__settings_manager.save_plugin_permission_level(name)
            else:
                self.__settings_manager.save_field(field)
        except OSError as exc:
            return ErrorMessage(content=f"Could not save settings: {exc}")
        return self.__update_session_state(
            configuration=self.__build_session_configuration(settings)
        )

    def __update_session_state(self, **changes: Any) -> ErrorMessage | None:
        """Update through Sessions without recursively saving a checksum failure.

        Args:
            **changes: the supported SessionManager state replacements.

        Returns:
            A temporary save notice or None on success.
        """
        try:
            return self.__session_manager.update_state(**changes)
        except OSError as exc:
            return ErrorMessage(content=f"Could not save session: {exc}")

    def __set_working_directory(self, path: str | Path) -> None:
        """Resolve and validate a path, then store a successful process-directory move.

        Args:
            path: a path resolved relative to the owned directory, expanding home.

        Raises:
            ValueError: when the target is missing or not a directory.
        """
        resolved = (self.__working_directory / Path(path).expanduser()).resolve()
        if not resolved.exists():
            raise ValueError(f"{resolved} does not exist")
        if not resolved.is_dir():
            raise ValueError(
                f"{resolved} not a directory: did you mean {resolved.parent}?"
            )
        os.chdir(resolved)
        self.__working_directory = resolved

    @staticmethod
    def __build_session_configuration(settings: Settings) -> SessionConfiguration:
        """Take the part of the settings that each session saves.

        Args:
            settings: the active settings snapshot.

        Returns:
            A detached configuration with the model section and the default
            permission levels.
        """
        return SessionConfiguration(
            model=settings.model,
            permissions=dict(settings.permissions),
        )

    @staticmethod
    def __plugin_categories(settings: Settings) -> dict[PermissionCategory, str]:
        """Find the permission category of each plugin that created one.

        A plugin category exists only after its plugin was loaded with a ``CUSTOM``
        tool, so plugins that are disabled or have no such tool add nothing.

        Args:
            settings: the snapshot holding the plugin entries.

        Returns:
            The existing ``PLUGIN:<name>`` categories mapped to their lowercase
            plugin names.
        """
        categories: dict[PermissionCategory, str] = {}
        for name in settings.plugins:
            category = PermissionCategory.get(
                f"{PLUGIN_PERMISSION_CATEGORY_PREFIX}{name}"
            )
            if category is not None:
                categories[category] = name
        return categories

    @staticmethod
    def __grants(settings: Settings) -> dict[PermissionCategory, PermissionLevel]:
        """Combine default levels with the levels of existing plugin categories.

        Args:
            settings: the snapshot holding default levels and plugin entries.

        Returns:
            A new category to level mapping for Core.
        """
        grants = dict(settings.permissions)
        for category, name in Application.__plugin_categories(settings).items():
            grants[category] = settings.plugins[name].permission_level
        return grants

    @staticmethod
    def __format_permission_changes(
        settings: Settings,
        changes: Mapping[PermissionCategory, PermissionLevel],
    ) -> str:
        """Preserve existing single/all wording and format mixed changes per category.

        Args:
            settings: the checked active snapshot after the edit.
            changes: the supplied nonempty category-to-level changes.

        Returns:
            The existing confirmation lines, joined with Markdown hard breaks.
        """
        grants = Application.__grants(settings)
        if (
            len(changes) > 1
            and set(changes) == set(grants)
            and len(set(changes.values())) == 1
        ):
            level = next(iter(changes.values()))
            return (
                "all permissions removed"
                if level is PermissionLevel.NONE
                else f"all permission levels set to {level.value}"
            )
        lines: list[str] = []
        for category in changes:
            name = category.name.lower()
            level = grants[category]
            lines.append(
                f"{name} permission removed"
                if level is PermissionLevel.NONE
                else f"{name} permission level set to {level.value}"
            )
        return "  \n".join(lines)
