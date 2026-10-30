"""Public Sessions facade for conversation data, recording and reconstruction."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from arancio.core.messages.assistant import AssistantMessage
from arancio.core.messages.base import ErrorMessage, Message
from arancio.core.messages.user import UserMessage
from arancio.core.permissions.types import PermissionCategory
from arancio.sessions.base import (
    Session,
    SessionConfiguration,
    SessionRegistryEntry,
    SessionState,
)
from arancio.sessions.codec import is_message_record, message_from_record
from arancio.sessions.constants import (
    ARANCIO_SESSIONS_DIR,
    SESSION_DIRECTORY_HASH_LENGTH,
    SESSION_DIRECTORY_SLUG_MAX_LENGTH,
    SESSION_LINEAGE_SUFFIX_PATTERN,
)
from arancio.sessions.recorder import SessionRecorder
from arancio.sessions.registry import SessionRegistry
from arancio.sessions.validator import SessionValidator
from arancio.storage.manager import StorageManager


class SessionManager:
    """Own the active Session and expose session operations and snapshots.

    Application supplies configuration and file-read snapshots and applies restored
    values to runtime services. Sessions owns persistence and record repair.

    Attributes:
        __storage_manager: the injected file I/O service.
        __sessions_dir: the session location under Storage's root.
        __categories: the known permission-category identities at construction.
        __session: the managed active conversation.
        __recorder: the record builder and writer.
        __registry: the reconstructed saved-session index.
        __validator: the shared discovery, load and state-change validator.
        __pending_sources: saved fork sources whose writes still need retrying.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        configuration: SessionConfiguration,
    ) -> None:
        """Discover saved sessions and create an unsaved opening conversation.

        Session logs live under the supplied Storage root, as they did before
        the explicit-location refactor.

        Args:
            storage_manager: the generic file I/O service.
            configuration: the already-loaded opening configuration.
        """
        self.__storage_manager = storage_manager
        self.__sessions_dir = self.__storage_manager.root / ARANCIO_SESSIONS_DIR
        self.__categories = tuple(PermissionCategory)
        self.__pending_sources: list[Session] = []
        self.__validator = SessionValidator(self.__storage_manager, self.__categories)
        # reject invalid opening values before discovery can create a directory
        self.__validator.validate_state_changes({"configuration": configuration})
        self.__registry = SessionRegistry(
            self.__storage_manager, self.__sessions_dir, self.__validator
        )
        self.__recorder = SessionRecorder(
            self.__storage_manager, lambda: self.__current, self.__after_write
        )
        self.__session: Session = self.create(configuration)

    def current_state(self) -> SessionState:
        """Return independent current values without events or retry internals.

        Returns:
            A detached snapshot whose mutable values cannot change managed state.
        """
        session = self.__current
        return SessionState(
            id=session.id,
            created_at=session.created_at,
            creation_working_directory=session.creation_working_directory,
            working_directory=session.working_directory,
            path=session.path,
            configuration=session.configuration.copy(),
            explicit_name=session.explicit_name,
            forked_from=session.forked_from,
            file_states=dict(session.file_states),
            created_on_disk=session.created_on_disk,
            has_pending_events=any(not event.saved for event in session.events),
        )

    def create(
        self,
        configuration: SessionConfiguration,
        explicit_name: str | None = None,
    ) -> Session:
        """Select a new conversation without writing its creation header.

        The new conversation starts in the current process working directory.
        Unsaved active records are discarded. Application clears runtime history
        and file-read guards separately.

        Args:
            configuration: the configuration values to copy.
            explicit_name: the chosen name, or None for the ID fallback.

        Returns:
            The newly created and selected session.
        """
        return self.__create(Path.cwd(), configuration, explicit_name, forked_from=None)

    def find(self, query: str) -> list[SessionRegistryEntry]:
        """Search saved sessions and return independent entry data.

        Args:
            query: an exact identity or case-insensitive name substring.

        Returns:
            All matching entries, including damaged and duplicate identities.
        """
        return [
            replace(
                entry,
                configuration=(
                    entry.configuration.copy()
                    if entry.configuration is not None
                    else None
                ),
            )
            for entry in self.__registry.find(query)
        ]

    def load(self, session_id: str) -> tuple[SessionState, ErrorMessage | None]:
        """Validate, select and repair a saved conversation without runtime changes.

        A missing or stale checksum is refreshed after successful full validation.
        Validation and checksum errors propagate. Repair-save notices are returned
        with the selected state, retaining all unsaved synthetic results for retry.

        Args:
            session_id: the unambiguous saved-session identity.

        Returns:
            Detached selected state and an optional repair-save notice.
        """
        entry = self.__registry.get(session_id)
        session = self.__validator.read(entry.log_path)
        self.__validator.certify(entry.log_path)
        self.__session = session
        error = self.__recorder.close_interrupted_tool_calls()
        if error is None:
            self.__registry.update(session)
        return self.current_state(), error

    def record_message(
        self,
        message: Message,
        *,
        file_states: Mapping[str, float],
        visible: bool = True,
    ) -> ErrorMessage | None:
        """Record one message and the caller's copied read-state snapshot.

        Chunks are skipped; unsupported types raise ValueError through the codec.
        Save notices are returned to Application without entering Core or this log.

        Args:
            message: the normalized message to save.
            file_states: the supplied absolute-path-to-mtime snapshot.
            visible: whether the message belongs in visible replay.

        Returns:
            A temporary save notice, or None when saved or skipped.
        """
        return self.__recorder.message(
            message, file_states=dict(file_states), visible=visible
        )

    def record_command(
        self, raw_input: str, name: str, args: list[str]
    ) -> ErrorMessage | None:
        """Record a typed invocation without executing it.

        Application decides command attribution before switching sessions.

        Args:
            raw_input: the exact text supplied by the user.
            name: the parsed command name or shell marker.
            args: the parsed argument list to copy.

        Returns:
            A temporary save notice, or None on success.
        """
        return self.__recorder.command(raw_input, name, args)

    def update_state(self, **changes: Any) -> ErrorMessage | None:
        """Validate supplied replacements and persist one complete state snapshot.

        Omitted keys are unchanged. An explicit_name of None clears the name;
        configuration and working_directory cannot be None. Unknown keys and bad
        values raise ValueError before mutation or writing. A failed save retains
        the changed in-memory fields and pending event.

        Args:
            **changes: configuration, working_directory and/or explicit_name values.

        Returns:
            A temporary save notice, or None on success or an empty update.
        """
        if not changes:
            return None
        replacements = self.__validator.validate_state_changes(changes)
        session = self.__current
        for field, value in replacements.items():
            setattr(session, field, value)
        return self.__recorder.state_changed()

    def fork(
        self,
    ) -> tuple[SessionState, AssistantMessage, list[ErrorMessage]]:
        """Select a copied conversation and save confirmations for a saved source.

        Events are captured before a source-only rename. The child's final state
        follows copied state records so its name survives loading. Unused sources
        and children remain unsaved. During confirmation recording, checksum
        failures propagate after both writes have been attempted; queued writes
        remain retryable.

        Returns:
            Detached child state, a display-only confirmation, and save notices.
        """
        source = self.__current
        source_was_saved = source.created_on_disk
        events_to_copy = list(source.events[1:])
        had_name = source.explicit_name is not None
        base_name = (
            re.sub(SESSION_LINEAGE_SUFFIX_PATTERN, "", source.explicit_name)
            if had_name
            else None
        )
        notices: list[ErrorMessage] = []
        if had_name and source.forked_from is None:
            if source_was_saved:
                error = self.update_state(explicit_name=f"{base_name}:main")
                if error is not None:
                    notices.append(error)
            else:
                source.explicit_name = f"{base_name}:main"

        forked_name = None
        if had_name:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            forked_name = f"{base_name}:fork_{timestamp}"
        child = self.__create(
            source.working_directory,
            source.configuration,
            forked_name,
            forked_from=source.id,
        )
        for event in events_to_copy:
            child.add_event(event.record)
        child.file_states = dict(source.file_states)
        if any(event.record.get("type") == "state_changed" for event in events_to_copy):
            self.__recorder.queue_state()

        confirmation = AssistantMessage(
            content=f"Session {source.name} forked to {child.name}",
            in_history=False,
        )
        if source_was_saved:
            self.__pending_sources.append(source)
            checksum_error: OSError | None = None
            for session in (child, source):
                try:
                    error = self.__recorder.message_into(session, confirmation)
                except OSError as exc:
                    if checksum_error is None:
                        checksum_error = exc
                else:
                    if error is not None:
                        notices.append(error)
            if checksum_error is not None:
                raise checksum_error
        return self.current_state(), confirmation, notices

    def flush(self) -> ErrorMessage | None:
        """Retry retained fork sources, then flush the active conversation.

        Log failures do not prevent attempts for the remaining sessions. Checksum
        failures propagate, leaving the affected session available for another try.

        Returns:
            The first temporary log-save notice, or None when all writes complete.
        """
        first_error = None
        for session in (*self.__pending_sources, self.__current):
            error = self.__recorder.flush(session)
            if first_error is None:
                first_error = error
        return first_error

    @property
    def model_history(self) -> list[Message]:
        """Rebuild independent messages selected for model context.

        Returns:
            Model-history messages in stored event order.
        """
        return [
            message_from_record(deepcopy(event.record))
            for event in self.__current.events
            if is_message_record(event.record)
            and event.record.get("in_history") is True
        ]

    @property
    def visible_messages(self) -> list[Message]:
        """Rebuild independent visible messages, including typed command lines.

        Returns:
            Visible messages in stored event order.
        """
        messages: list[Message] = []
        for event in self.__current.events:
            record = event.record
            if record.get("visible") is not True:
                continue
            if record.get("type") == "command":
                messages.append(UserMessage(content=record["raw_input"]))
            elif is_message_record(record):
                messages.append(message_from_record(deepcopy(record)))
        return messages

    @property
    def __current(self) -> Session:
        """Give the internal recorder access to the managed active session.

        Returns:
            The active Session.
        """
        return self.__session

    def __after_write(self, session: Session) -> None:
        """Publish durable metadata and release completed fork-source retries.

        Args:
            session: the session whose log and checksum were saved.
        """
        self.__registry.update(session)
        self.__pending_sources = [
            pending for pending in self.__pending_sources if pending is not session
        ]

    def __create(
        self,
        working_directory: Path,
        configuration: SessionConfiguration,
        explicit_name: str | None,
        forked_from: str | None,
    ) -> Session:
        """Validate values, select a fresh identity and queue its creation record.

        Args:
            working_directory: the absolute creation/current working directory.
            configuration: the configuration to validate and copy.
            explicit_name: the optional chosen name.
            forked_from: the optional source identity.

        Returns:
            The managed new session, for internal fork coordination only.
        """
        checked = self.__validator.validate_state_changes(
            {
                "configuration": configuration,
                "working_directory": working_directory,
                "explicit_name": explicit_name,
            }
        )
        session_id = self.create_new_id()
        created_at = datetime.now().astimezone()
        session_path = (
            self.__sessions_dir
            / self.__directory_slug(working_directory)
            / created_at.strftime("%Y")
            / created_at.strftime("%m")
            / created_at.strftime("%d")
            / f"{session_id}.jsonl"
        )
        session = Session(
            id=session_id,
            created_at=created_at,
            creation_working_directory=working_directory,
            working_directory=working_directory,
            path=session_path,
            configuration=checked["configuration"],
            explicit_name=checked["explicit_name"],
            forked_from=forked_from,
            created_on_disk=False,
        )
        self.__session = session
        self.__recorder.created()
        return session

    def create_new_id(self) -> str:
        """Generate an identity absent from the saved-session index.

        Returns:
            A new UUID4 hexadecimal identity.
        """
        session_id = uuid.uuid4().hex
        while self.__registry.contains(session_id):
            session_id = uuid.uuid4().hex
        return session_id

    @staticmethod
    def __directory_slug(path: Path) -> str:
        """Preserve the readable creation-directory name and hash suffix.

        Args:
            path: the supplied absolute creation directory.

        Returns:
            A filesystem-safe readable slug with a collision-resistant suffix.
        """
        raw_path = str(path)
        readable = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_path).strip("-")
        readable = (readable or "root")[:SESSION_DIRECTORY_SLUG_MAX_LENGTH]
        digest = hashlib.sha256(raw_path.encode("utf-8")).hexdigest()
        return f"--{readable}--{digest[:SESSION_DIRECTORY_HASH_LENGTH]}"
