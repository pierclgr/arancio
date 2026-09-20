"""Fork command."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand
from arancio.core.messages import AssistantMessage
from arancio.sessions.session import SessionConfiguration

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager

# a trailing ":main" or ":fork_<14-digit timestamp>" marks a name as already
# belonging to a fork lineage; stripped so re-forking chains off the original
# name instead of piling up suffixes
_LINEAGE_SUFFIX = re.compile(r":(main|fork_\d{14})$")


class ForkCommand(BaseCommand):
    """Command that duplicates the active session into a new active session."""

    name = "fork"
    description = "Fork the active session into a new one."

    @classmethod
    def execute(cls, session_manager: SessionManager) -> str:
        """Copy the active session's history onto a new session and switch to it.

        ``SessionManager.create`` only builds a fresh session identity; this
        command decides what a fork actually carries over — every event but
        the header, and the file-read safety state — since that duplication
        is specific to forking, not generic session creation.

        When the source session has a name and is not itself already a fork,
        it is renamed to ``{base_name}:main`` so the lineage is visible at a
        glance; the new child is always named ``{base_name}:fork_<timestamp>``.
        An unnamed source is left untouched and the child stays unnamed, same
        as before this naming convention existed.

        Forking a chat that was never saved switches to the fork without
        writing anything: there is nothing to copy, and a fork of nothing does
        not deserve a log of its own. Otherwise the confirmation is recorded
        into both the source's log and the fork's, so either one's replayed
        history shows the fork happened.

        Args:
            session_manager: the active session manager, whose current
                session is forked; the fork becomes the new active session.

        Returns:
            Confirmation text naming the source session and the new fork,
            by name when the source has one, by id otherwise.
        """
        source = session_manager.current
        had_name = source.explicit_name is not None
        is_already_fork = source.forked_from is not None
        # snapshotted before any rename below, so a source-only state_changed
        # record never leaks into the fork's copied history
        events_to_copy = list(source.events[1:])

        base_name = _LINEAGE_SUFFIX.sub("", source.explicit_name) if had_name else None
        if had_name and not is_already_fork:
            source.name = f"{base_name}:main"
            session_manager.session_recorder.state_changed()

        forked_name = None
        if had_name:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            forked_name = f"{base_name}:fork_{timestamp}"

        forked = session_manager.create(
            working_directory=source.working_directory,
            configuration=SessionConfiguration.from_dict(
                source.configuration.to_dict()
            ),
            explicit_name=forked_name,
            forked_from=source.id,
        )
        for event in events_to_copy:
            forked.add_event(dict(event.record), saved=False)
        forked.file_states = dict(source.file_states)

        confirmation = (
            f"Session {source.name} forked to {forked.name}"
            if had_name
            else f"Session {source.id} forked to {forked.id}"
        )
        if source.created_on_disk:
            session_manager.session_recorder.flush()
            session_manager.session_recorder.message_into(
                source, AssistantMessage(content=confirmation, in_history=False)
            )
        return confirmation
