"""Resume command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.core.agents import Agent
    from arancio.sessions.manager import SessionManager
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class ResumeCommand(BaseCommand):
    """Command that reopens a previous chat session."""

    name = "resume"
    description = "Resume a previous session by ID or name."
    joins_arguments = True

    @classmethod
    def execute(
        cls,
        query: str,
        application: App,
        agent: Agent,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
    ) -> str | None:
        """Resolve ``query`` against the registry and resume the one match.

        An exact session ID match wins outright; otherwise every session
        whose name contains ``query`` is a candidate. No match is an error,
        more than one is reported instead of guessed. A resumed session may
        belong to a different directory than the one the app is currently
        running in — the app's working directory (and the process's, via
        ``os.chdir``) is moved to match it, unconditionally, even when it is
        already the same directory, so this is always the single source of
        truth for "where the resumed session's directory ended up" rather
        than something a caller has to special-case.

        Args:
            query: a session ID or a fragment of its name, bound to every
                prompt word, joined with a space.
            application: the running app whose working directory, log and
                toolbar are updated to match the resumed session.
            agent: the agent whose conversation history is replaced.
            settings_manager: the active settings manager, which receives
                the resumed session's saved configuration.
            session_manager: the active session manager, whose registry is
                searched and whose active session is replaced.

        Returns:
            The disambiguation listing when more than one session matches; a
            warning when the resumed session's saved directory no longer
            exists and the launch directory was used instead; or ``None`` on
            an ordinary successful resume — the repopulated log is itself
            the confirmation, the same way ``/clear`` never announces its
            own effect.

        Raises:
            ValueError: when no session matches ``query``, or when the match
                is already the current session.
        """
        matches = session_manager.registry.find(query)
        if not matches:
            raise ValueError(f"No session matches {query!r}.")
        if len(matches) > 1:
            listing = ", ".join(f"{entry.name} ({entry.id})" for entry in matches)
            return f"Multiple sessions match {query!r}: {listing}"

        entry = matches[0]
        if session_manager.current.id == entry.id:
            raise ValueError(f"Session {query} is the current session.")

        working_directory, warning = session_manager.restore_runtime(
            entry.id,
            agent,
            settings_manager,
            launch_directory=application.working_directory,
        )
        application.set_working_directory(working_directory)
        application.clear_log()
        application.populate_log()
        settings = settings_manager.settings
        if settings.provider and settings.model_name:
            application.set_displayed_model_id(settings.model_id)
        application.set_displayed_effort(settings.thinking_effort)
        return warning
