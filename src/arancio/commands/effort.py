"""Effort command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class EffortCommand(BaseCommand):
    """Command that sets the model's thinking effort."""

    name = "effort"
    description = "Set the model's thinking effort."

    @classmethod
    def execute(
        cls, level: str, application: App, settings_manager: SettingsManager
    ) -> str:
        """Replace the thinking effort and apply it.

        Any text is accepted as the effort level; no fixed set of values is
        enforced. The literal word ``"null"`` (case-insensitive) disables
        thinking entirely. Accessing ``settings.model_id`` below raises
        ``ValueError`` before anything is persisted when no model is
        currently configured.

        Args:
            level: the new thinking effort, bound to the prompt's first word;
                ``"null"`` (any case) disables thinking.
            application: the running app whose toolbar is refreshed with the
                new effort level.
            settings_manager: the manager used to apply and persist the
                change.

        Returns:
            Confirmation text naming the new effort level.
        """
        settings_manager.settings.model_id
        new_effort = None if level.lower() == "null" else level
        settings_manager.settings.thinking_effort = new_effort
        settings_manager.apply()
        settings_manager.save_thinking_effort()
        application.set_displayed_effort(new_effort)
        return (
            f"Thinking effort set to {new_effort if new_effort is not None else 'null'}"
        )
