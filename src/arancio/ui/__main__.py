"""Entry point wiring the agent stack to the Textual UI and running it."""

from pathlib import Path

from arancio.core.agents import Agent
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.permissions.manager import PermissionManager
from arancio.core.tools.manager import ToolManager
from arancio.core.tools.session import shared_session
from arancio.sessions.manager import SessionManager
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager
from arancio.storage.manager import StorageManager
from arancio.ui.app import App
from arancio.ui.controller import UIController


def main() -> None:
    """Wire the client, managers, agent and controller, then run the UI.

    The controller is built first (the app it drives needs the agent, which needs the
    permission manager, which needs the controller); its ``app`` is assigned once the
    app exists.
    """
    storage_manager = StorageManager()
    storage_manager.bind_litellm_login_dir()
    session_manager = SessionManager(storage_manager, tool_session=shared_session)

    controller = UIController()
    client = LiteLLMClient(stream=True, controller=controller)
    summary_client = LiteLLMClient(stream=False, controller=controller)
    # the summary client never thinks (setters take None; the constructor ignores it)
    summary_client.thinking_effort = None
    summary_client.thinking_summary = None
    tool_manager = ToolManager(web_summary_client=summary_client)
    permission_manager = PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
    )
    agent = Agent(client=client, permission_manager=permission_manager)

    settings_manager = SettingsManager(
        storage_manager=storage_manager,
        client=client,
        summary_client=summary_client,
        agent=agent,
    )
    settings, startup_messages = settings_manager.load()
    session_manager.create(
        working_directory=Path.cwd(),
        configuration=SessionConfiguration.from_settings(settings),
    )

    app = App(
        agent=agent,
        model_id=client.model_id,
        settings_manager=settings_manager,
        session_manager=session_manager,
        startup_messages=startup_messages,
    )
    controller.app = app
    app.run()


if __name__ == "__main__":
    main()
