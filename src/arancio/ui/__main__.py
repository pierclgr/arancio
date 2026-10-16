"""Entry point wiring the agent stack to the Textual UI and running it."""

from arancio.core.agents import Agent
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.hooks.manager import HookManager
from arancio.core.permissions.manager import PermissionManager
from arancio.core.plugins.manager import PluginManager
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
    app exists. The hook manager is built alongside the controller and shared
    identically into ``tool_manager``, ``permission_manager``, ``agent`` and ``app``
    (which forwards it to its executor's own tool instances), so one handler registered
    anywhere in that graph sees every dispatch from every component. The plugin manager
    loads straight after, so the plugins are attached to that same hook manager before
    anything can dispatch, and before ``settings_manager.load()`` applies the permission
    grants and rebuilds the tool catalog. Its messages are shown alongside the settings
    ones. The session manager comes after the settings are loaded, because the session
    it opens snapshots them.
    """
    storage_manager = StorageManager()
    storage_manager.bind_litellm_login_dir()

    controller = UIController()
    hook_manager = HookManager()
    plugin_manager = PluginManager(hook_manager=hook_manager)
    plugin_messages = plugin_manager.load()
    client = LiteLLMClient(stream=True, controller=controller)
    summary_client = LiteLLMClient(stream=False, controller=controller)
    # the summary client never thinks (setters take None; the constructor ignores it)
    summary_client.thinking_effort = None
    summary_client.thinking_summary = None
    tool_manager = ToolManager(
        web_summary_client=summary_client, hook_manager=hook_manager
    )
    permission_manager = PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
        hook_manager=hook_manager,
    )
    agent = Agent(
        client=client, permission_manager=permission_manager, hook_manager=hook_manager
    )

    settings_manager = SettingsManager(
        storage_manager=storage_manager,
        client=client,
        summary_client=summary_client,
        agent=agent,
    )
    settings, settings_messages = settings_manager.load()
    startup_messages = plugin_messages + settings_messages

    session_manager = SessionManager(
        storage_manager,
        configuration=SessionConfiguration.from_settings(settings),
        tool_session=shared_session,
    )

    app = App(
        agent=agent,
        model_id=client.model_id,
        settings_manager=settings_manager,
        session_manager=session_manager,
        hook_manager=hook_manager,
        startup_messages=startup_messages,
    )
    controller.app = app
    # leaves the terminal's own mouse-tracking off, so the terminal emulator's native
    # text selection works instead of Textual capturing clicks/drags
    app.run(mouse=False)


if __name__ == "__main__":
    main()
