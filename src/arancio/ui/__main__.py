"""Entry point wiring the agent stack to the Textual UI and running it."""

import os

from arancio.core.agents import Agent
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.permissions.manager import PermissionManager
from arancio.core.tools.manager import ToolManager
from arancio.core.types.permissions import PermissionCategory, PermissionLevel
from arancio.ui.app import App
from arancio.ui.controller import UIController

DEFAULT_MODEL_ID = "ollama_chat/glm-5.2:cloud"


def main() -> None:
    """Wire the client, managers, agent and controller, then run the UI.

    The controller is built first (the app it drives needs the agent, which needs the
    permission manager, which needs the controller); its ``app`` is assigned once the
    app exists.
    """
    model_id = os.environ.get("ARANCIO_MODEL", DEFAULT_MODEL_ID)
    controller = UIController()
    client = LiteLLMClient(model_id=model_id, stream=True)
    tool_manager = ToolManager(web_summary_client=client)
    permission_manager = PermissionManager(
        tool_manager=tool_manager, controller=controller
    )
    agent = Agent(client=client, permission_manager=permission_manager)

    print(agent)
    client.model_id = "ollama_chat/deepseek-v4-pro:cloud"
    client.thinking_effort = "high"
    client.thinking_summary = None  # to use with Ollama models
    agent._max_turns = 20
    # agent.remove_permission(PermissionCategory.EXECUTE)
    agent.set_permission_level(PermissionCategory.WEB, PermissionLevel.AUTO)
    print(agent)

    app = App(agent=agent, model_id=model_id)
    controller.app = app
    app.run()


if __name__ == "__main__":
    main()
