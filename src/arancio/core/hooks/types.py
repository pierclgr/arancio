"""Names of the points the agent, tools and permission manager can dispatch against."""

from enum import Enum


class Hook(Enum):
    """A named point in the agent's execution a handler can register against.

    Dispatched from :meth:`~arancio.core.agents.Agent.__call__` (most
    members, plus ``ERROR`` with ``source="model"``/``"agent"``),
    :meth:`~arancio.core.agents.Agent._system_prompt`
    (``SYSTEM_PROMPT_BUILD``), :meth:`~arancio.core.tools.base.BaseTool.call`
    (the ``TOOL_CALL`` members, plus ``ERROR`` with ``source="tool"``) and
    :meth:`~arancio.core.permissions.manager.PermissionManager.validate`
    (the ``PERMISSION_CHECK`` members); see AGENTS.md's Hooks section for the
    full call-site and keyword-argument contract.

    Attributes:
        AGENT_START: the agent loop begins.
        AGENT_END: the agent loop ends normally.
        TURN_START: a turn begins.
        TURN_END: a turn ends.
        SYSTEM_PROMPT_BUILD: the system prompt is built.
        BEFORE_MODEL_REQUEST: a request is about to be sent to the model.
        AFTER_MODEL_RESPONSE: a response was received from the model.
        MESSAGE_RECEIVED: a message was received.
        BEFORE_TOOL_CALL: a tool call is about to run.
        AFTER_TOOL_CALL: a tool call finished.
        BEFORE_PERMISSION_CHECK: a permission check is about to run.
        AFTER_PERMISSION_CHECK: a permission check finished.
        ERROR: an error happened — a tool call failed, a model request
            failed, or the agent loop is ending on an unrecovered error.
            Dispatched with a ``source`` kwarg (``"tool"``, ``"model"`` or
            ``"agent"``) naming which of the three it was; see AGENTS.md's
            Hooks section for the rest of each source's kwargs.
    """

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    SYSTEM_PROMPT_BUILD = "system_prompt_build"
    BEFORE_MODEL_REQUEST = "before_model_request"
    AFTER_MODEL_RESPONSE = "after_model_response"
    MESSAGE_RECEIVED = "message_received"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_PERMISSION_CHECK = "before_permission_check"
    AFTER_PERMISSION_CHECK = "after_permission_check"
    ERROR = "error"
