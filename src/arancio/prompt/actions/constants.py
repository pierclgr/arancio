"""Constants shared by the action executor and the commands it runs."""

# execute() parameter names the action executor injects itself instead of
# binding them to prompt words; a command declares one of these by name to
# receive it
INJECTABLE_COMMAND_PARAMETERS = frozenset({"application", "settings_manager", "agent"})
