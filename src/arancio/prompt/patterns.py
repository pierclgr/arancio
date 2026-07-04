"""Prompt parsing patterns."""

import re

# a prompt is a command when it starts with a slash: /<name> <args...>
COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^/(\S+)(?:\s+(.*))?$")
