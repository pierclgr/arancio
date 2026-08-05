"""Prompt parsing patterns."""

import re

# a prompt is a command when it starts with a slash: /<name> <args...>
COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^/(\S+)(?:\s+(.*))?$")

# a hidden shell command starts with !!; a history shell command starts with
# exactly one !; DOTALL keeps the complete remainder as executable shell text
HIDDEN_SHELL_COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^!!(.*)$", re.DOTALL)
SHELL_COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^!(?!!)(.*)$", re.DOTALL)

# a token starting with @ is a candidate file mention: a quoted @"path with spaces"
# or a bare @path ending at the next whitespace; mirrors dynamic_markdown's own
# "@path" include syntax, extended with quoting since real paths can contain spaces
MENTION_PATTERN: re.Pattern[str] = re.compile(r'@"([^"]+)"|@(\S+)')
