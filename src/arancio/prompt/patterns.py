"""Prompt parsing patterns."""

import re

# a prompt is a command when it starts with a slash: /<name> <args...>
COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^/(\S+)(?:\s+(.*))?$")

# a token starting with @ is a candidate file mention: a quoted @"path with spaces"
# or a bare @path ending at the next whitespace; mirrors dynamic_markdown's own
# "@path" include syntax, extended with quoting since real paths can contain spaces
MENTION_PATTERN: re.Pattern[str] = re.compile(r'@"([^"]+)"|@(\S+)')
