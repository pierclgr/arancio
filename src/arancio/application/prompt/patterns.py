"""Prompt parsing patterns."""

import re

# a prompt is a command when it starts with a slash: /<name> <args...>
COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^/(\S+)(?:\s+(.*))?$")

# a hidden shell command starts with !!; a history shell command starts with
# exactly one !; DOTALL keeps the complete remainder as executable shell text
HIDDEN_SHELL_COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^!!(.*)$", re.DOTALL)
SHELL_COMMAND_PATTERN: re.Pattern[str] = re.compile(r"^!(?!!)(.*)$", re.DOTALL)

# quoted paths can put their selector inside or immediately after the quotes
MENTION_PATTERN: re.Pattern[str] = re.compile(
    r'@"(?P<quoted>[^"]+)"(?P<suffix>:\S*)?|@(?P<bare>\S+)'
)

# match the complete selector so malformed suffixes fall back to whole-file reads
LINE_RANGE_PATTERN: re.Pattern[str] = re.compile(r"([0-9]+)(?:-([0-9]+))?")
