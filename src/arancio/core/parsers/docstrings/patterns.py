"""Docstring parameter section and entry patterns."""

import re

# parameter sections: the whole block of parameters, captured as the first group

# "Args:", then the indented or blank lines that follow it
GOOGLE_PARAMETERS_SECTION_PATTERN = re.compile(
    r"^Args:\n((?:[ \t]+[^\n]*(?:\n|\Z)|\n)*)", re.MULTILINE
)
# "Parameters" and its dashes line, then the body up to the next heading (a line
# followed by dashes) or the end of the docstring
NUMPY_PARAMETERS_SECTION_PATTERN = re.compile(
    r"^Parameters\n-+\n((?s:.*?))(?=^\S[^\n]*\n-+$|\Z)", re.MULTILINE
)

# parameter entries: one parameter line, with the name as the first group

# "    name: text" or "    name (type): text"
GOOGLE_PARAMETER_ENTRY_PATTERN = re.compile(r"^ {4}(\w+)(?: \([^)]*\))?:\s*(.*)$")
# "name" or "name : type", with the description on the following indented lines
NUMPY_PARAMETER_ENTRY_PATTERN = re.compile(r"^(\w+)(?: *:.*)?$")
# ":param name: text" or ":param type name: text"
SPHINX_PARAMETER_ENTRY_PATTERN = re.compile(r"^:param (?:[^:]+ )?(\w+):\s*(.*)$")
