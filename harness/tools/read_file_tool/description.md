Reads a text file from the local filesystem and returns its content with `cat -n`-style 1-indexed line-number prefixes, so the output can be used verbatim by downstream edit tools.

## PRE-EXECUTION STEPS
Before reading the file, please follow these steps:
1. Path Verification:
   - `file_path` *MUST* be an absolute path. Relative paths are rejected, since the working directory is not persistent across calls.
   - If unsure whether the file exists or whether the path is a directory, use `ShellCommandTool` with `ls` first to verify.

## USAGE
  - `file_path` argument is required and must be an absolute path to an existing regular file.
  - `offset` argument is optional. When specified, it is the 1-indexed line number to start reading from. Defaults to `1`.
  - `limit` argument is optional. When specified, it is the maximum number of lines to read starting from `offset`. Defaults to <field>_default_limit</field> and is capped at <field>_default_limit</field>.
  - The file is read as UTF-8 with replacement on decoding errors, so binary noise does not crash the tool but may render as replacement characters.
  - Each returned line is prefixed with its 1-indexed line number, right-padded to 6 characters, followed by a tab, then the line content. This matches the output of `cat -n`.
  - Any single line longer than <field>_max_line_chars</field> characters is truncated and a `… [line truncated]` marker is appended to it. The `truncated_lines` field in the returned dict counts how many lines were truncated this way.

## **VERY IMPORTANT**
- The line-number prefix is purely an output formatting artefact. When constructing the `old_string` argument for a future edit tool, *NEVER* include the `<lineno><TAB>` prefix; use only the actual line content that appears after the tab, preserving its exact indentation.
- This tool is for text files only. Images, PDFs and Jupyter notebooks are not supported in this iteration.
- This tool *MUST* be preferred over `ShellCommandTool` with `cat`, `head` or `tail`, since shell commands do not provide line numbers and have no per-line size cap.

## RETURNS
The tool returns a dict with the following keys:
  - `file_path` (str): the canonical absolute path that was read.
  - `content` (str): the requested slice of the file formatted as `<lineno><TAB><line>` per line, joined by newlines. Empty when the file is empty or when `offset` is past the last line.
  - `start_line` (int): the 1-indexed line number of the first returned line, or `0` when no lines were returned.
  - `end_line` (int): the 1-indexed line number of the last returned line, or `0` when no lines were returned.
  - `total_lines` (int): the total number of lines in the file.
  - `truncated_lines` (int): the number of returned lines that exceeded <field>_max_line_chars</field> characters and were truncated.
