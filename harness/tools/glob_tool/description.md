Searches filenames in a directory using ripgrep (`rg --files`) and returns paths sorted by modification time descending. By default, respects `.gitignore`/`.ignore` rules and skips hidden files and directories.

## PRE-EXECUTION STEPS
Before searching, please follow these steps:
1. Pattern Validation:
   - The `pattern` argument is a glob, not a regex. Supported syntax: `**` (recursive), `*` (any chars in one segment), `?` (single char), `[abc]` (charclass), `{a,b}` (brace expansion).
   - Examples: `**/*.py`, `src/**/*.{ts,tsx}`, `*.md`, `**/test_*.py`.
   - Prefer this tool over `ShellCommandTool` with `find`, since it is faster and sorts by mtime.

2. Path Verification:
   - When `path` is specified, it must be an absolute path to an existing directory. If unsure whether the path exists, use `ShellCommandTool` with `ls` first.

## USAGE
  - `pattern` argument is required. Glob pattern matched against filenames (e.g. `**/*.py`, `src/**/*.{ts,tsx}`).
  - `path` argument is optional. Absolute path to the directory to search. Defaults to the current working directory (`.`).
  - `head_limit` argument is optional. Maximum number of paths to return. Defaults to <field>_default_head_limit</field>.

## **VERY IMPORTANT**
- This tool is for **filename**-based search only. For content search inside files, use `GrepTool`.
- For open-ended multi-iteration filename hunts, delegate to a subagent rather than calling GlobTool repeatedly.
- Prefer this tool over `ShellCommandTool` with `find`: `find` is slower and does not sort by mtime.
- When no files match, the tool returns an empty result — this is NOT an error. Do not retry with different patterns unless you have concrete reason to believe matching files exist.
- Results are sorted by modification time **descending** (most recently modified first), so recent edits surface at the top of the list.

## RETURNS
The tool returns a dict with the following keys:
  - `matches` (list[str]): list of absolute file path strings, sorted by mtime descending.
  - `total_matches` (int): total number of results before `head_limit` was applied.
  - `truncated` (bool): `true` when the output was capped by `head_limit`.
  - `timed_out` (bool): `true` when the search exceeded the timeout.
  - `exit_code` (int): the ripgrep process exit code (0 = files found, 1 = no files matched, 2 = error).
  - `search_path` (str): the resolved absolute search path.
  - `pattern` (str): the pattern that was searched.
