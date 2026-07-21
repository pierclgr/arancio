Searches files in a directory using ripgrep (`rg`) and returns structured results. Respects `.gitignore` and uses full regex syntax (not POSIX).

## PRE-EXECUTION STEPS
Before searching, please follow these steps:
1. Pattern Validation:
   - The `pattern` argument uses ripgrep's regex syntax, not POSIX. Literal braces, dots and other regex metacharacters must be escaped (e.g. `interface\{\}` to match the literal string `interface{}`).
   - Prefer this tool over `ShellCommandTool` with `grep` or `find`, since it gives `.gitignore`-aware results and supports glob/type filters.

2. Path Verification:
   - When `path` is specified, it must be an absolute path to an existing file or directory. If unsure whether the path exists, use `ShellCommandTool` with `ls` first.

## USAGE
  - `pattern` argument is required. Full regex syntax: `log.*Error`, `def\s+\w+`, `TODO|FIXME`.
  - `path` argument is optional. Absolute path to a file or directory to search. Defaults to the current working directory (`.`).
  - `glob` argument is optional. Filename pattern filter (e.g. `*.py`, `*.{js,ts}`, `!*.test.js`). Matches ripgrep's `-g` flag.
  - `file_type` argument is optional. Language type using rg's built-in type map (e.g. `py`, `js`, `rust`, `markdown`). Matches ripgrep's `-t` flag.
  - `output_mode` argument is optional. One of:
    - `files_with_matches` (default): returns only matching file paths, one per line.
    - `content`: returns matching lines as `file:lineno:text`. Context lines (when `-A`/`-B`/`-C` are set) are returned as `file-lineno-text`.
    - `count`: returns match counts per file as `file:N`.
  - `i` argument is optional. When `true`, search is case insensitive. Default `false`.
  - `n` argument is optional. When `true` (default), line numbers are shown in `content` mode.
  - `A` argument is optional. Number of lines to show after each match.
  - `B` argument is optional. Number of lines to show before each match.
  - `C` argument is optional. Number of lines to show before and after each match.
  - `head_limit` argument is optional. Maximum number of matches or lines to return. Defaults to <field>_default_head_limit</field>.
  - `multiline` argument is optional. When `true`, enables multi-line matching across line boundaries (`.` matches `\n`). Default `false`.

## **VERY IMPORTANT**
- This tool is for content search only. For filename-based search, use `GlobTool`.
- For open-ended multi-iteration searches, delegate to a subagent rather than calling GrepTool repeatedly.
- Literal braces, dots and other regex metacharacters in patterns must be escaped: `interface\{\}` not `interface{}`. The model must escape them before passing to this tool.
- When no matches are found, the tool returns an empty result — this is NOT an error. Do not retry with different patterns unless you have a concrete reason to believe the content exists.

## RETURNS
The tool returns a dict with the following keys:
  - `matches` (list): the matched results. Format depends on `output_mode`:
    - `files_with_matches`: list of absolute file path strings.
    - `content`: list of dicts with keys `file` (str), `line` (int), `content` (str), `is_context` (bool).
    - `count`: list of dicts with keys `file` (str), `count` (int).
  - `total_matches` (int): total number of results before `head_limit` was applied.
  - `truncated` (bool): `true` when the output was capped by `head_limit`.
  - `timed_out` (bool): `true` when the search exceeded the timeout.
  - `exit_code` (int): the ripgrep process exit code (0 = matches found, 1 = no matches, 2 = error).
