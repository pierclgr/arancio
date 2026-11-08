Reads a page of source lines from a local text file, with line numbers.

## PRE-EXECUTION STEPS
1. Path Verification:
   - `file_path` *MUST* be absolute; relative paths are rejected.
   - If unsure the file exists, check with `ShellCommandTool` and `ls` first.

## USAGE
  - `file_path` argument is required and must be an absolute path to an existing regular file.
  - `offset`: 1-indexed start line (default `1`; past EOF, reads from `1`).
  - `limit`: lines to read, at most <field>_default_limit</field>.
  - `raw_markdown`: defaults to `false`. Set to `true` to read raw `.md`
    source with source line numbers. When `false`, dynamic tags expand and
    the expanded page is numbered from `1`. Ignored for other file types.
  - Lines over <field>_max_line_chars</field> chars end in
    `… [line truncated]`.

## **VERY IMPORTANT**
- *NEVER* copy the `<lineno><TAB>` prefix into an edit's `old_string`.
- Text files only: no images, PDFs or notebooks.
- Prefer this tool over `cat`, `head` or `tail`.

## RETURNS
A dict with:
- `file_path` (str)
- `content` (str)
- `start_line` (int)
- `end_line` (int)
- `total_lines` (int)
- `truncated_lines` (int)

Expanded Markdown adds:
- `source_start_line` (int)
- `source_end_line` (int)
- `source_total_lines` (int), raw file bounds
