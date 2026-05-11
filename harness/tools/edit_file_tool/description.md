Performs an exact-substring replacement in an existing UTF-8 file. The tool finds the verbatim `old_string` in the file's content and replaces it with `new_string`, refusing to act when the snippet is ambiguous or when the file has not been read this session.

## PRE-EXECUTION STEPS
Before editing the file, please follow these steps:
1. Path Verification:
   - `file_path` *MUST* be an absolute path. Relative paths are rejected, since the working directory is not persistent across calls.
   - The file *MUST* already exist. Use `WriteFileTool` to create a new file from scratch instead.
2. Read-First Verification:
   - You *MUST* have called `ReadFileTool` on the same path in this session before invoking `EditFileTool`.
   - If the file was modified on disk after your last `ReadFileTool` call (for example by another process or by a `BashCommandTool` invocation), you *MUST* re-read it before editing; the tool rejects edits whose recorded read mtime has drifted.
3. Snippet Construction:
   - Build `old_string` from the content shown by `ReadFileTool` *after* the `<lineno><TAB>` prefix. *NEVER* include the line-number prefix or tab character in `old_string` — only the actual file content.
   - Preserve indentation, whitespace and line endings exactly as they appear in the file.

## USAGE
  - `file_path` argument is required and must be an absolute path to an existing regular file.
  - `old_string` argument is required and must be non-empty. The match is strictly byte-exact: no whitespace normalization, no regex, no fuzzy matching.
  - `new_string` argument is required but may be empty; an empty `new_string` deletes the matched snippet from the file.
  - `replace_all` argument is optional, defaults to `false`. When `false`, the tool requires `old_string` to appear exactly once in the file and aborts otherwise. When `true`, every occurrence is replaced.
  - The tool refuses edits where `old_string == new_string` (nothing to do).
  - Prefer `EditFileTool` over `WriteFileTool` for incremental modifications to an existing file, since `EditFileTool` ships only the diff. Use `WriteFileTool` for full rewrites or new files.

## **VERY IMPORTANT**
- This tool only edits valid UTF-8 text files. Binary content or files with non-UTF-8 byte sequences are rejected with a decode error.
- *NEVER* attempt to bypass the read-first guard by invoking `BashCommandTool` (e.g. `sed -i`, `tee`, `awk -i inplace`). Such bypasses defeat the safety mechanism that prevents blind clobbers of files the model has not seen.
- Successful edits update the session's read record automatically with the new file mtime, so an immediately-subsequent `EditFileTool` or `WriteFileTool` call on the same path does not require an intervening `ReadFileTool` call.
- When `old_string` is not unique in the file, *DO NOT* simply enable `replace_all` to silence the error: first decide whether all occurrences should change. If not, expand `old_string` with surrounding context until it matches exactly once.

## RETURNS
The tool returns a dict with the following keys:
  - `file_path` (str): the canonical absolute path that was edited.
  - `replacements` (int): the number of occurrences replaced (`1` when `replace_all=false`, possibly more when `replace_all=true`).
  - `bytes_before` (int): the file's UTF-8 byte size before the edit.
  - `bytes_after` (int): the file's UTF-8 byte size after the edit.
  - `action` (str): always `"edited"`.
