Creates a new file or overwrites an existing one on the local filesystem with the provided UTF-8 content. Overwrites of existing files are gated by a read-first guard enforced in code, not just by description.

## PRE-EXECUTION STEPS
Before writing the file, please follow these steps:
1. Path Verification:
   - `file_path` *MUST* be an absolute path. Relative paths are rejected, since the working directory is not persistent across calls.
   - The parent directory *MUST* already exist; this tool does not create intermediate directories. If unsure, use `BashCommandTool` with `mkdir -p` first.
2. Read-First Verification (overwriting existing files):
   - If `file_path` points to a file that already exists, you *MUST* have called `ReadFileTool` on the same path in this session before invoking `WriteFileTool`.
   - If the file was modified on disk after your last `ReadFileTool` call (for example by another process or by a `BashCommandTool` invocation), you *MUST* re-read it before overwriting; the tool rejects writes whose recorded read mtime has drifted.
   - Creating a brand-new file at a path that does not yet exist does *not* require a prior read.

## USAGE
  - `file_path` argument is required and must be an absolute path.
  - `content` argument is required. It is written verbatim as UTF-8; no implicit trailing newline is appended. To produce a POSIX-style file ending in a newline, include `\n` at the end of `content`.
  - Prefer `EditFileTool` over `WriteFileTool` for incremental modifications to an existing file, since `EditFileTool` ships only the diff. Use `WriteFileTool` for new files or full rewrites.
  - This tool is free to create documentation files (`*.md`, `README*`) at the model's own initiative. The `codo` agent intentionally has no "no auto-create docs" rule because documentation generation is a core part of this agent's behavior.

## **VERY IMPORTANT**
- *NEVER* attempt to bypass the read-first guard by invoking `BashCommandTool` (e.g. `echo ... > file`, `tee`, `sed -i`). Such bypasses defeat the safety mechanism that prevents blind clobbers of files the model has not seen.
- Successful writes update the session's read record automatically with the new file mtime, so an immediately-subsequent `WriteFileTool` call on the same path does not require an intervening `ReadFileTool` call.

## RETURNS
The tool returns a dict with the following keys:
  - `file_path` (str): the canonical absolute path that was written.
  - `bytes_written` (int): the number of UTF-8 bytes written to disk.
  - `action` (str): `"created"` when the file did not exist before the call, `"overwritten"` when it did.
  - `total_lines` (int): the number of lines in `content`, computed via `str.splitlines()`.
