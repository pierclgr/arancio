Executes a given shell command in a non-persistent `/bin/sh` session with optional timeout and working directory, ensuring proper handling and capturing of output.

## PRE-EXECUTION STEPS
Before executing the command, please follow these steps:
1. Directory Verification:
   - If the command creates new directories or files, first run `ls` to verify the parent directory exists and is the 
     correct location (e.g. for command `mkdir dir/folder`, first run `ls dir` to check that `dir` exists and is the 
     intended parent directory)

2. Command Execution:
   - It is mandatory to quote file paths containing spaces, use double quotes to wrap the path (e.g., `rm "dir path  
   with spaces/file.txt"`)
   - Correct examples:
     - `ls "/path/with spaces.txt"` (correct)
     - `rm "/path/with spaces/script.py"` (correct)
   - Incorrect examples:
     - `ls /path/with spaces.txt` (incorrect - will fail)
     - `run /path/with spaces/script.py` (incorrect - will fail)

## USAGE
  - `command` argument is required.
  - `timeout` argument is optional; if specified, it must be in seconds (capped to <field>_max_timeout</field> seconds).
    If not specified, the command uses a default timeout of <field>_default_timeout</field> seconds. On timeout, `exit_code` is set 
    to `-1`, `timed_out` is `true`, and any partial output captured before the timeout is preserved in `stdout` and 
    `stderr`. 
  - `cwd` argument is optional. If specified, it must be an absolute path in which the command will be executed. If not 
    specified, the command runs in the current process working directory (namely, the equivalent of `cd .`).
  - Write a clear, concise and short of what this command does (~10 words); this helps you better understanding 
    the 
    command and its purpose.
  - If either `stdout` or `stderr` exceeds <field>_output_limit</field> characters, that stream will be truncated and a `… [N chars 
  truncated]` marker will be appended to it. The `truncated` flag in the returned dict is set to `true` whenever 
    truncation occurs on either stream.
  - When executing multiple commands, use the `;` or `&&` operator to separate them on a single line. NEVER use 
   newlines to separate commands, newlines are only allowed inside quoted strings.
  - Prefer absolute paths in commands. To run a command in a different directory, use the `cwd` argument rather than 
    `cd <directory> && <command>`, since the shell state does not persist across calls.

## **VERY IMPORTANT**
- This shell session is *NOT* persistent across calls. Each invocation starts a fresh `/bin/sh -c` subprocess, so the 
  working directory, environment variables, exported functions, and shell aliases set in one call do NOT carry over 
  to the next. Use the `cwd` parameter to set the working directory rather than relying on a `cd` from a previous call.
- You *MUST* avoid using search commands like `find` and `grep`. Use instead the dedicated `GrepTool` and `GlobTool` 
  for searching, unless explicitly requested by the user.
- You *MUST* avoid read commands like `cat`, `head`, and `tail`, and use the dedicated `ReadFileTool` to read files, 
  unless explicitly requested by the user.
- You *MUST NOT* run interactive commands (e.g., `vim`, `less`, `nano`, `ssh` with password prompts, REPLs without 
  scripted input). They will hang the tool until the configured `timeout` expires.

## RETURNS
The tool returns a dict with the following keys:
  - `stdout` (str): the captured standard output of the command, truncated to <field>_output_limit</field> characters when needed.
  - `stderr` (str): the captured standard error of the command, truncated to <field>_output_limit</field> characters when needed.
  - `exit_code` (int): the exit code of the process, or `-1` when the command timed out.
  - `timed_out` (bool): `true` when the command exceeded the configured `timeout`, otherwise `false`.
  - `truncated` (bool): `true` when either `stdout` or `stderr` was truncated to fit the <field>_output_limit</field>-character cap, otherwise `false`.
