Run a command in a non-interactive PowerShell session and return its
stdout, stderr and exit code.

- Uses powershell.exe when available, then falls back to pwsh.
- Use absolute paths. Quote paths containing spaces with single quotes.
- Do NOT run interactive commands — they will hang or fail.
- Each stream is truncated at <field>_output_limit</field> characters; a marker
`… [N chars truncated]` is appended when truncation occurs.
- Default timeout is <field>_default_timeout</field> seconds, maximum <field>_max_timeout</field> seconds.
- PowerShell state (cwd, env vars, functions, aliases) is NOT persisted
across calls — pass `cwd` explicitly when the command depends on it.
