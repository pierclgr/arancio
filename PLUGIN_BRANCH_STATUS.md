# Plugin branch status

Branch: `feature/plugins`. Last commit: `04fc2f0`.
This file says what the branch adds and where the work stopped.

## What the branch adds (committed)

- **Hooks**: a `Hook` enum with 13 dispatch points and a `HookManager`.
  The agent loop, `BaseTool.call` and `PermissionManager.validate` fire them.
- **Plugins**: a folder in `~/.arancio/plugins/<name>/` with `manifest.yml`
  and `module.py`. `PluginLoader`, `PluginManager` and
  `PluginManifestValidator` find, load and register them. One `Plugin` base
  class.
- **`@hook`**: binds a plugin method to a hook. Stack it for many hooks. If a
  method raises, only that method is disabled.
- **`@tool`**: turns a plugin method into a tool. The category is `READ`,
  `WRITE`, `WEB`, `EXECUTE`, `PLUGIN` (default) or `CUSTOM`. `CUSTOM` makes
  a category `PLUGIN:<ClassName>`. `PluginTool` runs the method.
- **`PermissionCategory`**: now a registry class, not an `Enum`, so plugins
  can add tools and categories.
- **Commands**: `joins_arguments` lets `/rename` and `/resume` take the whole
  text.

## Not committed yet (14 files, 430 tests pass, lint clean)

1. `BaseTool._load_prompt`: `PluginTool` now calls `BaseTool.__init__`. It
   only overrides how the description and schema are loaded.
2. **Shared context dict**: every hook handler gets one argument,
   `HookManager.context`. `run()` writes the event data into it first.
   Event keys stay until a later dispatch overwrites them. Plugin hook
   methods are `method(self, context)`.
3. **Self-registration**: core fills the permanent keys itself. `HookManager`
   adds `hook_manager`, `ToolManager` adds `summary_client`,
   `PermissionManager` adds `permission_manager`, `Agent` adds `agent` and
   `client`. The UI does nothing.
4. **Read-back**: core reads `context["system_prompt"]` after
   `system_prompt_build`, and `context["result"]` after `after_tool_call`.
   A plugin can replace both.
5. `AGENTS.md` and the tests are updated for all of this.

## Where the chat stopped

- Next step: commit and push, merge into `main`, then start a new branch in
  a worktree under `.worktrees/`.
- `.worktrees/` is not in `.gitignore` yet. Add it first.
- A new worktree has no `.venv`. Run `uv sync` there. The git hooks use
  `.venv/bin/ruff`, and the main `.venv` points at the main `src/`.

## Open work (ideas, nothing started)

- Plugin methods can still not return values. Only the two read-back keys
  work. Maybe read back more keys: `arguments`, `decision`, `request`.
- Hooks cannot block a call. `before_tool_call` and
  `before_permission_check` cannot stop it.
- `@tool` methods do not receive the context. `PluginTool` holds
  `_hook_manager`, so it could pass `context` to the method.
- A plugin tool that returns a `dict` probably fails. `PluginTool` uses a
  result parser whose `_render` is abstract. Read in the code, not tested.
- Core features that could become plugins: `SearchWebTool` (fits now),
  `FetchWebTool` (needs the summary client, now in the context), the date
  and working directory in the system prompt (a `system_prompt_build`
  plugin can do it now).
