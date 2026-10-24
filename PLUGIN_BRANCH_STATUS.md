# Plugin branch status

Branch: `feature/plugins`.
The plugin foundation was merged into `main` at `1f18570`. This file records
the implemented behavior and the later changes, verified with 445 tests.

## Implemented behavior

- **Hooks**: a `Hook` enum with 15 dispatch points and a `HookManager`.
  The app lifecycle, agent loop, `BaseTool.call` and `PermissionManager.validate`
  fire them.
- **Plugins**: a folder in `~/.arancio/plugins/<name>/` with `manifest.yml`
  and `plugin.py`. `PluginLoader`, `PluginManager` and
  `PluginManifestValidator` find, load and register them. One `BasePlugin` base
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

## Shared context update (committed in `e1599fd`)

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

## Earlier handoff notes

- The original commit/push/merge step was completed in `e1599fd` and `1f18570`.
- `.worktrees/` is not in `.gitignore` yet. Add it first.
- A new worktree has no `.venv`. Run `uv sync` there. The git hooks use
  `.venv/bin/ruff`, and the main `.venv` points at the main `src/`.

## Open work (ideas, nothing started)

- Only `system_prompt` and `result` are read back after hooks. More replacement
  keys remain an idea: `arguments`, `decision`, `request`.
- Hooks cannot block a call. `before_tool_call` and
  `before_permission_check` cannot stop it.
- `@tool` methods do not receive the context. `PluginTool` holds
  `_hook_manager`, so it could pass `context` to the method.
- Core features that could become plugins: `SearchWebTool` (fits now),
  `FetchWebTool` (needs the summary client, now in the context), the date
  and working directory in the system prompt (a `system_prompt_build`
  plugin can do it now).

## Dictionary tool result fix

- Goal: a plugin tool can return a JSON-compatible dictionary, including an
  empty one, without ending the agent run.
- Design: give `PluginTool` a concrete `PluginToolResultParser`. Keep the raw
  dictionary in `content` and render JSON in `display_text`.
- Verification: load real plugins returning empty and nested dictionaries;
  check the result, the after-tool hook and the next model turn. Run the full
  test suite, lint and docstring checks.
- Status: implemented. Both regression cases failed before the
  fix and now pass. All 432 tests pass; lint, docstring and formatting checks
  pass.

## Hook messages in the UI

- Goal: the HelloWold agent-start hook displays `Hello World` in the chat log.
- Design: forward hook return values from `PluginManager._run_hook` to
  `HookManager`, which already collects messages. The greeting hook returns
  an `AssistantMessage` with `in_history=False` instead of writing to stdout.
- Verification: load a real plugin, submit a prompt in the headless Textual
  app, and check that the greeting appears once and survives session restore
  without entering model history. Keep hook exception isolation unchanged.
- Status: implemented. The UI regression failed before the fix
  and now passes. All 433 tests pass; lint, docstring and formatting checks pass.

## Plugin entry filename

- Goal: load each plugin from `plugin.py` instead of `module.py`.
- Design: change `PLUGIN_MODULE_FILENAME` and derive the import name from the
  file stem, producing `arancio_plugins.<folder>.plugin`. Require `plugin.py`.
- Verification: check the loaded filename and import name, reject a folder
  containing only the old entry file, and preserve relative imports. Run the
  full suite and checks before renaming HelloWold's installed entry file.
- Status: implemented. All 435 tests pass, including the entry
  filename, import name and relative import checks. Lint, docstring and
  formatting checks pass.
- Installed example: `~/.arancio/plugins/HelloWold/plugin.py` has been renamed.

## Application lifecycle hooks

- Goal: add `STARTUP` once when the application opens, and `SHUTDOWN` once
  before the application finishes closing.
- Design: dispatch from Textual's app mount and unmount events with the same
  shared hook context. Display returned startup messages in the chat log.
  Shutdown is for cleanup; its returned messages are not displayed because
  the UI has already been unmounted. Neither event creates a session log.
- Keep `AGENT_START` and `AGENT_END` tied to user prompts. Switch HelloWold's
  greeting to `STARTUP`, preserving its manifest's enabled setting.
- Verification: open the headless app without a prompt; check the greeting
  and shared context. Check `/exit`, `/quit`, Ctrl+C and direct exit, plus
  plugin failure isolation. Run the suite, lint, docstring and format checks.
- Status: implemented. All 441 tests pass, including startup
  rendering, all four exit paths, shared context and plugin failure isolation.
  Lint, docstring and formatting checks pass.

## Plugin tool names

- Goal: expose plugin tools as `<plugin_name>-<tool_name>`.
- Prefix: the manifest's `name`, defaulting to the folder name.
- Suffix: the method name, or the decorator's `name=` override.
- Design: give each generated tool class the full name, keeping registration,
  permission lookup, hooks, agent execution and provider schemas consistent.
  Built-in tool names stay unchanged; no provider-name translation is needed.
- Verification: test default and explicit plugin names, decorator overrides,
  two plugins sharing a method name, permission routing and provider payloads.
  Run the full suite, lint, docstring and formatting checks.
- Status: implemented. All 445 tests pass, including default
  names, manifest prefixes, decorator overrides and two plugins sharing a
  method name. Lint, docstring and formatting checks pass. The installed
  example exposes `HelloWorld-HelloWorldTool`.
