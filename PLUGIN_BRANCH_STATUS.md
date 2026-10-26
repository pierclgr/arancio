# Plugin branch status

Branch: `feature/plugins`.
The plugin foundation was merged into `main` at `1f18570`. This file records
the implemented behavior and the later changes, verified with 477 tests.

## Implemented behavior

- **Hooks**: a `Hook` enum with 15 dispatch points and a `HookManager`.
  The app lifecycle, agent loop, `BaseTool.call` and `PermissionManager.validate`
  fire them.
- **Plugins**: plain `@plugin(...)` classes in `~/.arancio/plugins/<folder>/plugin.py`.
  Each decorated class is independent; one file can hold several. Metadata is in
  the decorator, with no YAML manifest, base class or injected runtime attributes.
  Decorators fill a shared `list[Plugin]`; `PluginManager` imports entry files and
  attaches bound methods. Wrong metadata skips only its class. Empty plugins load
  with a warning. Decorated helper classes also register.
- **`@hook`**: binds a plugin method to a hook. Stack it for many hooks. If a
  method raises, only that method is disabled.
- **`@tool`**: turns a plugin method into a tool. The category is `READ`,
  `WRITE`, `WEB`, `EXECUTE`, `PLUGIN` (default) or `CUSTOM`. `CUSTOM` makes
  a category `PLUGIN:<plugin_name>`. `PluginTool` runs the method.
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
- Plugin permissions should change. Either define a new permission category
  per plugin, or let a tool or plugin attach to an existing permission
  category through its decorator (`@tool` / `@plugin`).

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

## Plain-class plugin decorator

- Status: implemented. The earlier sections record historical changes; current
  discovery and metadata behavior is described here and under Implemented behavior.
- `@plugin(...)` returns the original class and declares name, version,
  description, author and enabled. Name defaults to the class name; other
  defaults and warning-based field validation are preserved.
- All three decorators live in `core/plugins/decorators.py`. Hook and tool
  discovery are the static `PluginLoader._find_hook_methods` and
  `PluginLoader._find_tool_specs` methods; tool specs and execution stay in
  `tool.py`.
- Discover immediate folders containing `plugin.py`; missing entries are skipped.
  Every directly decorated local class is an independent plugin. Imported classes,
  undecorated subclasses and duplicate class aliases do not create extra plugins.
- The loader checks enabled definitions across all folders before construction.
  All plugins with conflicting names are skipped, case-insensitively. Disabled
  classes are excluded from conflicts. Importing their file still executes code.
- Each instance is created once with no arguments. Constructor failures skip only
  that class. `Plugin` holds validated fields, instance and source path internally;
  the manager registers methods on that original instance.
- No directory, manifest, metadata or context is injected. Plugin code handles
  local resources itself. The old BasePlugin and YAML metadata reader are removed.
- Tool names use the decorator name (default: class name), followed by a hyphen
  and the method name or tool override. CUSTOM permission groups use that plugin
  name too. Old renamed grants follow existing settings warnings; new groups use ASK.
- Tests cover independent state, duplicate rejection before construction and
  registration, constructor isolation, disabled imports, metadata warnings,
  class selection, helper imports, custom permissions and shared-function failure
  isolation. Existing lifecycle, UI, history and tool-result regressions are retained.
- Verification: all 465 tests, lint, formatting and docstring checks pass.
  The installed HelloWold example uses the shared decorators module; its
  startup greeting and HelloWorld-HelloWorldTool work without a manifest.

## Metadata outside plugin objects

- Goal: plugin code has no direct metadata or `Plugin` record reference on its
  class or instance.
- Design: `@plugin(...)` records raw fields in an internal weak-key registry,
  keyed by the original class. The loader uses that registry for discovery and
  validation. Remove the metadata class attribute and its constant; keep
  validated metadata in `Plugin`.
- Verification: check that decoration leaves class attributes unchanged and
  that plugin constructors, hooks and tools have no metadata attributes.
  Keep discovery, validation, names and enabled behavior unchanged.
- Status: implemented. The three regression checks failed before the change
  and now pass. All 466 tests, lint, formatting and docstring checks pass.

## Nested plugin declarations

- Goal: keep plugin metadata, hook bindings and tool specs off plugin classes,
  instances and functions.
- Design: use one class-keyed registry with `metadata` and a `methods` mapping.
  Each method entry holds hooks and an optional tool spec. Method decorators
  collect fields in an internal weak-key function map before the class exists;
  `@plugin` groups them under the completed class without changing it. Keep
  pending entries available for functions shared by several plugin classes.
- Verification: check unchanged function attributes, both decorator orders,
  nested registry contents, a method named `metadata`, shared functions and
  access checks inside real plugin constructors, hooks and tools. Ordinary
  undecorated callable fields must not block class collection.
- Status: implemented. All 471 tests, lint, formatting and docstring checks pass.

## Metadata dictionary on Plugin

- Goal: remove the `PluginMetadata` record and keep validated metadata directly
  in `Plugin.metadata` as a dictionary.
- Design: the validator returns its normalized dictionary; the loader stores it
  without wrapping it. Update field access in loading, registration and tool
  construction. Keep validation, defaults, warnings and registry isolation unchanged.
- Verification: check dictionary results and loaded metadata, then run the full
  suite, lint, docstring and formatting checks.
- Status: implemented. All 471 tests, lint, formatting and docstring checks pass.

## Dedicated plugin registry module

- Goal: keep declaration storage out of the metadata validator module.
- Design: move `_PLUGIN_REGISTRY` and `_PENDING_METHODS` unchanged to
  `core/plugins/registry.py`. Update all consumers to import that shared state
  from its new owner; `validator.py` contains only metadata validation.
- Verification: check registry ownership and run the full test suite, lint,
  docstring and formatting checks without changing runtime behavior.
- Status: implemented. All 472 tests, lint, formatting and docstring checks pass.

## Plugin validator naming

- Goal: use `core/plugins/validator.py:PluginValidator` and remove the
  single-use `_invalid_field_text` helper.
- Design: rename the module and class, update imports, and build the same
  warning text directly in `validate`. Keep metadata and warnings unchanged.
- Verification: check the new module and exact invalid-field warning text,
  then run the full test suite and quality checks.
- Status: implemented. All 472 tests, lint, formatting and docstring checks pass.

## Plugin registry builder

- Goal: put registry construction in a `Builder` subclass and keep pending
  method declarations on that builder rather than in registry storage.
- Design: add `core/builders/plugin_registry.py:PluginRegistryBuilder`, with
  class-level `_PENDING_METHODS`, `add_hook`, `add_tool` and `build` classmethods.
  Decorators delegate to it; `build` groups one class's declarations, writes the
  entry and returns the shared `_PLUGIN_REGISTRY` kept in `plugins/registry.py`.
- Verification: check builder inheritance and direct construction, state
  ownership, unchanged plugin objects, stacked decorators and shared functions.
  Run the full test suite and quality checks.
- Status: implemented. All 473 tests, lint and docstring checks pass. Changed
  Python files pass formatting; the full format check flags pre-existing
  indentation in `core/plugins/loader.py`, left unchanged.

## PluginRegistry class

- Goal: replace the registry builder and separate storage with one
  `core/plugins/registry.py:PluginRegistry` class.
- Design: move both dictionaries and the existing `build`, `add_hook` and
  `add_tool` classmethods onto `PluginRegistry`, without `Builder` inheritance.
  Update all consumers and delete `core/builders/plugin_registry.py`.
- Verification: check class ownership and unchanged declaration construction,
  then run the full suite and quality checks. Preserve unrelated loader edits.
- Status: implemented. The builder module is removed. All 473 tests, lint and
  docstring checks pass; the full format check still flags pre-existing loader
  indentation, left unchanged.

## Hook discovery on PluginLoader

- Goal: remove `core/plugins/hook.py` and put its discovery function on
  `PluginLoader` as a static `_find_hook_methods` method.
- Design: move the function unchanged and update the loader, manager and tests
  to use that method, without changing registry lookup or method order.
- Verification: check static-method ownership and stacked-hook discovery,
  then run the test suite and quality checks. Leave unrelated loader edits alone.
- Status: moved and the old module deleted. Both focused discovery tests, lint
  and docstring checks pass. The full suite stops after 199 passing tests because
  the existing `PluginLoader.lload` name does not match the manager's `load` call.
  Existing loader indentation also still fails the full formatting check.

## Plugin record naming

- Goal: rename the internal `LoadedPlugin` record to `Plugin` and move
  `core/plugins/types.py` to `core/plugins/base.py`.
- Design: change names and imports only. Preserve the record's frozen fields,
  metadata dictionary, original instance and source path; do not restore the
  old plugin-author base class.
- Verification: construct the record through its new module and check that
  it keeps the original values. Run the test suite and quality checks.
- Status: renamed to `Plugin` in `base.py`, with callers and imports updated.
  All six focused tests, lint and docstring checks pass. The full suite stops
  after 199 passing tests on the existing `PluginLoader.lload` naming issue;
  the existing loader formatting issue remains unchanged. No files staged.

## Tool discovery on PluginLoader

- Goal: move tool discovery from `core/plugins/tool.py` to the static
  `PluginLoader._find_tool_specs` method, alongside hook discovery.
- Design: move the function without changing registry lookup, method order
  or directly-defined-method filtering. Update loader, manager and tests.
- Verification: check static-method ownership, both decorator orders,
  discovery order, inherited-method exclusion and unregistered classes.
  Run the suite and quality checks; preserve unrelated loader edits.
- Status: moved, with loader, manager, tests and documentation updated.
  All three focused tests and 379 non-plugin tests pass, as do lint and
  docstring checks. The full suite stops after 199 passing tests on the
  existing `PluginLoader.lload` naming issue. The existing loader formatting
  differences remain unchanged. No additional files staged.

## Tool construction on PluginTool

- Goal: replace the standalone `build_plugin_tool_class` function with the
  `PluginTool.build` classmethod.
- Design: keep the generated name, plugin instance, static function binding
  and spec unchanged. Update the manager to call the classmethod directly.
- Verification: check classmethod ownership, removal of the old function,
  registration, schema and successful calls on the original plugin instance.
  Run focused tests, the suite and quality checks without unrelated fixes.
- Status: moved to `PluginTool.build`, with the manager and documentation
  updated. All four focused tests and 379 non-plugin tests pass, plus lint,
  docstring and changed-file formatting checks. The full suite still stops
  after 199 passing tests on the existing `PluginLoader.lload` naming issue;
  existing loader formatting differences are untouched. No files staged.

## PluginRegistry instance state

- Goal: make `_PLUGIN_REGISTRY` and `_PENDING_METHODS` instance members,
  with fresh weak-key dictionaries for each `PluginRegistry()`.
- Design: initialize both stores in `__init__` and turn `build`, `add_hook`
  and `add_tool` into instance methods. Decorators and loader use one
  module-level `shared_registry`, matching the existing `shared_session`
  pattern. Keep declaration contents and plugin object isolation unchanged.
- Verification: prove two registries have independent declarations, pending
  hooks and tool specs; check shared-instance wiring and existing discovery
  and tool execution. Run tests and quality checks without unrelated fixes.
- Status: implemented. Both dictionaries and all three registry methods are
  instance-owned; decorators and loader use `shared_registry`. All nine
  focused tests and 379 non-plugin tests pass, plus lint and docstring checks.
  The full suite still stops after 199 passing tests on the existing `lload`
  typo, and the loader's existing formatting differences remain. These edits
  are unstaged, including changes to the previously staged `registry.py`.

## PluginRegistry member names

- Goal: rename `_PLUGIN_REGISTRY` to `_registry` and `_PENDING_METHODS` to
  `_pending_methods`.
- Design: change member names and references only. Keep instance ownership,
  weak-key storage and the shared registry unchanged.
- Verification: check the lowercase members, absence of the old names and
  unchanged registry isolation, discovery and tool execution. Run the tests
  and quality checks without changing unrelated loader issues.
- Status: renamed both members and updated consumers, tests and current
  documentation. No old names remain in source. All nine focused tests and
  379 non-plugin tests pass, plus lint and docstring checks. The full suite
  still stops after 199 passing tests on the existing `lload` typo, and the
  loader's existing formatting differences remain. Changes are unstaged.

## PluginLoader entry method name

- Goal: fix the `lload` typo so `PluginManager` can call `PluginLoader.load`.
- Design: rename the method to `load` without changing its body or formatting.
- Verification: use the existing failing plugin-loading test as the regression,
  then run the full suite, lint and docstring checks.
- Status: renamed to `load`. All 478 tests, lint and docstring checks pass.
  The full formatting check still flags the loader's existing formatting,
  left unchanged. The fix remains unstaged for the current loader review.

## Plugin package-root constant location

- Goal: move `PLUGIN_PACKAGE_ROOT` back to `core/plugins/constants.py`.
- Design: restore the constants module and import the constant in the loader,
  keeping its value and the synthetic import packages unchanged.
- Verification: import the constant from its module, check the loaded plugin
  namespace, and run the full suite and quality checks.
- Status: restored `constants.py` and updated the loader import. All 478
  tests, lint and docstring checks pass; existing loader formatting remains
  untouched. The restored constants file and loader edits are unstaged;
  the earlier staged deletion of `constants.py` remains pending review.

## Registry access and shared-instance ownership

- Goal: read declarations with `registry[plugin_cls]`, test membership with
  `plugin_cls in registry`, and create `shared_registry` in `decorators.py`.
- Design: add `__getitem__` and `__contains__` over the private weak-key
  storage. Return `self` from `build` rather than exposing that storage.
  Keep only the class in `registry.py`; decorators create the shared
  instance and the loader imports it from there. Do not add a runtime module.
- Verification: check lookup, missing-key `KeyError`, membership, `build`
  identity, instance isolation and shared-instance ownership. Keep missing
  classes returning no hook or tool declarations, and run the full suite
  and quality checks without unrelated formatting changes.
- Status: implemented. `registry.py` contains only the registry class;
  `decorators.py` owns the shared instance, and consumers use `[]` and `in`.
  `build` returns the registry object. All 479 tests pass, including all
  100 plugin tests, plus lint and docstring checks. The loader's existing
  formatting differences remain untouched. Changes to the previously staged
  registry and decorators files are left unstaged for review.

## Individual metadata fields on Plugin

- Goal: replace `Plugin.metadata` with separate `name`, `version`,
  `description`, `author` and `enabled` fields.
- Design: keep the record frozen and preserve `instance` and `source_path`.
  The validator still returns a dictionary; the loader unpacks it into
  the record. Read attributes directly in the manager and tool factory.
  Keep defaults in the validator, with no duplicate defaults on the record.
- Verification: check every field, frozen assignment, instance/path identity,
  metadata isolation and unchanged loading, tool names, permissions and errors.
  Run the full suite and quality checks without unrelated formatting changes.
- Status: implemented. The record exposes five typed metadata fields, with
  no metadata dictionary. Loading, registration, tool construction and tests
  use those fields directly. All 479 tests, lint and docstring checks pass;
  the loader's existing formatting differences are untouched.

## Shared registry name and location

- Goal: rename `shared_registry` to `plugin_registry` and create it in
  `core/plugins/registry.py` again.
- Design: decorators and loader import that one instance from `registry.py`.
  Keep instance-owned stores, `[]` lookup, membership and `build` returning self.
- Verification: check both consumers share `registry.plugin_registry`, the
  old name is absent, and loading and execution remain unchanged.
- Status: renamed and moved back to `registry.py`; decorators and loader
  share that instance. `[]` and `in` access are unchanged. All 479 tests,
  lint and docstring checks pass. Updates remain unstaged for review,
  including edits to previously staged files.

## Registry plugin registration method

- Goal: rename `PluginRegistry.build` to `add_plugin` and return no value.
- Design: keep declaration collection and storage unchanged. Remove the
  return statement, annotate the method with `None`, and update the decorator.
  Leave `PluginTool.build` and metadata validation unchanged.
- Verification: check that `add_plugin` returns `None`, the old method is
  absent, and stored declarations, instance isolation and loading still work.
  Run the full suite and quality checks without unrelated formatting changes.
- Status: renamed to `add_plugin(...) -> None`, with the return removed and
  decorator, tests and current documentation updated. All 479 tests, lint and
  docstring checks pass. `PluginTool.build` and validation are unchanged.
  Existing loader formatting is untouched; edits remain unstaged for review.

## Validate declarations when added

- Goal: make `PluginRegistry.add_plugin` validate metadata before storing a
  declaration, without loading the plugin.
- Design: each registry owns a `PluginValidator` instance. Registration
  derives the declaring file's diagnostic location, validates metadata, and
  stores the normalized dictionary, method declarations and warning messages.
  The loader reads those values and forwards warnings without validating again.
  Registration still returns `None` and never constructs plugin instances or
  registers runtime hooks, tools or permission categories.
- Verification: check normalized entries immediately after import, validator
  ownership, unchanged input objects, and no runtime effects before loading.
  Check warning delivery for enabled and disabled plugins, then run the suite
  and quality checks without unrelated formatting changes.
- Status: implemented. The registry owns its validator and stores only validated
  metadata, collected declarations and validation messages. The loader forwards
  those messages and no longer calls the validator. All 481 tests pass, including
  102 plugin tests, plus lint and docstring checks. Existing loader formatting
  remains untouched. Updates to previously staged files remain unstaged.

## Plugin objects as registry entries

- Goal: make every registry value a validated `Plugin` record rather than a
  metadata/methods/messages dictionary, while retaining class-keyed lookup.
- Design: construct the record in `add_plugin` after validation, with the
  source path, method declarations, warnings and `instance=None`. Make the
  record mutable as requested. After eligibility and duplicate checks, the
  loader creates the plugin instance and assigns it to that same record.
  Keep existing class and function objects unchanged.
- Verification: check record types, validated fields and no instance at
  registration; test method/warning preservation, isolated records and loaded
  object identity. Check editable fields and run the full suite and quality checks.
- Status: implemented. Registry lookup returns a mutable `Plugin` with validated
  fields, declarations, warnings and an initially absent instance. The loader
  attaches the runtime instance to that same object after eligibility checks.
  All 481 tests pass, including 102 plugin tests, plus lint and docstring checks.
  Existing loader formatting is untouched. Updates remain unstaged for review.

## Simple decorator registry and startup loading

- Goal: replace the loader/manager/validator/registry chain with one declaration
  list and one manager, using tests for the new contract rather than old internals.
- Design: decorators append `Plugin` records to `plugin_registry`. Records hold
  metadata defaults plus direct `hooks` and `tools` collections. Two weak-key
  pending maps retain method declarations until class decoration, including shared
  functions. `Plugin.__post_init__` enforces field types; invalid or unknown
  metadata reports one startup error and skips only that class. Empty declarations
  remain registered and warn, even when disabled.
- Loading: `PluginManager.registry` is the shared list. The manager imports sorted
  entry files, accepts all decorated classes (including helpers), checks enabled
  name conflicts before construction, binds methods once and attaches hooks/tools.
  Disabled, conflicting and constructor-failing records retain `instance=None`.
  Failed imports roll back partial registrations, diagnostics and cached plugin
  modules. Cached entries and constructed instances are reused on repeated loads.
- Removed: `PluginLoader`, `PluginValidator`, the `PluginRegistry` class, discovery
  methods, nested method dictionaries, stored validation messages, `_plugins` and
  the `plugins` property. Generated tools hold bound methods without a separate
  plugin instance or staticmethod wrapper.
- Verification: the new contract tests first failed against the old registry.
  All 477 tests pass, including 98 plugin tests. Full lint, docstring, formatting
  and diff-whitespace checks pass. Test fixtures reset registration state and
  synthetic module caches; metadata errors, empty warnings, helper imports,
  rollback/retry, bound methods, permissions and runtime failures are covered.
- Status: implemented in the existing worktree. Earlier sections describe past
  designs; this section and AGENTS.md describe the current plugin structure.

## Tool metadata stored directly on tool classes

- Goal: remove `PluginToolSpec` and store tool metadata directly, as with plugin
  metadata. Keep existing naming, permission routing, schema and result behavior.
- Design: `@tool` records raw fields in its pending map. `@plugin` creates a
  separate `PluginTool` subclass for each declaration, including shared methods.
  The class holds its tool name, category, description, raw schema and method.
  `Plugin.tools` stores those classes directly. During startup, the manager binds
  each method and registers the existing class in its permission category.
- Removed: `PluginToolSpec`, `_spec` and method/spec tuples. The normal ToolManager
  still creates tool instances only for granted categories, and preserves schema
  normalization and shared plugin state.
- Verification: updated declaration tests failed against the tuple-based design
  before implementation. All 98 plugin tests pass, including decorator order,
  shared functions, bound methods, category routing and provider descriptions and
  schemas. Full-suite and quality checks are recorded below.
- Status: all 477 tests pass. Full lint, docstring, formatting and diff-whitespace
  checks pass. Tests and current plugin documentation describe tool classes as the
  registry entries, with no separate metadata record.
