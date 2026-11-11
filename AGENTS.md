# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`arancio` ("another coding agent"): a Textual TUI coding agent running a
multi-turn loop against any LLM through the LiteLLM **Responses API**.
Python ≥ 3.11, managed with `uv`.

## Commands

All tools run from `.venv/` (`uv run <cmd>` also works).

- **Run**: `uv run arancio` (→ `arancio.ui.__main__:main`)
- **Sync deps**: `uv sync`; add one with `uv add <pkg>`
- **Tests**: `.venv/bin/python -m pytest tests/` (append `::test_name` for one)
- **Lint**: `.venv/bin/ruff check src/ tests/` (`E,F,I`)
- **Format**: `.venv/bin/ruff format src/ tests/`
- **Docstrings**: `.venv/bin/ruff check --select=D,DOC --preview src/ tests/`
- **Docstring wrap**: `.venv/bin/docformatter --in-place --recursive --wrap-summaries 88 --wrap-descriptions 88 src/ tests/`

Hooks in `hooks/` are active (`core.hooksPath`): `pre-commit` fixes, formats and
lints staged files; `pre-push` runs the docstring check and the test suite.

## Runtime directory: `~/.arancio/`

Paths are in `core/constants.py` and each component's `constants.py`.

- `harness/tools/<snake_tool_name>.md`: the tool description, read by
  `Tool.__init__` (raises if the file is missing). The input schema is not a
  file: it is derived from `_call` (see Tools).
- `harness/system_prompt.md`: seeded if absent, parsed once and cached by
  `SystemPromptBuilder`; `build()` appends the current date each turn. Put
  fresh runtime context in `build()`, never in a `<script>` tag.
- `settings.yml`: written by `SettingsManager` through Storage. Layout:
  `permissions: default: {read: ask, ...}`,
  `plugins: [{<name>: {enabled: bool, permission_level: ask|auto|null}}]`,
  `model: {provider, name, reasoning: {effort, summary}}` and
  `turn: {max, retry: {max, delay, multiplier}}`.
- `litellm/`: LiteLLM login data (moved/symlinked by Storage's generic ops).
- `sessions/<cwd-name+hash>/<y>/<m>/<d>/<uuid4hex>.jsonl`: one append-only log
  per session. No registry file; `SessionRegistry` rescans at launch.
- `plugins/<folder>/plugin.py`: placed by hand; nothing creates the directory.

The repo's `harness/` is only the **source** of those files; there is no copy
step. Storage's root anchors relative paths; absolute paths pass through.

## Architecture

Layered and provider-agnostic: a `base.py` abstract class plus provider files
(e.g. `litellm.py`). A new provider is a new `Client` subclass with its own
builder/parser trio. Every `__init__.py` under `core/` is empty; import from
the module (`core.hooks.base`, `core.hooks.manager`, ...).

### Wiring

`Application.build(ui_controller)` (`application/base.py`) creates
`StorageManager`, LiteLLM login paths, one shared `HookManager` and
`ToolSession`, the clients, `ToolManager`, then loads Settings, then plugins,
then `PermissionManager`, `Agent`, `SessionManager`, direct tools and
`Application`. **Settings load before plugins** (their `plugins` entries decide
which plugins load), and **plugins load before Permissions** so plugin
categories exist. Clients get `ui_controller.show_login` as login callback; they
get no HookManager. `ui/__main__.py` builds `TUIController` →
`Application.build` → `TUIApp(application)`, then attaches the app.

### Messages (`core/messages/`)

Each message has `content` (for the model), `display_text` (for the UI) and one
flag `in_history` (default true). `ErrorMessage` defaults to `False`;
`ToolErrorMessage` keeps `True` (MRO). `ChunkMessage` marks streaming deltas.
`WarningMessage` is a non-fatal `ErrorMessage` (`.warning` CSS). Persistence and
display are decided by their owners, not core.

### Agent (`core/agents/base.py`)

`__call__()` yields `Iterator[Message]` (no `run()`). It keeps a flat history
(optional copied `message_history` in the constructor, or `restore_history`).
Per turn: build request → stream (yield chunks, store finalized) → for each
`ToolCallMessage`: `PermissionManager.validate`, run, append result (+ optional
user note) → next turn. Stops on a text-only reply. `agent.turn` holds a frozen
`Turn(max, retry=Retry(max, delay, multiplier))`
(`core/agents/types.py`), the same object as `settings.turn` (defaults
in `DEFAULT_TURN`); change it by assigning a new one.
`turn.max="inf"` is unlimited; failures retry with exponential backoff
(`retry.delay`, grown by `retry.multiplier`) up to `retry.max`, except after
finalized messages were delivered. `agent.turn_index` is the current loop
index. The initial message is stored but **not yielded**. The agent saves nothing; consumers forward its stream as is.

### Clients (`core/clients/`, `core/builders/`, `core/parsers/`)

`Client` sets `_request_builder`, `_payload_builder`, `_response_parser`. Flow:
`build_request` → mutable `Request` → payload builder → `litellm.responses` →
`parse` / `parse_stream` → `Message`s. `LiteLLMClient` routes by `model_id`
prefix; credentials come from env vars. The static
`Client.get_model_id(provider, name)` builds a client's id (`provider/name` by
default, `None` when a part is unset); a client whose provider expects another
format overrides it. `_litellm_patches.apply()` patches pinned-LiteLLM bugs,
including the ChatGPT device-code login notice sent to the controller (UI-only,
never in history). `client.reasoning` and `request.reasoning` hold a frozen
`Reasoning(effort, summary)` (`core/clients/types.py`); change
it by assigning a new one. Core does not check the values, so a bad effort or
summary fails only when the request is sent. The payload builder sends it as
LiteLLM's `reasoning`. `core/clients/types.py` has the frozen
`Model(provider, name, reasoning)` with `check_configured()`, `to_dict()`
and `from_dict()`: the one model type that Settings, Sessions and Application
share, so Settings and Sessions never import each other.

The `litellm[mcp]` extra is required by the current Responses dispatcher even
for ordinary function tools; without it, model requests fail before reaching the
provider with `No module named 'mcp'`.

**Ollama caveat**: `reasoning={"effort": ...}` is always sent, but for
non-`gpt-oss` Ollama models LiteLLM turns it into `think: true|false`, so
low/medium/high are equal. Setting a reasoning summary there crashes LiteLLM
(`unhashable type: 'dict'`); keep `model.reasoning.summary` `null` for Ollama.

### Tools (`core/tools/`)

Subclasses implement `_call`; `call(tool_call)` returns `(result, hook_messages)`,
wrapping errors in `ToolErrorMessage` via `_result_parser`. Tool name = class
name. Descriptions are dynamic markdown (`<field>`, `<include>`, `<script>`).
The input schema is derived from `_call` by `InputSchemaBuilder`
(`core/builders/tool_input_schemas/`): type hints give types and required arguments,
`Annotated[..., Field(ge=, le=, min_length=)]` the constraints sent to the
model (each family's `constants.py` holds the shared defaults and bounds; the
tool still clamps/validates at runtime), and the docstring's parameter section
(Google `Args:`, NumPy `Parameters` or Sphinx `:param`, read by the parsers in
`core/parsers/docstrings/`) the per-parameter descriptions, so those lines are
**model-facing**. `Tool.__init__` closes it as
an object with `additionalProperties: false`. Variadic or untyped parameters and
unsupported types raise `ValueError`. The shared `ToolSession` (`_tool_session`)
tracks read files so write/edit refuse unread or externally changed files; it
survives `ToolManager.refresh`.
Families: `files/`, `commands/`, `web/` (fetch uses an injected summary client).

`ReadFileTool` pages raw source lines (`offset`/`limit`) **before** Markdown
expansion. Offset past EOF → first 2,000 lines; limit past EOF → remaining
lines. Expansion uses `DynamicMarkdownFile(path, content=...)`; output is
renumbered from one, not line-capped, and reports `source_start_line`,
`source_end_line`, `source_total_lines`. The per-line char cap applies after.
`raw_markdown=True` skips expansion for `.md` files and returns the same
source numbering and metadata as plain-text reads; the default is `False`.

### Permissions (`core/permissions/`)

`PermissionCategory` is **not** an `Enum` (plugins add tools and categories
later); a metaclass gives it `Enum`-like access over a case-insensitive
registry, plus `get`, `get_or_create`, `add_tool`. Built-ins: `READ`, `WRITE`,
`WEB`, `EXECUTE`, `PLUGIN`; default levels in `core/permissions/constants.py`,
others default to ASK. `PermissionLevel`: NONE, ASK, AUTO. Grants always cover
every registered category: Application builds the grants from
`permissions.default` plus the `permission_level` of each loaded plugin's
`PLUGIN:<name>` category, and sends a change of a plugin category to
`SettingsManager.set_plugin_permission_levels` /
`save_plugin_permission_level`, so Settings stays plain data and never reads
this registry. `PermissionManager` owns `ToolManager`; NONE tools are never instantiated.
`validate` returns `(PermissionDecision, hook_messages)`
(`PermissionOutcome` ALLOWED/DENIED/UNAVAILABLE + `note`); ASK goes to the
controller. The agent's `_permission_message` writes all model-facing wording.

### Hooks (`core/hooks/`)

`Hook` is a string enum. Each `HookManager` owns its registrations (no
singleton); registering a callable twice runs it twice. Handlers declare the
arguments they need by name (or `**kwargs`); positional-only is unsupported.
`provide(**deps)` registers permanent services: `hook_manager`,
`permission_manager`, `agent`, `client`. `run(hook, **event)` merges services
with event args (event wins). Handlers mutate the passed objects in place
(`agent.system_prompt`, `agent.turn_index`, `request`, `permission_decision`,
`tool_call.arguments`, `tool_call.id`, `result.content`, `result.display_text`);
returns are ignored except `Message`s, which are collected in order. Dispatch is
synchronous; ordinary handler exceptions propagate.

```python
@hook(Hook.SYSTEM_PROMPT_BUILD)
def customize(self, agent, **kwargs):
    agent.system_prompt += "\nAdditional instructions."
```

| Hook | Event args | Dispatched by |
| --- | --- | --- |
| startup / shutdown | none | `Application.start` / `.shutdown` |
| agent_start | message | `Agent`, before history; **not** retried |
| turn_start / turn_end | none | per turn; `turn_end` in `finally` |
| system_prompt_build | none | before `build_request`, same turn |
| before_model_request | request | after `build_request` |
| message_received | response_message | each finalized message |
| after_model_response | request, tool_calls | stream complete |
| agent_end | message_history | natural text-only stop |
| before/after_permission_check | tool_call (+ permission_decision) | once per `validate` |
| before/after_tool_call | tool_call (+ result) | `Tool.call`; after always fires |
| error `source="tool"` | source, error, tool_call | `_call` raised |
| error `source="model"` | source, error, request | every failed turn |
| error `source="agent"` | source, error | run ends on failure |

Notes:

- `request` is `None` for a model error raised inside `build_request`.
- `source="agent"` also fires with `error=None` when `turn.max` runs out.
  Exactly one of `agent_end` / `source="agent"` fires per run.
- A hook handler exception inside `Tool.call` surfaces as `source="model"` and
  is retried like a provider error (accepted behaviour).
- To change the target tool, edit the call in a model-response hook; selection
  is done before `before_tool_call`.
- Startup hook messages go to the UI only; shutdown messages are dropped.

### Plugins (`core/plugins/`)

`PluginManager.load(enabled) -> list[Message]` imports `plugin.py` from each
folder in `manager.PLUGINS_PATH` (sorted; hidden, `_`-prefixed and entry-less folders
skipped) as `arancio_plugins.<folder>.plugin`, so relative imports work.
Decorators (`core/plugins/decorators.py`) `@plugin`, `@hook`, `@tool` fill
`PLUGIN_REGISTRY` (`registry.py`) with `Plugin` records (`base.py`: internal,
not a base class). Metadata: name (class name), version `"0.0.0"`,
description, author; spaces in plugin names become dashes before
duplicate checks, tool naming and category creation. Bad or unknown fields skip
the class with a startup error. Empty plugins warn. Failed imports roll back their
declarations and modules. Settings decide loading, not code: `enabled` maps
lowercase plugin names (from the `plugins` settings) to their flag; a plugin
with no entry warns and is not loaded, a disabled one is skipped silently
(both are still imported, since the name is known only after import). Entries
for missing plugins are kept. Enabled names clashing case-insensitively are all
skipped. Each class is built once with `plugin_cls()`; its hooks and tools
share that instance. A failing plugin method returns one `ErrorMessage` and is
disabled; other methods keep running.

`@tool(...)` creates `PluginTool.create_class(...)` classes named
`<plugin_name>-<tool_name>` (method name by default). The description comes
from the decorator (no harness, no dynamic markdown); the schema is derived
from the decorated method like built-in tools (no `input_schema=` argument), and
a method that cannot be described is skipped with a startup error. `permission_category`:
`READ`, `WRITE`, `WEB`, `EXECUTE`, `PLUGIN` (default) or `CUSTOM` (→
`PLUGIN:<plugin_name>`, level from the plugin's `permission_level`).
`PluginToolResultParser` keeps dicts raw and displays them as JSON.

### Frontend (`ui/`)

`UIController` (`ui/controllers/base.py`): abstract `ask`,
`cancel_questions`, `show_login`, `reset_conversation`, `exit`. Answers: yes
allows, no/None denies, other text denies with a note. `TUIController` moves
every call to the UI thread with `call_from_thread`; `ask` blocks the worker
on a `threading.Event`; one pending ask at a time; after cancellation calls
are no-ops and `ask` returns None.

`TUIApp` (`ui/apps/tui.py`) holds only the Application: `on_mount` shows
`start()` output, `on_unmount` calls `shutdown()`, the toolbar reads
`status()`, and a worker thread consumes `submit(text)` into `MessageLog`
(closing the iterator in `finally`). `MessageLog` streams chunks into one
`MarkdownStream` per span; `LoginPanel` and `QuestionScreen` live in
`ui/widgets/`.

### Application (`application/`)

Only the application layer connects components: Commands call only
Application, and Settings and Sessions never read each other.

Public coordinator (`application/base.py`) over Core, Settings and Sessions;
returns Messages and `ApplicationStatus(model, working_directory)`
(`application/status.py`; `model`
is the active `Model`; the TUI shows its provider and name). Only
Application turns `settings.model` into an id, by calling each client's
`get_model_id`; Settings stores just `provider` and `name`, and
`Model.check_configured()` raises when either is missing.

- `update_settings(**changes)` takes the file layout:
  `model={"provider": ..., "name": ..., "reasoning": {"effort": ...}}` and/or
  `permissions={...}`. Application flattens them to the dotted paths that
  `SettingsManager` takes, rejects unknown keys first, then runs
  `model.provider` → `model.name` → `model.reasoning.effort` →
  `permissions`; each field is set, applied, saved alone, then recorded. No
  cross-layer rollback.
- Owns directory changes (`os.chdir`), `rename`, `new_session`, `resume`
  (restores settings without saving globals, history, unchanged file mtimes,
  directory) and `fork`.
- `start()` returns plugin/settings notices and STARTUP messages; `shutdown()`
  releases questions and dispatches SHUTDOWN once. Mutation while Core is
  running raises `RuntimeError`.
- `submit(text)` hands the input to `ActionExecutor` (see Prompt below).
- Application builds each `SessionConfiguration` from the active settings.

### Prompt & commands (`application/prompt/`, `commands/`)

Everything that comes from the user's input lives in `application/prompt/`.
`PromptManager` parses it into typed `Action`s (`actions/types.py`).
`ActionExecutor` (`actions/executor.py`) carries them out: it runs commands and
direct tools, records the user message before Core, yields each Core message
then records it with a `ToolSession` snapshot, and asks the controller for
permission questions. `FileMention(path, start_line,
end_line, raw_markdown=False)` supports `@file:110` and `@file:110-120`
(quoted paths too). Only `.md` file mentions support a trailing `:raw`
(`@file.md:raw`, `@file.md:110-120:raw`), passed as `raw_markdown=True`; the
executor turns valid ranges into `offset`/`limit`, invalid ones into a full
read. `!` keeps the exchange in history; `!!` does not. Mentions and `!`
use direct tools that bypass permissions.

Commands receive only `application`. Commands in
`commands/registry.py:SESSION_DISCARDING_COMMANDS` are recorded before
switching; others after. String results become `in_history=False`
`AssistantMessage`s.

### Settings & storage (`settings/`, `storage/`)

`SettingsManager(storage, supported_providers)` does no I/O at construction.
`settings` / `global_settings` return detached copies. `set_field`,
`restore_active`, `reset_global` change active values only; `save`,
`save_field`, `reset_default` write. `load()` returns a snapshot plus notices.
Missing file → created from defaults; bad YAML → left untouched, defaults used.
`SettingsValidator` returns checked mappings and Messages; strict for edits,
lenient with notices on load. Only the manager builds `Settings`.
`set_field` / `save_field` take dotted paths as written in the file
(`model.name`, `turn.max`, `permissions`); the manager walks the nested
`to_dict()` and re-validates strictly. `Settings.model` is Core's
`Model` (`provider`, `name`, `reasoning: Reasoning(effort,
summary)`) and `Settings.turn` is Core's `Turn`; a missing or null
`model` / `reasoning` / `turn` / `retry` section means defaults,
unknown keys warn, and there is no support for the old flat model and loop
keys.
`restore_active(model=, permissions=)` puts a
session's model section and permissions over the global settings.
`Settings.permissions` holds only the default categories
(`DEFAULT_PERMISSION_LEVELS`); `Settings.plugins` maps lowercase plugin names
to `PluginSettings(enabled, permission_level)`; plugin levels are edited by
plugin name with `set_plugin_permission_levels` and saved with
`save_plugin_permission_level`. A missing or null `plugins` means no plugins.
Invalid plugin entries warn and are left out; there is no support for the old
flat `permissions` layout.

`StorageManager` is generic file I/O rooted at `ARANCIO_DEFAULT_DIR` (text,
lines, 64 KiB byte chunks, sizes, discovery, writes, `move_directory`,
`create_symlink` — the link is created **at source**). Writes return None and
raise on failure. Callers must close abandoned byte iterators.

### Sessions (`sessions/`)

`SessionManager(storage, session_configuration)` is the only public service. Data types:
`sessions/base.py` (`Session`), `sessions/configurations.py`
(`SessionConfiguration`), `sessions/events/base.py`
(`SessionEvent`), `sessions/state.py` (`SessionState`), `sessions/scan.py`
(`SessionScan`) and `sessions/registry/entry.py` (`SessionRegistryEntry`); the
registry itself is `sessions/registry/base.py`. A session starts in
`Path.cwd()`; it is written only on its first recordable event.
`record_message(message, file_states=..., visible=True)` and
`record_command(...)` return `ErrorMessage | None`; chunks are skipped,
unsupported types (incl. `WarningMessage`) raise `ValueError`.
`update_state(**changes)` takes `configuration`, `working_directory`,
`explicit_name` (only this may be None). A `SessionConfiguration(model,
permissions)` stores the whole model section as Core's `Model` (summary
included) and only the default permission levels; plugin levels are global
settings, so sessions keep loading when plugins change. Sessions knows nothing
about Settings: Application builds the configuration from the settings and, on
resume, passes the session's `Model` to `restore_active` as it is. `find`,
`load` (validates, refreshes checksum, adds synthetic errors for unmatched tool
calls), `fork` (returns state, confirmation, notices). Internals
(`SessionRecorder`, `SessionChecksum`, `SessionRegistry`) are private. Sessions
never touches live Agent, Settings, `ToolSession` or the UI; Application applies
the returned data.

## Conventions

- **Docstrings**: Google style on all public symbols, tests included.
  Property summaries describe values rather than actions.
- **Style**: `ruff` (`E,F,I`), `ruff format`.
- **Comments**: lowercase start, no final period, only for non-obvious code.
- **Files**: `snake_case.py`, Markdown `UPPER_SNAKE_CASE.md`; `src/`, `tests/`.
- **Exceptions**: built-ins only; no custom exception classes.
- **Private members**: double leading underscore; constants in the package's
  `constants.py`; shared data types in `base.py`.
- **New provider / tool / command**: subclass the `base.py`. A tool also needs
  `harness/tools/<snake_name>.md`; type-hint `_call` and document its
  parameters in the docstring (Google `Args:` section). A command goes in
  `commands/registry.py` (and `SESSION_DISCARDING_COMMANDS` if it switches
  session) and delegates to Application.
- **New plugin**: plain class with `@plugin(...)` in `plugin.py`, plus
  `@hook`/`@tool` methods; no-arg constructor; no base class, no YAML.
- **Git**: branches `feature/snake_case` / `fix/snake_case`; past-tense commit
  messages naming the files; never credit coding agents.

## Testing

Only `pytest` (no mock/asyncio/coverage plugins); use `monkeypatch`,
`tmp_path`, stdlib. Tests are flat in `tests/`, one file per layer, plus
`conftest.py` and `fakes.py`; `test_integration.py` wires the real graph.

- **Fake only the real boundaries** (in `tests/fakes.py`): model
  (`ScriptedClient`), user (`ScriptedController`), Textual app (`FakeApp`,
  `RecordingApp`), network (monkeypatched `trafilatura`/`ddgs`). Everything else
  is real, on `tmp_path`.
- **Never touch the real home.** Repoint, on the *importing* module,
  `tools_base.TOOLS_HARNESS_PATH` and
  `system_prompt_builder.SYSTEM_PROMPT_HARNESS_PATH` (to repo `harness/`),
  `storage.manager.ARANCIO_DEFAULT_DIR` and `core.plugins.manager.PLUGINS_PATH`;
  chdir to a temp dir before creating sessions. `conftest.py` does this.
- **Plugins** are written to disk under `tmp_path` and loaded for real; an
  autouse fixture resets the registry and plugin modules.
- **ToolSession**: build one explicitly; Sessions only gets copied snapshots.
- **Docstring rules**: one line is enough for a test. `DOC501` wants `Raises:`
  for `raise ValueError(...)`; `DOC502` rejects it for `raise some_variable`.
- **Known bugs**: pin with `@pytest.mark.xfail(strict=True, reason=...)`.
