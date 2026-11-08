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

- `harness/tools/<snake_tool_name>/{description.md,input_schema.yml}`: read by
  `Tool.__init__` (raises if missing).
- `harness/SYSTEM_PROMPT.md`: seeded if absent, parsed once and cached by
  `SystemPromptBuilder`; `build()` appends the current date each turn. Put
  fresh runtime context in `build()`, never in a `<script>` tag.
- `settings.yml`: written by `SettingsManager` through Storage.
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

`Application.build(controller)` (`application/base.py`) creates
`StorageManager`, LiteLLM login paths, one shared `HookManager` and
`ToolSession`, the clients, `ToolManager`, then loads plugins, then Settings,
then `PermissionManager`, `Agent`, `SessionManager`, direct tools and
`Application`. **Plugins load before Settings/Permissions** so plugin
categories exist. Clients get `controller.show_login` as login callback; they
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
user note) → next turn. Stops on a text-only reply. `max_turns="inf"` is
unlimited; failures retry with exponential backoff up to `max_retries`, except
after finalized messages were delivered. The initial message is stored but
**not yielded**. The agent saves nothing; consumers forward its stream as is.

### Clients (`core/clients/`, `core/builders/`, `core/parsers/`)

`Client` sets `_request_builder`, `_payload_builder`, `_response_parser`. Flow:
`build_request` → mutable `Request` → payload builder → `litellm.responses` →
`parse` / `parse_stream` → `Message`s. `LiteLLMClient` routes by `model_id`
prefix; credentials come from env vars. `_litellm_patches.apply()` patches
pinned-LiteLLM bugs, including the ChatGPT device-code login notice sent to the
controller (UI-only, never in history).

The `litellm[mcp]` extra is required by the current Responses dispatcher even
for ordinary function tools; without it, model requests fail before reaching the
provider with `No module named 'mcp'`.

**Ollama caveat**: `reasoning={"effort": ...}` is always sent, but for
non-`gpt-oss` Ollama models LiteLLM turns it into `think: true|false`, so
low/medium/high are equal. Setting `thinking_summary` there crashes LiteLLM
(`unhashable type: 'dict'`); keep it `None` for Ollama.

### Tools (`core/tools/`)

Subclasses implement `_call`; `call(call)` returns `(result, hook_messages)`,
wrapping errors in `ToolErrorMessage` via `_result_parser`. Tool name = class
name. Descriptions are dynamic markdown (`<field>`, `<include>`, `<script>`).
The shared `ToolSession` (`_tool_session`) tracks read files so write/edit
refuse unread or externally changed files; it survives `ToolManager.refresh`.
Families: `files/`, `commands/`, `web/` (fetch uses an injected summary client).

`ReadFileTool` pages raw source lines (`offset`/`limit`) **before** Markdown
expansion. Offset past EOF → first 2,000 lines; limit past EOF → remaining
lines. Expansion uses `DynamicMarkdownFile(path, content=...)`; output is
renumbered from one, not line-capped, and reports `source_start_line`,
`source_end_line`, `source_total_lines`. The per-line char cap applies after.

### Permissions (`core/permissions/`)

`PermissionCategory` is **not** an `Enum` (plugins add tools and categories
later); a metaclass gives it `Enum`-like access over a case-insensitive
registry, plus `get`, `get_or_create`, `add_tool`. Built-ins: `READ`, `WRITE`,
`WEB`, `EXECUTE`, `PLUGIN`; default levels in `core/permissions/constants.py`,
others default to ASK. `PermissionLevel`: NONE, ASK, AUTO. Grants always cover
every registered category. `PermissionManager` owns `ToolManager`; NONE tools
are never instantiated. `validate` returns `(PermissionDecision, hook_messages)`
(`PermissionOutcome` ALLOWED/DENIED/UNAVAILABLE + `note`); ASK goes to the
controller. The agent's `_permission_message` writes all model-facing wording.

### Hooks (`core/hooks/`)

`Hook` is a string enum. Each `HookManager` owns its registrations (no
singleton); registering a callable twice runs it twice. Handlers declare the
arguments they need by name (or `**kwargs`); positional-only is unsupported.
`provide(**deps)` registers permanent services: `hook_manager`,
`permission_manager`, `agent`, `client`. `run(hook, **event)` merges services
with event args (event wins). Handlers mutate the passed objects in place
(`agent.system_prompt`, `agent.turn`, `request`, `decision`, `call.arguments`,
`call.id`, `result.content`, `result.display_text`); returns are ignored except
`Message`s, which are collected in order. Dispatch is synchronous; ordinary
handler exceptions propagate.

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
| before/after_permission_check | call (+ decision) | once per `validate` |
| before/after_tool_call | call (+ result) | `Tool.call`; after always fires |
| error `source="tool"` | source, error, call | `_call` raised |
| error `source="model"` | source, error, request | every failed turn |
| error `source="agent"` | source, error | run ends on failure |

Notes:

- `request` is `None` for a model error raised inside `build_request`.
- `source="agent"` also fires with `error=None` when `max_turns` runs out.
  Exactly one of `agent_end` / `source="agent"` fires per run.
- A hook handler exception inside `Tool.call` surfaces as `source="model"` and
  is retried like a provider error (accepted behaviour).
- To change the target tool, edit the call in a model-response hook; selection
  is done before `before_tool_call`.
- Startup hook messages go to the UI only; shutdown messages are dropped.

### Plugins (`core/plugins/`)

`PluginManager.load() -> list[Message]` imports `plugin.py` from each folder in
`manager.PLUGINS_PATH` (sorted; hidden, `_`-prefixed and entry-less folders
skipped) as `arancio_plugins.<folder>.plugin`, so relative imports work.
Decorators (`core/plugins/decorators.py`) `@plugin`, `@hook`, `@tool` fill
`PLUGIN_REGISTRY` (`registry.py`) with `Plugin` records (`base.py`: internal,
not a base class). Metadata: name (class name), version `"0.0.0"`,
description, author, enabled `True`; spaces in plugin names become dashes before
duplicate checks, tool naming and category creation. Bad or unknown fields skip
the class with a startup error. Empty plugins warn. Failed imports roll back their
declarations and modules. Enabled names clashing case-insensitively are all
skipped. Each class is built once with `plugin_cls()`; its hooks and tools
share that instance. A failing plugin method returns one `ErrorMessage` and is
disabled; other methods keep running.

`@tool(...)` creates `PluginTool.build(...)` classes named
`<plugin_name>-<tool_name>` (method name by default). Description and schema
come from the decorator (no harness, no dynamic markdown). Category: `READ`,
`WRITE`, `WEB`, `EXECUTE`, `PLUGIN` (default) or `CUSTOM` (→
`PLUGIN:<plugin_name>`, default ASK). `PluginToolResultParser` keeps dicts raw
and displays them as JSON.

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

### Application (`application/base.py`)

Public coordinator over Core, Settings and Sessions; returns Messages and
`ApplicationStatus(model_id, thinking_effort, working_directory)`.

- `update_settings(**changes)`: provider → model_name → thinking_effort →
  permissions; unknown names rejected first; each field is set, applied,
  saved alone, then recorded. No cross-layer rollback.
- Owns directory changes (`os.chdir`), `rename`, `new_session`, `resume`
  (restores settings without saving globals, history, unchanged file mtimes,
  directory) and `fork`.
- `start()` returns plugin/settings notices and STARTUP messages; `shutdown()`
  releases questions and dispatches SHUTDOWN once. Mutation while Core is
  running raises `RuntimeError`.

### Prompt & commands (`prompt/`, `commands/`)

`PromptManager` returns typed `Action`s. `FileMention(path, start_line,
end_line)` supports `@file:110` and `@file:110-120` (quoted paths too); the
executor turns valid ranges into `offset`/`limit`, invalid ones into a full
read. `ActionExecutor.execute(text)` routes input, records the user message
before Core, yields each Core message then records it with a `ToolSession`
snapshot. `!` keeps the exchange in history; `!!` does not. Mentions and `!`
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

`StorageManager` is generic file I/O rooted at `ARANCIO_DEFAULT_DIR` (text,
lines, 64 KiB byte chunks, sizes, discovery, writes, `move_directory`,
`create_symlink` — the link is created **at source**). Writes return None and
raise on failure. Callers must close abandoned byte iterators.

### Sessions (`sessions/`)

`SessionManager(storage, configuration)` is the only public service; data types
are in `sessions/base.py`. A session starts in `Path.cwd()`; it is written only
on its first recordable event. `record_message(message, file_states=...,
visible=True)` and `record_command(...)` return `ErrorMessage | None`; chunks
are skipped, unsupported types (incl. `WarningMessage`) raise `ValueError`.
`update_state(**changes)` takes `configuration`, `working_directory`,
`explicit_name` (only this may be None). `find`, `load` (validates, refreshes
checksum, adds synthetic errors for unmatched tool calls), `fork` (returns
state, confirmation, notices). Internals (`SessionRecorder`, `SessionChecksum`,
`SessionRegistry`) are private. Sessions never touches live Agent, Settings,
`ToolSession` or the UI; Application applies the returned data.

## Conventions

- **Docstrings**: Google style on all public symbols, tests included.
- **Style**: `ruff` (`E,F,I`), `ruff format`.
- **Comments**: lowercase start, no final period, only for non-obvious code.
- **Files**: `snake_case.py`, Markdown `UPPER_SNAKE_CASE.md`; `src/`, `tests/`.
- **Exceptions**: built-ins only; no custom exception classes.
- **Private members**: double leading underscore; constants in the package's
  `constants.py`; shared data types in `base.py`.
- **New provider / tool / command**: subclass the `base.py`. A tool also needs
  `harness/tools/<snake_name>/`. A command goes in `commands/registry.py`
  (and `SESSION_DISCARDING_COMMANDS` if it switches session) and delegates to
  Application.
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
