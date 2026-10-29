# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project

`arancio` — "another coding agent". A terminal (Textual TUI) coding agent that drives a
multi-turn loop against any LLM through the LiteLLM **Responses API**. Python ≥ 3.11,
managed with `uv`.

## Commands

Everything runs against the project venv (`.venv/`). The git hooks call binaries directly
(`.venv/bin/ruff`); interactively you can also use `uv run <cmd>`.

- **Run the app**: `uv run arancio` (console script → `arancio.ui.__main__:main`), or `.venv/bin/arancio`
- **Install / sync deps**: `uv sync` (add a dep with `uv add <pkg>`)
- **Tests**: `.venv/bin/python -m pytest tests/`
  - single file: `... -m pytest tests/test_agents.py`
  - single test: `... -m pytest tests/test_agents.py::test_name`
  - the suite is flat under `tests/`, one file per layer, plus `conftest.py`
    (isolation and shared fixtures) and `fakes.py` (every hand-written double).
    `tests/test_integration.py` wires the real object graph end to end.
- **Lint**: `.venv/bin/ruff check src/ tests/` (rules `E,F,I`); autofix `ruff check --fix`; format `ruff format src/ tests/`
- **Docstring completeness** (Google style, enforced on push): `.venv/bin/ruff check --select=D,DOC --preview src/ tests/`
- **Docstring wrapping**: `.venv/bin/docformatter --in-place --recursive --wrap-summaries 88 --wrap-descriptions 88 src/ tests/`

`core.hooksPath` is set to `hooks/`, so the checked-in hooks are already active:
`pre-commit` auto-fixes + formats staged `src/`/`tests/` and blocks on remaining lint
errors; `pre-push` runs the docstring check and the full test suite.

## Runtime working directory (important)

At runtime tools do **not** read their prompts from the repo. They read from the arancio
working directory `~/.arancio/` (shared paths are in `core/constants.py`, with
component-specific paths in each component's `constants.py`).
Storage uses `ARANCIO_DEFAULT_DIR` as its root;
startup and Sessions caller migration is deferred; Settings now uses Storage's text API:

- `~/.arancio/harness/tools/<snake_tool_name>/{description.md,input_schema.yml}` — each
  tool's description and JSON input schema, loaded from disk by `Tool.__init__`.
- `~/.arancio/harness/SYSTEM_PROMPT.md` — the agent's system prompt.
  `SystemPromptBuilder` seeds it with a default when absent, parses it as dynamic
  markdown once, and caches the text; `build()` (called once per agent turn) returns
  that cached text with the current date appended. Runtime context that must stay
  fresh belongs in `build()`, **not** in a `<script>` tag in the markdown — the file is
  parsed once, so a date computed there would freeze at startup.
- `~/.arancio/settings.yml` — values persisted by SettingsManager through Storage;
  `settings/constants.py` supplies only the relative `ARANCIO_SETTINGS_FILENAME`.
- `~/.arancio/litellm/` — the intended destination for LiteLLM login data. Future
  startup wiring will supply the paths to Storage's generic `move_directory` and
  `create_symlink`; the provider-specific Storage helper has been removed.
- `~/.arancio/sessions/<creation-cwd>/<year>/<month>/<day>/<id>.jsonl` — one
  append-only chat session per file. The readable creation-CWD directory has a hash suffix
  to avoid collisions; session IDs are UUID4 hex strings. There is no persisted registry:
  `SessionRegistry` is rebuilt by scanning these files at launch.
- `~/.arancio/plugins/<folder>/plugin.py` — entry files scanned at launch by
  `PluginManager` (see **Plugins** below). Each `@plugin(...)` class is an independent
  plugin; one file can hold several. The user places folders here by hand: nothing
  creates the directory, and a missing one means no plugins. Folders without `plugin.py`
  are skipped. YAML manifests are not read.

The repo's `harness/` is the **source** for those prompt files; there is no auto-copy, so
editing `harness/` in the repo has no effect on a real run until the files are placed
under `~/.arancio/harness/`. **Tests** must sidestep this by repointing
`tools_base.TOOLS_HARNESS_PATH` and
`system_prompt_builder.SYSTEM_PROMPT_HARNESS_PATH` at the repo's `harness/` before any
test imports, or they read the real home directory. The previous suite's
`tests/conftest.py` did this at import time. Test sources were removed before the
Storage refactor and are not restored in this step. `Tool.__init__` reads the path
at construction and raises when the directory is missing.

**Storage's root anchors paths; it is not a sandbox.** Every Storage operation
anchors relative paths to the application root; explicit absolute paths are accepted
as supplied. No Storage method substitutes a global settings or login path.
Other modules still have their own home-directory defaults. Future tests must
redirect these on the *importing* module or pass explicit temporary paths:

| Constant, as imported | Read by | What an unpatched test does |
| --- | --- | --- |
| `core.plugins.manager.PLUGINS_PATH` | `PluginManager.__init__` | imports and runs the user's real plugins |
| `sessions.manager` `root` default | `SessionManager.__init__` | scans and writes the user's real session history |

Storage tests should repoint `storage.manager.ARANCIO_DEFAULT_DIR` and use temporary
source/target paths. Plugin loading remains a separate I/O boundary, outside Storage.

## Architecture

Layered and provider-agnostic. The recurring pattern is a `base.py` abstract class plus a
provider file (e.g. `litellm.py`); adding a provider means a new `Client` subclass
with its own builder/parser trio, not touching the core loop.

**Wiring** (`ui/__main__.py:main`) — the existing entry point is legacy caller code
awaiting its architecture migration. It still calls Storage's removed login helper,
supplies SettingsManager's removed client/agent arguments and expects `load_settings()` to
apply settings to Core. Commands still mutate returned settings and call the removed
`apply()`; Sessions still calls `apply_session_configuration()`. These callers
are deliberately not adapted in the Settings step, so the full app is not wired yet.

The next startup migration must discover plugins before constructing SettingsManager
and supply its supported provider names. Loading returns a detached Settings
snapshot and notices; Application will apply model/thinking values, Agent loop limits
and permissions to Core. Build the initial session after loading so its header uses the
selected values. Startup notices are displayed without entering history or creating a
session log. Application will own later apply/save/session coordination.

**Messages** (`core/messages/`) — the lingua franca between clients, agent and UI. Every
message carries `content` (fed to the model) and `display_text` (shown in the UI), plus
**one** routing flag: `in_history` (model context). That is the only consumer core speaks
for, because it owns neither a session nor a UI — persistence and display are each
decided by the consumer that owns them: `SessionRecorder` keeps everything but chunks and
takes `visible` as an argument from whoever asks for the write, and the UI renders what it
is handed. `in_history` defaults to true, except on `ErrorMessage`, which defaults to
`False` — an error reports what went wrong in this run, not what the conversation was, so
it is shown and saved but never fed back to the model. `ToolErrorMessage` is the exception
to the exception: `ToolResultMessage.__init__` wins in the MRO, so a failed tool call keeps
`in_history=True`, being the outcome the model asked for. The hierarchy uses multiple inheritance
for the chunk/error variants (`AssistantChunkMessage`, `ReasoningChunkMessage`,
`ToolErrorMessage`). `ChunkMessage` marks streaming deltas;
finalized messages are everything else. `WarningMessage` mirrors `ErrorMessage` at
non-fatal severity (rendered with the `.warning` CSS class instead of `.error`); settings
validation is its main producer. There is no audit message type: a fact that only a
session would keep — the mtime a file tool left behind — is derived by the session layer
from the shared `ToolSession`, not announced by core.

**Agent loop** (`core/agents/base.py`) — holds a flat `List[Message]` history, empty unless the
constructor is given a `message_history` (copied, not aliased) — the seam a resumed session
will use to hand its restored model context to the agent at startup, alongside the existing
post-construction `restore_history`. Per turn: build
request → stream response messages (yielding chunks live, appending only finalized ones to
history) → if any `ToolCallMessage`, gate each through `PermissionManager.validate`, run
it, append the `ToolResultMessage` (+ optional user note) → next turn. Natural stop on a
text-only reply. `max_turns="inf"` means unlimited; failed turns retry with exponential
backoff up to `max_retries` consecutive failures. A post-stream exception that fires
*after* finalized messages were delivered does not retry.

**The agent does not save anything.** Persistence is not agent business logic, and core
owns no session. The agent is called directly — `__call__()` (there is no `run()`) yields
`Iterator[Message]`: `_emit` appends a message to history
and yields it, nothing more. Every message it yields is meant for the user, so a consumer
forwards the stream unfiltered — see `ActionExecutor` under **Prompt & commands**.
The initial `message` is appended to history but **never yielded**, because the caller
that built it already holds it.

**Client + request pipeline** (`core/clients/`, `core/builders/`, `core/parsers/`) —
`Client` declares strategy class attrs `_request_builder` / `_payload_builder` /
`_response_parser`. Flow: `build_request` → normalized frozen `Request` dataclass
(model, thinking, system prompt, tools, messages) → `send_request` builds LiteLLM kwargs
via the payload builder → `litellm.responses(**kwargs)` → response parser turns provider
output back into normalized `Message`s (`parse` for a single response, `parse_stream` for
SSE). `LiteLLMClient` routes *any* provider by `model_id` prefix (`openai/…`,
`anthropic/…`, …) and lets LiteLLM resolve credentials from env vars. The Responses API is
used specifically because it round-trips reasoning items (`ReasoningMessage`).
`_litellm_patches.apply()` monkeypatches bugs in the pinned LiteLLM at import;
`_register_native_streaming` works around LiteLLM's fake-stream path for models missing
from its registry.

For a ChatGPT device-code login, the LiteLLM patch emits a one-way controller request with
the verification URL and user code before LiteLLM polls for authorization. `UIController`
places a clickable link and copyable URL/code in the log, returning immediately; sign-in
details are UI-only and never enter agent history.

**Thinking effort is not graded on every provider.** `reasoning={"effort": ...}` is sent
unconditionally by `LiteLLMPayloadBuilder.build` regardless of provider. For Ollama models
(`ollama/…`, `ollama_chat/…`), LiteLLM bridges the Responses call through its chat-completion
path, and its Ollama param mapping (`litellm/llms/ollama/{chat,completion}/transformation.py`)
reduces `reasoning_effort` to a boolean: `optional_params["think"] = value in {"low",
"medium", "high"}` for any model not prefixed `gpt-oss`. So `/effort low|medium|high` all
collapse to the same `think: true` (no depth distinction) and any other value silently
becomes `think: false` — only `gpt-oss` models forward the tier string as-is. Separately,
setting `thinking_summary` alongside `thinking_effort` for a non-`gpt-oss` Ollama model
crashes at the same mapping line with `TypeError: unhashable type: 'dict'` (LiteLLM keeps
the whole `reasoning` dict as `reasoning_effort` once `summary` is present), so
`thinking_summary` should stay `None` for Ollama models until upstream fixes this.

**Tools** (`core/tools/`) — `Tool` subclasses implement `_call`; the public `call`
wraps success/exception into a `ToolResultMessage`/`ToolErrorMessage` through the tool's
`_result_parser`. It returns `(result, hook_messages)`, keeping plugin errors separate
from the tool outcome. The tool's **name is its class name**, and its description/input schema
are loaded from the harness dir (see runtime section); descriptions are dynamic-markdown
(`<field>`, `<include>`, `<script>` tags expand against the tool instance). A
`ToolSession` records which files were read so write/edit tools can refuse blind
overwrites of unread or externally-modified files. Tools receive it as `tool_session`
and store it as `_tool_session`. The same instance survives the tool rebuilds performed
by `ToolManager.refresh_tools` whenever permissions change. Families: `files/` (read, write, edit),
`commands/` (shell), `web/` (search, fetch — fetch takes an
injected summary client).

**Permissions** (`core/permissions/`) — `PermissionCategory` groups tool classes sharing a
capability: the built-in `READ`/`WRITE`/`WEB`/`EXECUTE`/`PLUGIN`, plus whatever per-plugin
categories `@tool(category="CUSTOM")` creates at plugin-load time (see **Plugins** below).
It is **not** an `enum.Enum` — a category's tool set has to stay open for a plugin tool
to join it, or for a wholly new category to appear, after the built-ins already exist,
which a closed `Enum` cannot do. A small `_PermissionCategoryMeta` metaclass gives it the
same class-level surface an `Enum` would (`for category in PermissionCategory`,
`PermissionCategory[name]`, `PermissionCategory.__members__`), backed by a `_REGISTRY`
dict keyed by uppercased name so lookup stays case-insensitive; `get(name)`/
`get_or_create(name)` are the non-raising, idempotent ways to read or register a category,
and `add_tool(tool_cls)` grows its mutable `.tools` set (not names → still refactor-safe).
`PermissionLevel` is NONE, ASK or AUTO. The grants dict always covers every *currently
registered* category — a category not granted is present at `NONE`, never absent — which
is why `PluginManager.load_plugins()` (see **Plugins**) has to run, and does run, before any
`PermissionManager`/`SettingsManager` is built. Settings snapshots the known categories
from the registry at construction; categories discovered afterward are not
added to its snapshots.
`PermissionManager` owns the
`ToolManager`, so tools in a NONE category are **never even instantiated**; the
`allowed_tools` property is what builds them. `validate` runs
AUTO calls silently and delegates ASK calls to the controller; denials and user notes are
fed back to the model as messages rather than raising. It returns `(decision, hook_messages)`. The plain
`PermissionDecision` (`core/permissions/types.py`): a `PermissionOutcome`
(ALLOWED / DENIED / UNAVAILABLE) plus the user's `note`. It forwards hook messages but
builds no permission messages — the
agent's `_permission_message` turns the decision into the `ToolErrorMessage` (worded
differently for a denial and for a missing tool) or the report-then-answer `UserMessage`,
so model-facing wording stays in the one place that talks to the model. How a call was
authorized is **not** recorded — the session keeps only what a resumed run needs, and the
`ToolResultMessage`/`ToolErrorMessage` already say whether the tool ran.

**Hooks** (`core/hooks/`) — `Hook` (`core/hooks/base.py`) is a string-valued enum naming
every dispatch point: `startup`/`shutdown`, `agent_start`/`agent_end`, `turn_start`/`turn_end`,
`system_prompt_build`, `before_model_request`/`after_model_response`,
`message_received`, `before_tool_call`/`after_tool_call`,
`before_permission_check`/`after_permission_check`, and a single `error`. `HookManager`
(`core/hooks/manager.py`) owns its own registrations — there is no global singleton, so
two managers never share handlers. `register(hook, handler)` appends a handler for a hook;
registering the same callable twice adds two invocations. Every handler takes **one**
argument: the manager's public `context` dict. `run_hook(hook, **event)` first writes the
event's own data into `context` under those keys (object identity kept), then calls every
handler registered for that hook synchronously, on the caller's thread, in registration
order, passing the **same** `context` object to each — never a copy. So a handler changes
a value by assigning a key (`context["system_prompt"] = ...`), and the next handler and
the caller both see it. Event keys are **not** erased after dispatch: they stay until a
later `run_hook` overwrites them, so a handler should read only the keys its own hook sets
(a `turn_start` handler still finds the last tool call's `result`). Besides the event
keys, core fills the permanent keys **by itself**, with no outside wiring: each core
component adds itself to the context of the hook manager it is built with —
`HookManager.__init__` starts it as `{"hook_manager": self}`, `PermissionManager` adds
`permission_manager`, and `Agent` adds `agent` and `client`. The clients never receive
a hook manager, so the object holding each one registers it. So any code that builds the
core objects with one shared hook manager gets a full context, whatever frontend (if
any) drives them. If two objects of the same kind share one hook manager, the last one
built owns the key. Returned `Message` objects are collected into a list in dispatch order; other return
values are ignored. An exception raised by an ordinary handler propagates immediately,
so handlers after it do not run. Dispatch iterates a snapshot of the handler list taken when `run_hook` starts, so
a handler that registers another handler mid-dispatch only affects the next call to `run_hook`.
A hook with no registered handlers returns an empty list. Both `core/hooks/__init__.py` and every
sibling `core` package's `__init__.py` are empty; a caller imports `Hook` from
`core.hooks.base` and `HookManager` from `core.hooks.manager` directly, the same as any
other two-file `base.py`/`manager.py` package (`core/permissions/`, `core/controllers/`).

Threading follows the same explicit-constructor-injection pattern as
`tool_session`, since two callers must be able to hold independent hook managers, stored as
`self._hook_manager`. Every constructor that takes `hook_manager` (`Agent`,
`PermissionManager`, `ToolManager`, `Tool`/`FetchWebTool`, `ActionExecutor`) requires
it, with no `None` default: production code (`ui/__main__.py`) never has a reason to omit
it for any of them, so requiring it catches a caller that forgot to thread the shared
instance through, rather than silently falling back to a private no-op manager.
`ui/__main__.py:main` builds **one** `HookManager()` alongside `controller` and passes the
same instance into `ToolManager`, `PermissionManager`, `Agent` and `App` (which forwards it
to the `ActionExecutor` it builds and keeps it for application lifecycle hooks);
`ToolManager.create_tools` re-threads it into every tool
it builds, so one handler registered anywhere in that graph sees every dispatch from every
component — including the `!`/`!!` shell path and `@mention` reads, which run through
`ActionExecutor`'s own `ReadFileTool`/`ShellCommandTool` instances (built once in its
`__init__`, not as module-import-time class attributes, precisely so the shared instance can
reach them).
`Client`/`LiteLLMClient` never receive one: `before_model_request` /
`after_model_response` / the model-sourced `error` are dispatched by `Agent.__call__`
around its calls to `self._client.build_request`/`send_request`, not by the client itself —
placing them inside `LiteLLMClient` would never fire for `ScriptedClient`, which overrides
`send_request` outright and never calls into `LiteLLMClient` code. The same reasoning means
`FetchWebTool._call`'s own internal summarization request (it calls
`self._client.build_request`/`send_request` directly, outside any `Agent` loop) is
invisible to those hooks too — only the `before_tool_call`/`after_tool_call`/`error`
wrapped around the whole `FetchWebTool.call` see it.

Dispatch sites, by hook:
- `App.on_mount`: `startup` (no event keys), once the app's log is mounted and
  before any user prompt. The shared context already contains the core objects.
  Returned messages join the settings and plugin-load startup messages in the UI;
  they are not added to agent history or saved to a session.
- `App.on_unmount`: `shutdown` (no event keys), once during final application
  teardown before exit. It runs for `/exit`, `/quit`, Ctrl+C and direct app exit.
  This hook is for cleanup; the widgets have been unmounted, so returned messages
  are not displayed or saved. Application hooks do not create a session log.
- `Agent.__call__`: `agent_start` (`message`) — first statement, before the message is
  even recorded to history, and before the turn loop's own error handling exists: a
  handler exception here is not caught, retried or turned into an `ErrorMessage`, unlike
  every other dispatch below. Then per turn: `turn_start` (`turn`, the loop index, 0-based)
  — first line inside that turn's `try`, so a broken handler is retried like any other turn
  failure and still gets a paired `turn_end`; `before_model_request` (`request`) right after
  `build_request` returns; `message_received` (`response_message`) for each finalized
  (non-chunk) message as it streams in; `after_model_response` (`request`, `tool_calls`)
  once the stream completes; then exactly one of `agent_end` (`message_history`,
  natural text-only stop) or `error` with `source="model"` (`error`, `request` — `None`
  when `build_request` itself failed before returning one), dispatched on *every* failed
  turn, retried or not. `error` with `source="agent"` (`error`) additionally fires only
  when the run is about to actually end on that failure (the `received_finalized` early
  return, or `max_retries` exhausted), and once more with `error=None` when `max_turns` is
  exhausted by loop exhaustion rather than an exception — so exactly one of
  `agent_end`/`source="agent"` fires per completed run, modulo an unguarded `agent_start`
  failure. `turn_end` (`turn`) fires from a `finally` wrapping the whole turn (one
  `try`/`except`/`finally`, no extra nesting), exactly once per turn regardless of exit
  path — including, per ordinary Python `finally` semantics, that a `turn_end` handler's
  own exception replaces an in-flight `return` or exception rather than being appended
  after it. Returned messages are yielded except during `GeneratorExit`, when cleanup
  runs without yielding.
- `Agent.__call__`: `system_prompt_build` (`system_prompt`), immediately after building
  the prompt and before `build_request`, inside the same per-turn `try`. The agent then
  reads `context["system_prompt"]` back and builds the request from that, so a handler
  that replaces it changes the prompt of the **same** turn. With `result` (see
  `Tool.call` below), it is one of the two event keys core reads back today.
- `Tool.call`: `before_tool_call` (`name`, `call_id`, `arguments`) before running;
  `error` with `source="tool"` (`name`, `call_id`, `arguments`, `error`) when `_call`
  raises; `after_tool_call` (`name`, `call_id`, `arguments`, `result`) once the result
  message is built, unconditionally — both the `source="tool"` `error` and
  `after_tool_call` fire on a failed call. `call` then returns `context["result"]`, read
  back after `after_tool_call`, so a handler that replaces it (even with a
  `ToolErrorMessage`) changes the result the agent and executor receive. Only `_call`'s own exceptions are caught by
  `call`'s try/except; a handler exception raised during any of these three dispatches
  propagates out of `call` itself, unlike a tool's own failure. When that happens inside
  `Agent.__call__`'s tool-calling loop, it is caught by the *same* `except Exception as
  e:` that catches a genuine model-call failure and dispatched the same way — `error`
  with `source="model"`, surfaced as `"Error while executing user request: ..."`, and
  retried/backed-off exactly like a provider outage, never as `source="tool"` since
  `_call` itself never ran. This is accepted, not fixed: splitting the turn's try/except
  to disambiguate would change existing retry semantics beyond wiring dispatch calls.
- `PermissionManager.validate`: `before_permission_check` (`call`) then
  `after_permission_check` (`call`, `decision`), exactly one pair per call regardless of
  which of the unavailable/auto/ask outcomes resolves it — `validate` is now a thin
  before/after wrapper around `_resolve`, which holds the unchanged unavailable/auto/ask
  logic `validate` used to hold directly, so the dispatch pair isn't tripled across the
  three early returns.

The three error sources (`tool`/`model`/`agent`) are unified under the single `error` hook
rather than three separately named ones, distinguished by a `source` key — a handler that
wants every failure registers once for `error` and reads `source`; one that only cares about
tool failures filters on `context["source"] == "tool"`. The names in parentheses in the
list above are the event keys each dispatch writes into `context`.

**Plugins** (`core/plugins/`) — `PluginManager.load_plugins()` imports `plugin.py` from
immediate folders under `~/.arancio/plugins/`, sorted by folder name. Hidden and
underscore-prefixed folders and folders without that entry file are skipped. The root
is the module constant `manager.PLUGINS_PATH`, read by `PluginManager.__init__`, and
nothing creates it. Tests must repoint that importing module's constant; the autouse
home-isolation fixture does this. No manifest or plugin base class is required.

`core/plugins/registry.py` owns one `PLUGIN_REGISTRY: list[Plugin]`, filled by the
`@plugin`, `@hook` and `@tool` decorators in `core/plugins/decorators.py`. There is no
registry class, loader class or metadata validator. Method decorators run before their
class exists, so two weak-key pending maps hold hook events and raw tool fields. The
class decorator reads its own functions once and stores hook declarations and generated
tool classes in separate `hooks` and `tools` lists on a `Plugin` record. Each plugin
gets its own tool classes, including when sharing decorated functions. Pending entries
remain reusable while the function lives. Stacked hooks and either hook/tool
decorator order work without changing the original functions or classes.

Each valid decorated class registers immediately, including decorated classes in
imported helpers. There is no later class discovery or source filter. Class aliases
and undecorated subclasses add no declarations. Only methods defined directly on the
decorated class are collected. An entry without decorated classes contributes nothing.
A valid plugin without hooks or tools stays registered and produces a startup warning,
including when disabled; enabled empty plugins are still constructed.

`Plugin` (`core/plugins/base.py`) is a mutable internal record, not a base class for
plugin authors. It keeps `plugin_cls`, `source_path`, metadata, `instance`, `hooks` and
`tools`. Name defaults to the class name; version defaults to `"0.0.0"`, description and
author to empty strings, and enabled to `True`. `Plugin.__post_init__` requires strings
for the text fields and a boolean for enabled; it raises `TypeError` for wrong types and
`ValueError` for an empty name. Unknown decorator fields are also errors, rather than
ignored values. The class decorator catches metadata errors, queues one startup error
naming the source class,
field and bad value, and returns the unchanged class without registering it. Sibling
classes in the same file continue loading. Runtime fields cannot be supplied as metadata.

`PluginManager.registry` references the shared list directly; it replaces `_plugins`
and the old `plugins` property. Disabled plugins, conflicting names and constructor
failures remain in that list with `instance=None`. After all imports, enabled names
are compared case-insensitively. Every member of a conflicting group is reported and
skipped before construction or hook/tool registration. Disabled records do not
participate. Each eligible class is constructed once with `plugin_cls()`; a constructor
failure reports only that class. A class's hooks and tools share its one instance,
including when permission changes rebuild tool instances.

Entry files load as `arancio_plugins.<folder>.plugin`. Their parent package points to
the entry folder, allowing `from .helpers import X` without changing `sys.path` or
mixing helpers from different folders. `PLUGIN_PACKAGE_ROOT` lives in
`core/plugins/constants.py`. Importing executes module-level code even for disabled
plugins. An unexpected import failure discards all declarations and queued diagnostics
added during that import, removes newly cached plugin modules (including helpers), and
reports the failure; unrelated entries still load. Cached entry modules and constructed
instances are reused on subsequent `load_plugins()` calls. This is startup loading, not live reload.

`PluginManager.load_plugins() -> list[Message]` drains the temporary decorator diagnostic buffer
and returns startup errors and warnings through the normal UI path. It runs before
permission and settings initialization, so plugin tool categories exist when default
grants and the tool catalog are built. Instances receive no injected metadata, paths
or context attributes. Bundled resources are available through
`Path(__file__).resolve().parent`; the hook context remains separate from metadata.

The manager binds each declared method once after construction. `@hook(Hook.X)` methods
receive the shared context dictionary. Returned messages enter the normal message
stream; other return values are ignored. Notices should use `in_history=False` to stay
outside model history. A failed method returns one `ErrorMessage` and is disabled across
all its hooks for that instance. The disabled key uses instance identity and the original
function, so shared functions and unhashable plugin instances remain independent. Other
methods, tools and plugins keep running, and ordinary hook handlers still propagate errors.
Runtime plugin errors are shown and saved without causing agent retries.

`@tool(...)` defines a real tool named `<plugin_name>-<tool_name>`, with the method name
as the tool-name default. Permission lookup, hook events, agent catalog, history and
provider payloads use that same identifier. Description and input schema come directly
from the decorator, without harness files or dynamic-markdown expansion. Categories are
`READ`, `WRITE`, `WEB`, `EXECUTE`, `PLUGIN` (default), or `CUSTOM`. CUSTOM creates
`PLUGIN:<plugin_name>`, shared by that plugin's tools and defaulting to ASK. Unknown
categories report and skip only their tool. Renamed categories use normal settings
validation, with no automatic grant transfer.

The class decorator calls `PluginTool.build(plugin_name, method, **fields)` to create
each tool class. Tool name, category, description, raw input schema and method live
directly on that class; there is no `PluginToolSpec`, `_spec` attribute or method/spec
tuple. The default category is declared on `PluginTool`. During startup, the manager
binds each tool's method and adds that same class to its permission category. Tool
instances are still created through the normal ToolManager only when permissions allow
them, using `tool_cls(hook_manager=...)`. Each instance keeps the normal closed-object
schema, without changing the class's raw schema. `PluginToolResultParser` keeps
JSON-compatible dictionaries as raw content and displays readable JSON; other results
and exceptions retain base behavior.

**Controller port** (`core/controllers/`) — the core↔UI seam (ports & adapters). Core
sends a `ControllerRequest` (e.g. `PermissionRequest`) and gets a
`ControllerResponse` (e.g. `PermissionResponse` carrying a `Decision`), dispatched by
`isinstance`. `ui/controller.py:UIController` is the Textual adapter: it blocks the
**worker** thread on a `threading.Event` until the user answers a `QuestionScreen`, never
the UI thread. Its `app` is genuinely `None` until wiring finishes — the app needs the
agent, which needs the permission manager, which needs the controller — so every dispatch
goes through `require_app()`, which raises rather than letting each call site assume the
attachment happened. `LiteLLMClient` **requires** its controller: both production clients
get one, and the login notice has nowhere to go without it. `ChatGPTLoginRequest` is
one-way: the frontend shows the login instructions while the worker waits for the
provider's authorization polling. Plugin failures use the normal message stream.


**UI** (`ui/app.py`) — Textual app. A worker thread runs the agent/executor; each message
is rendered on the main thread via `call_from_thread`. Assistant and reasoning chunks
stream into one `MarkdownStream` widget per span, and the finalized echo of an
already-streamed span is suppressed so text is not shown twice.

**Prompt & commands** (`prompt/`, `commands/`) — `PromptManager.resolve_prompt` classifies
input: `!<shell command>`/`!!<shell command>` → `ShellCommandAction`,
`/<name> <args>` → `CommandAction`, anything else → `PromptAction`. Every action carries
the exact text the user typed as a **required** `raw_input`; `ui/app.py` rejects empty
input before building one, so nothing downstream has to reconstruct an approximation of
the typed line. Once either shell
pattern matches, the entire remainder is passed unchanged to `ShellCommandTool`; slash
commands and `@` mentions inside it are not parsed. Both forms yield the same synthetic
tool call/result pair without consulting EXECUTE permissions or requiring a configured
model. `!` appends both messages to agent history for the next model prompt, while `!!`
appends neither. The stored `!` pair is preceded by a history-only `UserMessage` stating
`User explicitly ran the following command:`, preventing the model from treating the synthetic tool
call as its own request.
`ActionExecutor` runs a command locally (looked up in `commands/registry.py:COMMAND_REGISTRY`)
or sends the prompt to the agent; both yield the same `Message` stream so the UI renders
them identically. **The executor holds no recording logic of its own**: it has no `record_*`
or wrapper methods, and every write is a direct
`self._session_manager.session_recorder.<method>(...)` call — the recorder is reached through the
manager at the point of use, never cached. `_execute_prompt` records the user message it
just built, then wraps `agent(...)` in `session_recorder.record_stream` and yields the
whole stream. What to skip lives in `SessionRecorder.message`, and every recorder write
returns an `ErrorMessage | None` rather than an error string, so the executor has nothing
left to wrap. The executor has no session-effect mapping of its own: which command changes
what session state is each command's own knowledge, not the executor's (see **Sessions**
below for `/cd`, `/provider`, `/model`, `/effort` and `/permissions`). Two asymmetries: the
initial user message is recorded before the loop rather than from it, since it must replay
on restore but `ui/app.py:145` already mounted the typed text, so yielding it would render
it twice; and the `!` attribution line is recorded with `visible=False`, the one write that
asks to stay out of the replayed log while staying in model history.

A command's plain-string result is wrapped into an `AssistantMessage` by
the executor. A slash command subclasses `Command`, sets `name`/`description`,
implements a typed `execute(...)`; `run` receives the raw prompt words plus the
injectable objects, binds the words in order to `execute`'s parameters and coerces them to
their annotations — the executor never reads `execute`'s signature. Words beyond the
declared parameters are dropped, unless the command sets `joins_arguments = True`
(default `False`): then its last parameter receives every remaining word joined with a
space, which is how `/rename my new name` gets the whole name and `/resume my new name`
searches for all of it. A command that changes the active session's *saved* state subclasses
`StateChangeCommand` (`commands/state_change.py`) instead — an abstract layer between the
two that carries the two helpers such a command needs (see **Sessions** below); the
executor and the registry only ever type on `Command`, so nothing else changes.
A parameter named after one of
`prompt/actions/constants.py:INJECTABLE_COMMAND_PARAMETERS` (currently `application`,
`settings_manager`, `agent`, `session_manager`) is supplied by the executor itself instead of being bound to a prompt
word — this is how a command performs an action against the running app or mutates
settings (e.g. `/model`), rather than the app doing it on the command's behalf. Both
command errors (`_execute_command`) and an unconfigured model on a plain prompt
(`_execute_prompt`, via `Settings.model_id`) are caught by the executor and turned into an
`ErrorMessage` rather than raising out to the UI.

**Settings & storage** (`settings/`, `storage/`) — SettingsManager is the only Settings
entry point. Its constructor requires StorageManager and `supported_providers`, without
live Agent, Client, Sessions or frontend services. Construction performs no I/O. The
manager snapshots registered categories and copies the supplied provider metadata.
Plugins must be loaded first so their categories are included.

`settings` and `global_settings` always return detached snapshots, including separate
permission dictionaries. `set_field(field, value)`, `restore_active` and `reset_to_global_settings`
change active values only and return None. `set_field` accepts any settings field, validates
before publication and merges permission changes; model selection requires a usable
model ID, effort changes require a configured model, and provider selection may precede
model selection. Application will read `settings` afterward and apply it to Core.

`save_settings(settings)` promotes the supplied snapshot; `save_field(field, category)`
promotes only the selected field or permission category. Writes happen before publishing
global defaults, and failures propagate.
`reset_to_default_settings()` creates code defaults inside SettingsManager, writes
them, then replaces both active and global snapshots. A failed write leaves both in-memory
snapshots unchanged. It returns None. `reset_to_global_settings()` performs no file I/O.
`restore_active()` accepts four plain values (provider, model name, thinking effort and
complete permissions) and overlays them on global defaults without saving.

`Settings` remains a plain holder with equality, representation and derived `model_id`.
Its provider is a plain field, checked by SettingsValidator. SettingsManager creates
defaults using its snapshot of registered categories. Built-in category levels are
declared in `core/permissions/constants.py` and also used by PermissionManager; other
categories default to ASK. The instance method
`from_dict(data, categories)` fills its receiver and returns None. Only the manager
constructs and copies Settings. `model_id` raises ValueError while provider/model is unset.

`SettingsValidator` returns checked mappings and diagnostic Messages, never Settings
objects. File loading applies defaults, reports warnings and errors, and
reports unset provider/model values. Explicit edits use the same rules with strict
rejection, allowing nullable unconfigured values. Missing file permissions use the code
defaults; explicit null means NONE. Unknown names are ignored with notices on load and rejected
for edits. Strict permission maps must cover every known category. Types are checked
before membership/conversion; booleans are not numeric limits. The filename lives in
`settings/constants.py`; scalar predicates and allowed permission levels are private
class fields on SettingsValidator. Private members use double-leading underscores.

SettingsManager calls Storage's `read_text` and `write_file` with
`ARANCIO_SETTINGS_FILENAME = "settings.yml"`; Storage adds its root. A missing file is
created from code defaults and reported. Empty, malformed or non-mapping YAML remains
untouched while fallback values are used in memory. Other I/O and UTF-8 decoding failures
propagate. `load_settings()` initializes both snapshots and returns a detached snapshot plus
file/field notices, without applying values to Core.

SettingsManager uses PyYAML safe conversion directly and preserves mapping order when
dumping settings.

`StorageManager` (`storage/manager.py`) is now generic file I/O. It uses the fixed
`ARANCIO_DEFAULT_DIR` root and exposes instance operations for text, lines, byte chunks,
sizes, discovery, writes, directory moves and symlinks. Private `__root`, `__resolve_path`
and static `__validate_directory` use double-leading underscores. `READ_CHUNK_SIZE` lives
in `storage/constants.py` and bounds byte reads at 64 KiB. Consumers must close
an abandoned byte iterator. Every write, including `make_dir`, returns None;
failures propagate as exceptions. File-size queries and file removal raise when the
file is missing.

`move_directory(source, target)` validates source, moves its direct entries with
the existing `shutil.move` behavior and removes the emptied source. It creates or
reuses target. `create_symlink(source, target)` creates the link **at source**,
pointing to target. It validates target before checking source. An existing source
link resolving to target is a no-op; other occupied sources raise FileExistsError.
The shared validator rejects symlinks with ValueError, missing paths with
FileNotFoundError and non-directories with NotADirectoryError before writes.

**Caller migration is deferred.** Startup still references the removed
`bind_litellm_login_dir` and the old SettingsManager constructor. Commands must move
from direct mutation and `apply()` to setters, explicit saves and snapshot reads.
Sessions must replace `apply_session_configuration` with value-only `restore_active`.
SessionRegistry still consumes `make_dir`'s old return value, and SessionChecksum still
uses direct I/O. No compatibility wrappers or tests are added in this step.

**Sessions** (`sessions/`) — `SessionManager` decides *which* chat is active and owns the
read side; `SessionRecorder` (`sessions/recorder.py`) owns the **write** side; `StorageManager`
does only file I/O. **A session always exists; its log does not.**
`SessionManager.__init__` ends with `create()`, so `current` is a `Session` from
construction on and no consumer handles its absence. What is deferred is persistence:
`SessionRecorder.created()` only *adds* the `session_created` header to `session.events`,
leaving it unsaved, and `flush()` — which walks every unsaved event in order and already
uses `create_file` for the first one — carries it to disk under the first write that has
something to save. An untouched chat therefore leaves no file.

Which commands must never supply that first write is named once, as
`commands/registry.py:SESSION_DISCARDING_COMMANDS` — a frozenset of the classes that
end or switch the active chat: `ExitCommand` (`/exit`, `/quit`), `ClearCommand`
(`/clear`, `/new`), `ForkCommand` and `ResumeCommand`. It is keyed by **class**, so
aliases come along for free, and it drives two rules in `ActionExecutor`: *when* the
typed line is recorded — before the command runs, since the chat it belongs to is about
to be replaced — and *whether* anything is written at all. The second rule is the single
predicate `_writes_nothing(command)`, the only place the executor reads
`created_on_disk`, and it covers every write the executor makes for such a command: the
typed line (`_record_command`), the result message (`/fork`'s confirmation, `/resume`'s
"multiple matches", "already current" and directory-fallback notices) and the error from
a command that raised or whose arguments did not bind (`/resume <unknown>`). All three
are still shown; they are only not saved. Once the chat has a log, the same command is
recorded like any other, so a `/quit` mid-chat still shows up in the replayed log. The
result-message check runs *after* the command, so `/fork`'s confirmation is weighed
against the fork, not the session it came from. `ForkCommand` is the one command that
writes on its own, and its `session_recorder.flush()` is likewise conditional on the
**source** being `created_on_disk`: forking an unsaved chat switches to the fork without
writing either of them. Two holes are deliberate: an unknown command (`/quti`) and a
malformed prompt still write their error and so create the log — they are real mistakes
the session keeps, and neither resolves to a command the set could recognize. Nothing
in `App._run_agent` orders itself
around this any more: it resolves the action and executes it, and its `except` branch (a
malformed prompt) writes the `ErrorMessage` through `session_recorder.message` directly,
which persists the header along with it. `App` doesn't
hold its own `settings_manager` reference (checked: it's
passed straight into the `ActionExecutor` it builds and only used inline once at `__init__`),
which is why the recording lives on `ActionExecutor` rather than `App`. The chain is
`ActionExecutor` → `SessionRecorder` → `SessionManager`:
the recorder holds the manager and asks it which chat is open, so a caller passes only what it
wants written (`session_recorder.message(msg)`, not `record_message(session, msg)`). `SessionManager`
has **no** `record_*` methods; it exposes three seams the recorder uses — the
`session_recorder` property, `current` (the open chat) and `after_write(session)` (registry sync).
`ActionExecutor` reaches the recorder as `session_manager.session_recorder` at each point of
use, and holds the manager itself because `/clear` receives it through
`INJECTABLE_COMMAND_PARAMETERS`.

The recorder builds each event record, applies the state change it describes, appends it to
the session and flushes it, including the recovery-offset retry after a failed write.
**`flush()` is the single funnel** — every write reaches it through `event()`, the creation
header being the one record added outside it, which then rides along with the first real
write — so it is the
one place that calls `SessionManager.after_write`, and only on success. That keeps the
registry true without any caller remembering to sync it, and keeps `_register_durable_session`
and `_update_registry_entry` private to the manager. Both are idempotent, so running them on
every successful write is cheaper than tracking which write needs which. `event()` always
flushes whichever chat `session_manager.current` reports; `flush()` itself takes an optional
`session` (defaulting to the current one) so `ForkCommand` — the one command that writes on
its own — can reach `SessionRecorder.message_into(session, ...)` to persist the same message
into a session that is **not** current (the source, once `create()` has already switched
`current` to the new fork). It is the one deliberate exception to "the recorder always asks
which chat is open," and it skips the file-state sync `message()` does,
since that concerns the open chat's tool reads, not a session that just stopped being current.

The recorder is built as
`SessionRecorder(storage_manager, self, self._tool_session)` inside
`SessionManager.__init__`, receiving a half-built manager: it only stores the reference,
and first reads it on the earliest write. `create()` and `load()` must therefore set `self._current` **before** calling
the recorder — they do. The recorder also
builds the two records nobody asks for explicitly: `created` stamps the `session_created`
header that opens a log, unsaved (the schema version lives in `sessions/constants.py` as
`SESSION_FORMAT_VERSION`, since the recorder writes it and the manager validates it on read),
and `close_interrupted_tool_calls` gives a loaded session's unanswered tool calls a synthetic
tool error. **No class other than the recorder builds a session record or writes the log** —
`grep '"type":' sessions/manager.py` should stay empty.
`SessionRecorder.message` is the gate, and it skips exactly one thing: a message the
codec has no record for. `message_to_record` returns `None` for a `ChunkMessage` — a
fragment saved once it is finalized — so the knowledge of what the schema can hold lives
with the schema. That check is load-bearing rather than defensive: a chunk inherits from
its finalized parent, so without it the `isinstance` chain below would match
`AssistantChunkMessage` as an `AssistantMessage` and save every stream fragment as a
complete reply. Everything else is written, **errors included** —
the session replays what the user saw and what the model saw, so a mistyped command whose
only output was an error still leaves a trace. An error saves as an `error` record and
comes back `visible` but out of model history, following its own flags. The single
exception never reaches the gate: `SessionRecorder._save_error` builds the "Could not save
session" notice, and the recorder never hands it to its own `message()` — the write is
what failed. Every write also syncs the file-read state: `_sync_file_states` compares the
shared `ToolSession` with what the session already saved and writes a `file_state_changed`
record for each new read, so core never has to announce one. `SessionManager` receives
that same `ToolSession`, hands it to the recorder, clears it when a chat is replaced and
restores its saved file state when a chat is resumed. Finalized messages store model-facing
`content`, user-facing `display_text`, the `in_history` flag and the `visible` value the
caller asked for; only streaming chunks are not saved.
Loading rebuilds the agent history from `in_history` entries and the UI log from `visible`
entries; `model_history` and `visible_messages` are each **one walk** over the events, and
`visible_messages` rebuilds a command's typed line from its `command` record, since a
command has no message record of its own. A saved tool call with no paired
result receives a synthetic model-facing tool error during restoration so the model knows its
outcome is unknown. Every session starts with a versioned creation record carrying its initial
name, configuration and working directory, and later `state_changed` and `file_state_changed`
records update its state.

Session names store only an explicit choice: `Session.explicit_name` defaults to `None`,
while the `name` property returns the ID when unset. Assigning `session.name` updates the
explicit value; assigning `None` clears it. Creation and state records keep the required
`"name"` field as `null` or a string, never materializing the fallback. Scans and registry
entries preserve that nullable value; registry entries expose the same `name` fallback
for search and display. No migration of older logs is performed.

The session's name, command-controlled provider, model, thinking effort, complete permission map
and working directory are restored without overwriting global `settings.yml` defaults.
`session_recorder.state_changed()` takes no arguments: it only reads the session's *current*
`configuration`/`working_directory`/`name` and persists them together as one `state_changed`
record — **mutating those fields is not the recorder's job**. Each command that changes one of
them mutates the session itself, then calls
`StateChangeCommand._persist_state_change(session_manager, confirmation)`
(`commands/state_change.py`), which calls `state_changed()` and returns either `confirmation`
or the resulting `ErrorMessage`: `CdCommand` sets `session.working_directory`
to the app's new (already-resolved) directory; `ModelCommand`, `ProviderCommand`, `EffortCommand`
and the two mutating branches of `PermissionsCommand` reach the sibling helper
`StateChangeCommand._apply_configuration`, which sets `session.configuration =
SessionConfiguration.from_settings(settings_manager.settings)` after they applied and persisted
the settings change; `RenameCommand` sets `session.name`. Those six are exactly the commands
subclassing `StateChangeCommand` rather than `Command` directly — `/clear` and `/resume`
replace the whole session through `discard_and_create`/`restore_runtime` and never write a
`state_changed` record, so they stay on `Command`. All three fields are re-sent together
regardless of which one changed, so the log carries one atomic record of "the state at this
point" rather than several independently-timed event types. Each of these commands is injected
`session_manager` (`prompt/actions/constants.py`) for exactly this; the executor itself has no
session-effect mapping — `ActionExecutor` only records the command's own confirmation text and
its `command` line, nothing about session state. `_persist_state_change` returning the
`ErrorMessage` in place of `confirmation` is why a failed write still reaches the user, from the
command itself rather than a side-channel. Because the persisted record is a complete snapshot,
the bottom-most one already *is* the session's current state on its own, so reading it back
never needs to replay a record-by-record mutation:
`SessionValidator._state_from_record` converts a `session_created`/`state_changed` record into
`(configuration, working_directory, name)` once, and both readers of that final state build on
it — `_scan_state` (replacing the old
header-only `_read_header`) walks every line of a log checking only `type`, keeping the last
`state_changed` (or the header) without validating any non-winning record's shape, so
`SessionRegistry` (built once per process at startup) reflects every session's true last-known
name, configuration and working directory, not its stale creation-time header; `read` still
walks the whole log regardless, since message history requires it, but now derives the final
`Session.configuration`/`working_directory`/`name` from `_state_from_record` once rather than
via `_apply_record` mutating them on every `state_changed` occurrence it passes — `_apply_record`'s
`state_changed` branch still validates every record's shape (so "every line is validated" still
holds for a full `read`), it just no longer supplies the final value. `/provider`, `/model`,
`/effort` and `/permissions` update both their session snapshot and the corresponding global
default field; `/rename` (`commands/rename.py`) has no global counterpart to update — a session's
name is session-only state, unlike its configuration. `_update_registry_entry`
(`manager.py`) syncs a renamed active session's own registry entry immediately, the same way it
already keeps `working_directory`/`configuration` current, so a same-process `/resume` search
finds it under its new name without waiting for a restart to re-scan the registry.
`/clear` (aliased as `/new`) discards the old session's unsaved in-memory
events, restores global settings, resets file-read safety state and starts a new session
under the current CWD. Because it replaces the open chat, the executor writes its command
line *before* running it, and keys that on `SESSION_DISCARDING_COMMANDS` rather than the
typed name, so every alias is covered — and so is `ExitCommand`, which replaces nothing,
making pre-recording harmless: the line lands in the same still-current chat it would have
anyway.

`/resume <id-or-name>` is `restore_runtime`'s first caller. `SessionRegistry.find`
(`registry.py`) resolves the query: an exact ID match wins outright and is returned alone
even when it's ambiguous (a colliding ID marks both copies `damaged`, so that ambiguity
surfaces the same way a multi-name match does); otherwise every entry whose name contains
the query case-insensitively. Zero matches is an error, more than one is reported as a
listing rather than guessed at. A single match resumes across directories: `ResumeCommand`
calls `restore_runtime` for the session's history/configuration, then unconditionally calls
`App.set_working_directory` with the directory `restore_runtime` returns (the session's own,
or the launch directory as a fallback when the saved one no longer exists on disk) — even
when it's already the current directory, so nothing downstream has to special-case a no-op
move. `set_working_directory` mirrors that move to the process via `os.chdir`, same as
`/cd`. `ResumeCommand` then clears and repopulates
the log (`App.populate_log`, factored out of `on_mount` for this reuse), and refreshes the
toolbar's model/effort display, returning `None` on success — the repopulated log is the
confirmation, the same way `/clear`'s effect is only ever seen, never announced in-band.

`/fork` (`commands/fork.py`) is `create()`'s second caller after plain session creation:
`create()` gained `explicit_name`/`forked_from` parameters so it stays the one place a session's
identity and header are built, but it only builds an empty session — `ForkCommand` itself decides
what a fork carries over, the same way `ClearCommand` (not `discard_and_create`) decides to reset
the agent and UI. `ForkCommand` calls `create()` with the source session's `working_directory`, a
deep-copied `configuration`, its `explicit_name` and `forked_from=source.id`, then copies every
event but the header from `source.events` onto the new session and its `file_states`, and flushes
once through `session_manager.session_recorder`. The new header's `forked_from` is `None` on every
non-forked session (`SessionValidator.read` reads it directly off the header, bypassing
`_state_from_record` since it isn't part of a `state_changed` snapshot). Unlike `/clear`, nothing
in memory is discarded — agent history and the `ToolSession` guard survive untouched, since
forking duplicates the conversation's persisted identity rather than
starting a new one. Unlike `/resume`, there is no `clear_log`/`populate_log`/toolbar refresh,
since nothing rendered changes. `ForkCommand` is in `SESSION_DISCARDING_COMMANDS` for the
same reason as the others: it too replaces `session_manager.current`.

A named source that is not itself already a fork (`forked_from is None`) is renamed to
`{base_name}:main` before the switch, so its lineage stays visible; the new child is always
named `{base_name}:fork_<YYYYMMDDHHMMSS>`. `base_name` strips any trailing `:main` or
`:fork_<14 digits>` off the source's current name first, so re-forking a fork chains off the
original name instead of nesting suffixes, and `SessionRegistry.find`'s existing substring
match on `entry.name` already resolves a query for the shared base to both sessions — no
change needed there. An unnamed source is untouched, and the confirmation keeps using ids
instead of names, exactly as before this convention existed. The rename, when it happens, is
persisted via the ordinary `session_recorder.state_changed()` while `source` is still current
— a named session is always already `created_on_disk` by construction (naming only ever
happens through `/rename`, which flushes immediately, or through a previous fork, which
flushes its child right away too), so this is never a session's first write. The events
copied onto the fork are snapshotted **before** that rename, so the rename's own
`state_changed` record stays source-only and never shows up in the fork's replayed history.
The confirmation itself is recorded into both logs: the executor's generic post-command write
puts it into the fork (current by the time the executor gets to it), and `ForkCommand` puts
the same text into `source` directly via `session_recorder.message_into` (see **Sessions**
above), under the same `if source.created_on_disk:` guard that already decides whether the
fork gets a log at all.

Session events are kept in memory with a runtime-only `saved` flag. Failed writes leave events
unsaved and remember the starting byte offset; a later save truncates the unconfirmed tail and
retries it. If the process exits or `/clear` discards the session first, a partial JSONL line
remains and the next scan marks that session `damaged`, with a `damage_reason`; damaged entries
remain discoverable but cannot be restored. Successful Read/Edit/Write results record canonical
path/mtime state so a resumed session restores its read-first guard only for unchanged files.

## Conventions

- **Docstrings**: Google style, required on all public symbols (`pre-push` enforces `D,DOC`).
- **Code style**: PEP 8 via `ruff` (`E,F,I`); import sorting on. Format with `ruff format`.
- **Comments**: start lowercase, no trailing period; only comment non-obvious slices.
- **Files**: `snake_case.py`; Markdown `UPPER_SNAKE_CASE.md`. Source in `src/`, tests in `tests/`.
- **Exceptions**: prefer built-ins (`ValueError`, `TypeError`, …); do not define custom exception classes.
- **Settings internals**: double-leading underscores for private fields and methods;
  shared constants belong in `settings/constants.py`; validator-only rules are private
  class fields. Only SettingsManager constructs Settings.
- **Storage internals**: double-leading underscores for private fields and methods;
  constants belong in `storage/constants.py`. Callers own paths and serialization.
- **New provider / tool / command**: follow the existing seams — subclass the relevant
  `base.py`, and for a tool also add its `harness/tools/<snake_name>/` prompt pair; for a
  command register it in `commands/registry.py`, and subclass
  `commands/state_change.py:StateChangeCommand` instead when it changes the session's saved
  configuration, working directory or name. A command that ends or switches the active
  session also goes into `commands/registry.py:SESSION_DISCARDING_COMMANDS`.
- **New plugin**: import `plugin`, `hook` and `tool` from `core/plugins/decorators.py`.
  Mark a plain class with `@plugin(...)` in `plugin.py`; define `@hook(...)` methods,
  `@tool(...)` methods, or both. Each decorated class is independent;
  constructors take no required arguments. Do not add a base class or YAML manifest.
- **Git**: branches `feature/snake_case` or `fix/snake_case`; commit messages in past tense
  naming the file(s) touched. Do not mention the contribution of coding agents (including
  Claude) in commit messages — attribute commits to the human author only.

### Testing

Only `pytest` is available — no `pytest-mock`, `pytest-asyncio` or coverage plugin. Use
`monkeypatch`, `tmp_path` and the stdlib.

- **Fake only the four real boundaries**: the model (`ScriptedClient`), the user
  (`ScriptedController`), the Textual app (`FakeApp`, `RecordingApp`) and the network
  (monkeypatched `trafilatura`/`ddgs`). They all live in `tests/fakes.py`; everything
  else — storage, sessions, settings, permissions, tools, the executor — is the real
  object against `tmp_path`. The pre-refactor suite faked whole layers instead, which is
  how it drifted far enough from the source to shape it.
- **Plugins are written to disk, not faked.** `tests/test_plugins.py` builds real plugin
  folders under `tmp_path` and loads them through `PluginManager`. The autouse plugin
  fixture clears the registry, pending declarations and diagnostics around every test,
  and removes synthetic plugin modules afterward so helpers cannot leak between tests.
  Tests describe the current registration and runtime contract, not removed internals.
- **Never let a test reach the real home directory.** See the table under *Runtime
  working directory*. When tests are restored, rebuild the isolation fixtures. A test
  constructing `StorageManager` must repoint `storage.manager.ARANCIO_DEFAULT_DIR`;
  `SessionManager` must receive `root=tmp_path`, and generic directory move/link calls
  must also use explicit temporary source and target paths.
- **`shared_session` is a process-wide singleton.** An autouse fixture clears it around
  every test; a test asserting on it in isolation should inject its own `ToolSession()`.
- **Docstrings are enforced on `tests/` too.** A one-line docstring is enough for a test
  function — a fixture parameter needs no `Args:` entry. Two `DOC` rules bite: `DOC501`
  wants a `Raises:` for a literal `raise ValueError(...)`, while `DOC502` **rejects** one
  for `raise some_variable`, whose type ruff cannot resolve.
- **Pin a known defect with `@pytest.mark.xfail(strict=True)`** and a `reason` naming the
  cause, rather than asserting the wrong behaviour. The test turns green by itself once
  the bug is fixed, and `strict` then fails the suite until the marker is removed, so the
  fix and the marker go in together. The suite currently has no xfail.
