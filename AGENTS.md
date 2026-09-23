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
working directory `~/.arancio/` (base paths in `core/constants/path.py`, LiteLLM paths
in `storage/constants.py`):

- `~/.arancio/harness/tools/<snake_tool_name>/{description.md,input_schema.yml}` — each
  tool's description and JSON input schema, loaded from disk by `BaseTool.__init__`.
- `~/.arancio/harness/SYSTEM_PROMPT.md` — the agent's system prompt.
  `SystemPromptBuilder` seeds it with a default when absent, parses it as dynamic
  markdown once, and caches the text; `build()` (called once per agent turn) returns
  that cached text with the current date appended. Runtime context that must stay
  fresh belongs in `build()`, **not** in a `<script>` tag in the markdown — the file is
  parsed once, so a date computed there would freeze at startup. Settings use the
  dedicated `StorageManager.save_settings`.
- `~/.arancio/settings.yml` — persisted `Settings`.
- `~/.arancio/litellm/` — LiteLLM's config dir is symlinked here (LiteLLM hardcodes
  `~/.config/litellm`; `StorageManager.bind_litellm_login_dir` redirects it).
- `~/.arancio/sessions/<creation-cwd>/<year>/<month>/<day>/<id>.jsonl` — one
  append-only chat session per file. The readable creation-CWD directory has a hash suffix
  to avoid collisions; session IDs are UUID4 hex strings. There is no persisted registry:
  `SessionRegistry` is rebuilt by scanning these files at launch.
- `~/.arancio/plugins/<plugin_name>/{manifest.yml,module.py}` — one plugin per folder,
  scanned at launch by `PluginManager` (see **Plugins** below). Like the harness, the
  user places these here by hand: nothing creates the directory, and a missing one just
  means no plugins. A folder without a `manifest.yml` is not a plugin.

The repo's `harness/` is the **source** for those prompt files; there is no auto-copy, so
editing `harness/` in the repo has no effect on a real run until the files are placed
under `~/.arancio/harness/`. **Tests** must sidestep this by repointing
`tools_base.TOOLS_HARNESS_PATH` and
`system_prompt_builder.SYSTEM_PROMPT_HARNESS_PATH` at the repo's `harness/` before any
test imports, or they read the real home directory. `tests/conftest.py` does this at
import time, so the constants are already redirected before any test builds a tool —
`BaseTool.__init__` reads the path at construction and raises when the directory is
missing.

**A `StorageManager`'s `root` does not sandbox everything it writes.** Four absolute
module constants are read at call time and ignore it entirely, so anything exercising
those paths must repoint the constant on the *importing* module — patching
`core.constants.path` or `settings.constants` is too late, the name is already bound:

| Constant, as imported | Read by | What an unpatched test does |
| --- | --- | --- |
| `storage.manager.ARANCIO_SETTINGS_FILE` | `save_settings` / `load_settings` | overwrites the real `~/.arancio/settings.yml` |
| `storage.manager.ARANCIO_LITELLM_DIR` | `bind_litellm_login_dir` | moves the real `~/.config/litellm` — the user's ChatGPT login — and replaces it with a symlink |
| `core.plugins.manager.PLUGINS_PATH` | `PluginManager.__init__` | imports and runs the user's real plugins |
| `sessions.manager` `root` default | `SessionManager.__init__` | scans and writes the user's real session history |

`conftest.py` redirects the first three in an **autouse** fixture so no test can opt out
by forgetting; the fourth is covered by always passing `root=tmp_path`. The plugins one
never *writes*, which is why it is not a `StorageManager` concern at all — it reads
`~/.arancio/` exactly like `tools_base.TOOLS_HARNESS_PATH` does.

## Architecture

Layered and provider-agnostic. The recurring pattern is a `base.py` abstract class plus a
provider file (e.g. `litellm.py`); adding a provider means a new `BaseClient` subclass
with its own builder/parser trio, not touching the core loop.

**Wiring** (`ui/__main__.py:main`) — build order matters: `StorageManager` →
`UIController` + `HookManager` → `PluginManager.load()` → two `LiteLLMClient`s (main
*streaming*, summary *non-streaming*, thinking
disabled) → `ToolManager` → `PermissionManager` → `Agent` → `SettingsManager.load()`
(neither `PermissionManager` nor `Agent` receives the `SessionManager`: core does not save)
(validates `settings.yml` and applies the result to the live objects, returning any
fallback messages) → `SessionManager` → Textual `App`, which also receives the
`SettingsManager` so
settings-mutating commands (e.g. `/model`) can apply and persist their changes, and the
plugin plus `load()` messages as `startup_messages`, rendered once in `on_mount`. The
plugins load right after the hook manager they attach to, and **before** `ToolManager`/
`PermissionManager` are even constructed and `SettingsManager.load()` applies the
permission grants and rebuilds the tool catalog — the point any plugin-defined tool
(see **Plugins** below) must already be registered by, since `PermissionManager`'s and
`SettingsValidator`'s default-grants loops both iterate `PermissionCategory` once, early.
The controller is
created before the app because app→agent→permission-manager→controller; `controller.app`
is assigned once the app exists. The `SessionManager` comes **after** `SettingsManager.load()`
because its constructor opens the process's session and that session snapshots the loaded
settings — built earlier, its header would record the defaults. Its `current` is therefore
never `None`; what is deferred is the *log*, which reaches disk on the first write that has
something to save (see **Sessions** below), so closing the app without ever typing anything
— or having typed only `/exit` (or its `/quit` alias) — leaves no session file behind.

**Messages** (`core/messages.py`) — the lingua franca between clients, agent and UI. Every
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

**Agent loop** (`core/agents.py`) — holds a flat `List[Message]` history, empty unless the
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
`BaseClient` declares strategy class attrs `_request_builder` / `_payload_builder` /
`_response_parser`. Flow: `build_request` → normalized frozen `BaseRequest` dataclass
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

**Tools** (`core/tools/`) — `BaseTool` subclasses implement `_call`; the public `call`
wraps success/exception into a `ToolResultMessage`/`ToolErrorMessage` through the tool's
`_result_parser`. It returns `(result, hook_messages)`, keeping plugin errors separate
from the tool outcome. The tool's **name is its class name**, and its description/input schema
are loaded from the harness dir (see runtime section); descriptions are dynamic-markdown
(`<field>`, `<include>`, `<script>` tags expand against the tool instance). The module-level
`ToolSession` singleton `shared_session` records which files were read so write/edit tools
can refuse blind overwrites of unread or externally-modified files. Every tool reaches it
through the `BaseTool._session` **class** attribute, so the guard outlives the tool
instances that `ToolManager.create_tools` rebuilds whenever permissions change; there is
no per-instance override. Families: `files/` (read, write, edit),
`commands/` (shell), `web/` (search, fetch — fetch takes an
injected summary client).

**Permissions** (`core/permissions/`) — `PermissionCategory` groups tool classes sharing a
capability: the built-in `READ`/`WRITE`/`WEB`/`EXECUTE`/`PLUGIN`, plus whatever per-plugin
categories `@tool(category="CUSTOM")` creates at plugin-load time (see **Plugins** below).
It is **not** an `enum.Enum` — a category's tool set has to stay open for a plugin tool
to join it, or for a wholly new category to appear, after the built-ins already exist,
which a closed `Enum` cannot do. A small `_PermissionCategoryMeta` metaclass gives it the
same class-level surface an `Enum` would (`for category in PermissionCategory`,
`PermissionCategory[name]`, `PermissionCategory.__members__`), backed by a `_registry`
dict keyed by uppercased name so lookup stays case-insensitive; `get(name)`/
`get_or_create(name)` are the non-raising, idempotent ways to read or register a category,
and `add_tool(tool_cls)` grows its mutable `.tools` set (not names → still refactor-safe).
`PermissionLevel` is NONE, ASK or AUTO. The grants dict always covers every *currently
registered* category — a category not granted is present at `NONE`, never absent — which
is why `PluginManager.load()` (see **Plugins**) has to run, and does run, before any
`PermissionManager`/`SettingsManager` is built: both seed their default grants by
iterating `PermissionCategory`, so a category registered afterward would be invisible to
them. `PermissionManager` owns the
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

**Hooks** (`core/hooks/`) — `Hook` (`core/hooks/types.py`) is a string-valued enum naming
every dispatch point: `agent_start`/`agent_end`, `turn_start`/`turn_end`,
`system_prompt_build`, `before_model_request`/`after_model_response`,
`message_received`, `before_tool_call`/`after_tool_call`,
`before_permission_check`/`after_permission_check`, and a single `error`. `HookManager`
(`core/hooks/manager.py`) owns its own registrations — there is no global singleton, so
two managers never share handlers. `register(hook, handler)` appends a handler for a hook;
registering the same callable twice adds two invocations. `run(hook, **kwargs)` calls every
handler registered for that hook synchronously, on the caller's thread, in registration
order, forwarding the keyword arguments unchanged (including object identity) to each one.
Returned `Message` objects are collected into a list in dispatch order; other return
values are ignored. An exception raised by an ordinary handler propagates immediately,
so handlers after it do not run. Dispatch iterates a snapshot of the handler list taken when `run` starts, so
a handler that registers another handler mid-dispatch only affects the next call to `run`.
A hook with no registered handlers returns an empty list. Both `core/hooks/__init__.py` and every
sibling `core` package's `__init__.py` are empty; a caller imports `Hook` from
`core.hooks.types` and `HookManager` from `core.hooks.manager` directly, the same as any
other two-file `types.py`/`manager.py` package (`core/permissions/`, `core/controllers/`).

Threading follows the same explicit-constructor-injection pattern as
`controller`/`tool_manager` — never the class-attribute singleton `BaseTool._session`
uses, since two callers must be able to hold independent hook managers — stored as
`self._hook_manager`. Every constructor that takes `hook_manager` (`Agent`,
`PermissionManager`, `ToolManager`, `BaseTool`/`FetchWebTool`, `ActionExecutor`) requires
it, with no `None` default: production code (`ui/__main__.py`) never has a reason to omit
it for any of them, so requiring it catches a caller that forgot to thread the shared
instance through, rather than silently falling back to a private no-op manager.
`ui/__main__.py:main` builds **one** `HookManager()` alongside `controller` and passes the
same instance into `ToolManager`, `PermissionManager`, `Agent` and `App` (which forwards it
to the `ActionExecutor` it builds); `ToolManager.create_tools` re-threads it into every tool
it builds, so one handler registered anywhere in that graph sees every dispatch from every
component — including the `!`/`!!` shell path and `@mention` reads, which run through
`ActionExecutor`'s own `ReadFileTool`/`ShellCommandTool` instances (built once in its
`__init__`, not as module-import-time class attributes, precisely so the shared instance can
reach them).
`BaseClient`/`LiteLLMClient` never receive one: `before_model_request` /
`after_model_response` / the model-sourced `error` are dispatched by `Agent.__call__`
around its calls to `self._client.build_request`/`send_request`, not by the client itself —
placing them inside `LiteLLMClient` would never fire for `ScriptedClient`, which overrides
`send_request` outright and never calls into `LiteLLMClient` code. The same reasoning means
`FetchWebTool._call`'s own internal summarization request (it calls
`self._client.build_request`/`send_request` directly, outside any `Agent` loop) is
invisible to those hooks too — only the `before_tool_call`/`after_tool_call`/`error`
wrapped around the whole `FetchWebTool.call` see it.

Dispatch sites, by hook:
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
  the prompt and before `build_request`, inside the same per-turn `try`.
- `BaseTool.call`: `before_tool_call` (`name`, `call_id`, `arguments`) before running;
  `error` with `source="tool"` (`name`, `call_id`, `arguments`, `error`) when `_call`
  raises; `after_tool_call` (`name`, `call_id`, `arguments`, `result`) once the result
  message is built, unconditionally — both the `source="tool"` `error` and
  `after_tool_call` fire on a failed call. Only `_call`'s own exceptions are caught by
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
rather than three separately named ones, distinguished by a `source` kwarg — a handler that
wants every failure registers once for `error` and reads `source`; one that only cares about
tool failures filters on `kwargs["source"] == "tool"`.

**Plugins** (`core/plugins/`) — a plugin is a folder under `~/.arancio/plugins/` holding a
`manifest.yml` and a `module.py`, plus whatever else its own code needs. It lives **in
core**, and `core` imports nothing outside `core` — `grep -rn "from arancio\." src/arancio/core | grep -v arancio.core`
should stay empty. One thing follows from that and is load-bearing: the plugins
directory is a module constant rather than a `StorageManager` call (below). Only
`ui/__main__.py` imports `core.plugins`. A future command plugin will need care here,
since attaching it means reaching `commands/registry.py`, which core may not import;
`PluginManager` will have to take a registrar from outside instead.

`manifest.yml` is the **marker** that makes a folder a plugin, the way `pyproject.toml`
marks a project — a folder without one is not a broken plugin, it is not a plugin, and is
skipped in silence. Its fields (`name`, `version`, `description`, `author`, `enabled`) are
metadata plus the one switch; every one has a non-`None` default, so an empty file is
valid and **no manifest field can ever produce an error**. `PluginManifestValidator`
(`core/plugins/manifest.py`) copies `SettingsValidator`'s policy exactly — nothing raises, a
missing field is silently defaulted, an invalid one falls back with a `WarningMessage`, an
unknown key is dropped with one — differing only in taking the file's path as an argument,
since the message has to read `plugins/<folder>/manifest.yml:` rather than `settings.yml:`.
The **folder name is the plugin's identity** (it names the import package and every
message); the manifest's `name` is only a display name defaulting to it, so the two cannot
desync.

`module.py` defines **exactly one** plugin class; zero or several is an error. Every
plugin extends `Plugin` (`core/plugins/base.py`), which supplies `manifest` and
`directory` so a plugin reaches the extra files it ships without the loader knowing they
exist. A plugin participates two independent ways: it binds to hook entry points via
`hooks: ClassVar[frozenset[Hook]]` and overrides `execute(hook=..., **kwargs)` to run on
each one — the base `execute` is a no-op default, not `@abstractmethod`, so a tools-only
plugin never has to override it — and/or it defines tools by decorating instance methods
with `@tool(...)` (below). `hooks` lives in code rather than in the manifest — there is
deliberately no `kind:` field, since a manifest field can contradict the code while the
code cannot — and a plugin declaring **neither hooks nor tools** is reported and skipped
since it could never run.

`@tool(...)` (`core/plugins/tool.py`) marks a `Plugin` instance method as a real tool:
`description`/`input_schema` are given directly as decorator keyword arguments — a plugin
cannot ship harness files, so this bypasses `BaseTool`'s on-disk
`~/.arancio/harness/tools/<name>/` requirement entirely, unlike a built-in tool — and
`category` picks the `PermissionCategory` (see **Permissions** above) it falls under: an
existing one (`READ`/`WRITE`/`WEB`/`EXECUTE`), the fixed `PLUGIN` category (the default
when omitted), or the keyword `CUSTOM`, which resolves to a category dedicated to the
owning plugin, `PLUGIN:<PluginClassName>`, created on first use and shared by every
`CUSTOM` tool that plugin defines. Every category defaults to `PermissionLevel.ASK` the
same way regardless of how it came to exist, since nothing about `CUSTOM` is
special-cased in the default-grants loops. `find_tool_specs(plugin_cls)` finds the
decorated methods — only those defined directly on the class, mirroring
`_find_plugin_class`'s own rule — and `PluginManager._register` resolves each one's
category (an unknown name is one `ErrorMessage` that skips only that tool, not the whole
plugin) and hands it to `build_plugin_tool_class`, which synthesizes a fresh `PluginTool`
subclass per method via `type()`, with the plugin instance, the underlying function and
the spec baked in as class attributes. The function is wrapped in `staticmethod` there —
otherwise `self._method` would be auto-bound by Python's descriptor protocol to the
*tool* instance instead of staying the plain function called as
`method(plugin, **kwargs)`, silently double-passing the plugin instance as the method's
first argument. Binding it this way keeps `PluginTool` constructible as
`tool_cls(hook_manager=...)` like any built-in tool, so `ToolManager.create_tools` needs
no plugin-specific branch.

`PluginLoader.load(directory)` (`core/plugins/loader.py`) returns
`(Plugin | None, list[Message])` and **never raises**: a manifest that cannot be read,
is not YAML or is not a mapping; a missing `module.py` or one that raises on import; a
module with no plugin class or several; a plugin declaring neither hooks nor tools — each
is one `ErrorMessage` that skips only that folder. `module.py` is imported as
`arancio_plugins.<folder>.module`, under a synthetic parent package whose `__path__` is
the plugin folder, so `from .helpers import X` works inside it and two plugins each
shipping a `helpers.py` never collide — `sys.path` is left alone. Class discovery only
accepts a concrete `Plugin` subclass whose `__module__` is that module, so the base
the plugin imports at the top of its own file is not mistaken for the plugin.

`PluginManager` (`core/plugins/manager.py`) discovers, loads and registers in one
`load() -> list[Message]` pass, mirroring `SettingsManager.load()`; `main()` concatenates
its messages with the settings ones into `startup_messages`, so plugin problems render
through the existing `.warning`/`.error` path with no new UI code. Discovery walks the
**immediate** subdirectories only, skipping names starting with `.` or `_`. It takes no
`StorageManager`: exactly like the tool harness, the directory is the absolute module
constant `PLUGINS_PATH` (`core/constants/path.py`, beside `TOOLS_HARNESS_PATH` and the
two `PLUGIN_*_FILENAME`s), read in `__init__` so it can be repointed. Nothing creates it
— the user places their folders there by hand, and a missing directory simply means no
plugins. **That makes it a fourth root-unaware absolute path**: a test must repoint
`core.plugins.manager.PLUGINS_PATH` on the *importing* module, the same rule as
`tools_base.TOOLS_HARNESS_PATH`, or it loads the user's real plugins. `conftest.py`'s
autouse `_isolate_arancio_home` does this, so no test can forget.

A plugin is registered as a `functools.partial` binding its hook to
`PluginManager._run_plugin`, so `execute(hook=..., **kwargs)` knows which hook fired.
The wrapper catches plugin exceptions, disables that plugin across all its hooks for
the rest of the process, and returns one `ErrorMessage`. Later plugins still run;
ordinary handler exceptions retain the hook manager's propagation behavior.

`HookManager.run()` collects returned messages. Tools and permissions return those
messages alongside their result or decision; the agent yields them before the normal
outcome. The executor does the same for shell commands and includes mention errors in
the prelude. No pending queue or controller notification is involved. Runtime plugin
errors are displayed and saved once, but excluded from model history. They do not
trigger agent retries. Load-time failures still travel as `startup_messages`.

**Controller port** (`core/controllers/`) — the core↔UI seam (ports & adapters). Core
sends a `BaseControllerRequest` (e.g. `PermissionRequest`) and gets a
`BaseControllerResponse` (e.g. `PermissionResponse` carrying a `Decision`), dispatched by
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
the executor. A slash command subclasses `BaseCommand`, sets `name`/`description`,
implements a typed `execute(...)`; `run` coerces the prompt words to `execute`'s parameter
annotations. A command that changes the active session's *saved* state subclasses
`StateChangeCommand` (`commands/state_change.py`) instead — an abstract layer between the
two that carries the two helpers such a command needs (see **Sessions** below); the
executor and the registry only ever type on `BaseCommand`, so nothing else changes.
A parameter named after one of
`prompt/actions/constants.py:INJECTABLE_COMMAND_PARAMETERS` (currently `application`,
`settings_manager`, `agent`, `session_manager`) is supplied by the executor itself instead of being bound to a prompt
word — this is how a command performs an action against the running app or mutates
settings (e.g. `/model`), rather than the app doing it on the command's behalf. Both
command errors (`_execute_command`) and an unconfigured model on a plain prompt
(`_execute_prompt`, via `Settings.model_id`) are caught by the executor and turned into an
`ErrorMessage` rather than raising out to the UI.

**Settings & storage** (`settings/`, `storage/`) — `Settings` is a plain class (not a
`dataclass`: `provider` needs a validating `@property`, which can't share a name with a
dataclass field) with `to_dict`/`from_dict`/`default`. The model is stored as separate
`provider`/`model_name` fields; `Settings.model_id` joins them, raising `ValueError`
(separately) when either is unset — `ModelCommand` checks `provider` itself before
mutating anything, and `ActionExecutor._execute_prompt` checks `model_id` before sending a
message, so both surface as an `ErrorMessage` instead of a crash.

`StorageManager.load_settings()` (`storage/manager.py`) does *file-level* I/O only: it
returns `(dict | None, list[Message])`. A missing `settings.yml` builds `Settings.default()`
and writes it to disk (first run), but returns the dictionary as `None` and reports one
`WarningMessage` — there is no file content to hand off, and deciding what "missing" should
fall back to is not this method's call. A file that exists but is empty, has a YAML syntax
error, or whose top level isn't a mapping likewise returns `None` with one `ErrorMessage`,
leaving the file untouched. Otherwise the parsed dict is returned as-is, unvalidated.
`SettingsManager.load()` (`settings/manager.py`) is the one that turns that `None` into
`Settings.default()`; when the dict is not `None` it instead calls
`SettingsValidator.validate(data)` (`settings/validator.py`), which does *field-level*
validation: a missing field silently takes its default, an invalid field falls back to its
default and is reported as a `WarningMessage`, or an `ErrorMessage` when the field's default
is `None`. `provider`/`model_name` are the only fields whose default is `None`, so `null` is
never a valid value for them (unlike e.g. `thinking_effort`, where `null` is an intentional,
supported state): present-as-`null` gets its own clearer `"<field>" is not set.` message,
distinct from the generic invalid-value wording used for a genuinely wrong value (unknown
provider name, wrong type). `permissions` is validated per entry against the total
4-category map `PermissionManager` requires — an unknown category is dropped with a
`WarningMessage`, a known category with an invalid level falls back to `"ask"`, and an
explicit `null` level (`PermissionLevel.NONE`) is accepted as a valid, meaningful value. A
top-level key naming none of the known fields is likewise dropped with a `WarningMessage`
(same treatment as an unknown permission category, for consistency) rather than silently
ignored. An invalid file is never written back, so its messages resurface on every load
until fixed. Each `SettingsValidator._FIELD_RULES` predicate is an inline lambda spelling
out exactly the check its field needs (type, positivity, membership in
`LITELLM_PROVIDER_NAMES`), so no shared predicate module exists;
`Settings.provider`'s setter checks `LITELLM_PROVIDER_NAMES` membership directly. An
invalid provider assignment raises an error containing the alphabetically sorted
`LITELLM_PROVIDER_NAMES`, which `/provider` surfaces through the command executor.

`SettingsManager.load()` concatenates the file-level and field-level messages lists,
applies the result, and returns `(Settings, list[Message])`. **`settings` and
`global_settings` are never `None`**: both start as `Settings.default()` and `load()`
replaces them before anything can read them, since `main()` calls it between building the
manager and building the `App`. Callers therefore dereference them directly — `/model`,
`/provider`, `/effort`, `/permissions`, `/clear` and the executor all do — instead of each
one guarding a state that cannot occur.
`SettingsManager.apply()` never raises: it falls back to a `None` client model_id when
unconfigured (safe at startup), and keeps the summary client on the same model as the main
client — there is no separate summary model setting. `apply()` also pushes thinking, agent
loop limits and permission grants (which rebuilds the tool catalog) into the live objects.

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
subclassing `StateChangeCommand` rather than `BaseCommand` directly — `/clear` and `/resume`
replace the whole session through `discard_and_create`/`restore_runtime` and never write a
`state_changed` record, so they stay on `BaseCommand`. All three fields are re-sent together
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
- **New provider / tool / command**: follow the existing seams — subclass the relevant
  `base.py`, and for a tool also add its `harness/tools/<snake_name>/` prompt pair; for a
  command register it in `commands/registry.py`, and subclass
  `commands/state_change.py:StateChangeCommand` instead when it changes the session's saved
  configuration, working directory or name. A command that ends or switches the active
  session also goes into `commands/registry.py:SESSION_DISCARDING_COMMANDS`.
- **New plugin**: subclass `core/plugins/base.py:Plugin`, declare the `hooks` it binds to
  and implement `execute(hook=..., **kwargs)`, define tools by decorating instance
  methods with `core/plugins/tool.py:tool(...)`, or both — never add a `kind:` field to
  `manifest.yml`.
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
  folders under `tmp_path` and loads them through the real `PluginLoader`. A loaded
  plugin's module stays in `sys.modules` under `arancio_plugins.<folder>.module` for the
  rest of the session, so each test uses a **distinct folder name** — reusing one would
  hand the second test the first one's cached submodules.
- **Never let a test reach the real home directory.** See the table under *Runtime
  working directory*. `conftest.py`'s autouse fixtures cover it; a test that builds its
  own `StorageManager`/`SessionManager` still has to pass `root=tmp_path` and an explicit
  `litellm_config_dir`.
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
