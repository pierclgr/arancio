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

The repo's `harness/` is the **source** for those prompt files; there is no auto-copy, so
editing `harness/` in the repo has no effect on a real run until the files are placed
under `~/.arancio/harness/`. **Tests** must sidestep this by repointing
`tools_base.TOOLS_HARNESS_PATH` and
`system_prompt_builder.SYSTEM_PROMPT_HARNESS_PATH` at the repo's `harness/` before any
test imports, or they read the real home directory. `tests/conftest.py` does this at
import time, because importing `prompt/actions/executor.py` builds two tools as class
attributes.

**A `StorageManager`'s `root` does not sandbox everything it writes.** Three absolute
module constants are read at call time and ignore it entirely, so anything exercising
those paths must repoint the constant on the *importing* module — patching
`core.constants.path` or `settings.constants` is too late, the name is already bound:

| Constant, as imported | Read by | What an unpatched test does |
| --- | --- | --- |
| `storage.manager.ARANCIO_SETTINGS_FILE` | `save_settings` / `load_settings` | overwrites the real `~/.arancio/settings.yml` |
| `storage.manager.ARANCIO_LITELLM_DIR` | `bind_litellm_login_dir` | moves the real `~/.config/litellm` — the user's ChatGPT login — and replaces it with a symlink |
| `sessions.manager` `root` default | `SessionManager.__init__` | scans and writes the user's real session history |

`conftest.py` redirects the first two in an **autouse** fixture so no test can opt out by
forgetting; the third is covered by always passing `root=tmp_path`.

## Architecture

Layered and provider-agnostic. The recurring pattern is a `base.py` abstract class plus a
provider file (e.g. `litellm.py`); adding a provider means a new `BaseClient` subclass
with its own builder/parser trio, not touching the core loop.

**Wiring** (`ui/__main__.py:main`) — build order matters: `StorageManager` →
`UIController` → two `LiteLLMClient`s (main *streaming*, summary *non-streaming*, thinking
disabled) → `ToolManager` → `PermissionManager` → `Agent` → `SettingsManager.load()`
(neither `PermissionManager` nor `Agent` receives the `SessionManager`: core does not save)
(validates `settings.yml` and applies the result to the live objects, returning any
fallback messages) → `SessionManager` → Textual `App`, which also receives the
`SettingsManager` so
settings-mutating commands (e.g. `/model`) can apply and persist their changes, and the
`load()` messages as `startup_messages`, rendered once in `on_mount`. The controller is
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
owns no session. `run()` yields `Iterator[Message]`: `_emit` appends a message to history
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
`_result_parser`. The tool's **name is its class name**, and its description/input schema
are loaded from the harness dir (see runtime section); descriptions are dynamic-markdown
(`<field>`, `<include>`, `<script>` tags expand against the tool instance). The module-level
`ToolSession` singleton `shared_session` records which files were read so write/edit tools
can refuse blind overwrites of unread or externally-modified files. Every tool reaches it
through the `BaseTool._session` **class** attribute, so the guard outlives the tool
instances that `ToolManager.create_tools` rebuilds whenever permissions change; there is
no per-instance override. Families: `files/` (read, write, edit),
`commands/` (shell), `web/` (search, fetch — fetch takes an
injected summary client).

**Permissions** (`core/permissions/`) — `PermissionCategory` (READ/WRITE/WEB/EXECUTE) maps
each member to a **frozenset of tool classes** (not names → refactor-safe);
`PermissionLevel` is NONE, ASK or AUTO. The grants dict always covers every category — a
category not granted is present at `NONE`, never absent. `PermissionManager` owns the
`ToolManager`, so tools in a NONE category are **never even instantiated**; the
`allowed_tools` property is what builds them. `validate` runs
AUTO calls silently and delegates ASK calls to the controller; denials and user notes are
fed back to the model as messages rather than raising. It returns a plain
`PermissionDecision` (`core/permissions/types.py`): a `PermissionOutcome`
(ALLOWED / DENIED / UNAVAILABLE) plus the user's `note`. It builds **no** messages — the
agent's `_permission_message` turns the decision into the `ToolErrorMessage` (worded
differently for a denial and for a missing tool) or the report-then-answer `UserMessage`,
so model-facing wording stays in the one place that talks to the model. How a call was
authorized is **not** recorded — the session keeps only what a resumed run needs, and the
`ToolResultMessage`/`ToolErrorMessage` already say whether the tool ran.

**Controller port** (`core/controllers/`) — the core↔UI seam (ports & adapters). Core
sends a `BaseControllerRequest` (e.g. `PermissionRequest`) and gets a
`BaseControllerResponse` (e.g. `PermissionResponse` carrying a `Decision`), dispatched by
`isinstance`. `ui/controller.py:UIController` is the Textual adapter: it blocks the
**worker** thread on a `threading.Event` until the user answers a `QuestionScreen`, never
the UI thread. Its `app` is genuinely `None` until wiring finishes — the app needs the
agent, which needs the permission manager, which needs the controller — so every dispatch
goes through `require_app()`, which raises rather than letting each call site assume the
attachment happened. `LiteLLMClient` **requires** its controller: both production clients
get one, and the login notice has nowhere to go without it.

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
just built, then wraps `agent.run(...)` in `session_recorder.record_stream` and yields the
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
until fixed. Each `SettingsValidator._FIELD_RULES` predicate composes generic single-purpose
checks from `settings/utils/validations.py` (`is_none`, `is_str`, `is_int`, `is_number`,
`is_positive`, `is_negative`, `is_known_provider`) with the and/or a field actually needs;
`Settings.provider`'s setter reuses `is_known_provider` too, instead of its own membership
check. An invalid provider assignment raises an error containing the alphabetically sorted
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
