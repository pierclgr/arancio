# Arancio

Arancio ("another coding agent") is a terminal-based coding assistant. It provides a Textual TUI and runs a multi-turn agent loop against LLM providers through LiteLLM's Responses API. The provider is selected by model identifier, so the same interface can work with providers supported by LiteLLM.

The agent can inspect and modify files, run shell commands, and search or fetch web pages. Each tool category is controlled by a permission level, allowing actions to run automatically, require confirmation, or remain unavailable. Conversations are persisted locally and can be resumed in later sessions.

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Credentials for the LLM provider and model you intend to use

## Getting Started

Install the project dependencies and launch the TUI:

```sh
uv sync
uv run arancio
```

Configure provider credentials as required by LiteLLM before sending a prompt. Arancio stores its local configuration, harness files, and session history under `~/.arancio/`.

## Features

- Provider-agnostic LLM access through LiteLLM and the Responses API
- Streaming responses and multi-turn tool-use loops
- File read, write, and edit tools protected by read-first and change-detection checks
- Shell, web search, and web fetch tools
- Configurable read, write, web, and command-execution permissions
- Persistent sessions, including conversation history and resumable work
- Built-in slash commands for changing models, permissions, effort, and sessions

## Project Structure

```text
src/arancio/
  core/         Agent loop, messages, clients, tools, permissions, and parsers
  ui/           Textual application, controller adapter, and widgets
  commands/     Slash-command implementations
  prompt/       Prompt parsing and user-action execution
  sessions/     Session creation, persistence, validation, and resume support
  settings/     Runtime settings and validation
  storage/      Local Arancio directory and configuration management
harness/        Source system prompt and tool descriptions/schemas
tests/          Unit and integration tests
hooks/          Git hooks for linting, formatting, docstrings, and tests
```

At runtime, the tool harness and system prompt are loaded from `~/.arancio/harness/`. The repository's `harness/` directory is the source copy used by tests and for populating that runtime location.

## Development

Run the test suite and linter with:

```sh
uv run pytest tests/
uv run ruff check src/ tests/
```

Format the code with:

```sh
uv run ruff format src/ tests/
```

The checked-in Git hooks run formatting and linting before commits, and docstring checks plus the full test suite before pushes.

## License

This repository does not currently declare a license.
