# Single-File AI Agent Tutorial

Build a complete AI agent in 218 lines of Python. No frameworks, no complexity. Just clean code that shows exactly how agents work.

This tutorial demonstrates a working agent that can read, write, and edit files through natural conversation with OpenAI's GPT models using responses api. Everything runs from a single Python file using `uv`'s inline dependencies: no virtual environments, no pip installs, no dependency conflicts.

You'll understand how AI agents parse responses, execute tools, and maintain conversation context. The mechanics laid bare, without abstractions.

## Credits

This repository is forked from Francis Beeson's implementation: [Single-File AI Agent Tutorial](https://github.com/leobeeson/single-file-ai-agent-tutorial)

The code is based on the tutorial by Thorsten Ball in ["How to Build an Agent"](https://ampcode.com/how-to-build-an-agent) from ampcode.com. Thank you, Thorsten. Brilliant guide.

## Features

- Single-file execution.
- No virtual environment or manual dependency installation required (except for `uv`).
- The AI agent can:
  - Read file contents.
  - List directory contents.
  - Edit existing files or create new ones.
- Interactive chat interface.
- Error handling and feedback.
- Logging of agent tool usage.

## Requirements

- `uv` package manager (see installation instructions below)
- Python 3.12 or higher (though `uv` will [handle this automatically](https://docs.astral.sh/uv/concepts/python-versions/))
- OpenAI API key (you can get one from [OpenAI's Platform](https://platform.openai.com/api-keys))

## Model Selection

The agent supports both reasoning and non-reasoning OpenAI models:

**Reasoning Models** (recommended for complex tasks):
- `gpt-5-mini`, `gpt-5`, `gpt-5-nano`
- `o1-preview`, `o1-mini`, `o3-mini`, `o4-mini`, `o3`, `o4`

**Non-reasoning Models** (faster, more cost-effective):
- `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-4.1-nano`

The agent automatically detects the model type and adjusts API parameters accordingly.

## Installing uv

`uv` is an extremely fast Python package manager that simplifies running Python scripts with inline dependencies.

It allows you to run Python scripts [without needing to create a virtual environment or manually install dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

To [install](https://docs.astral.sh/uv/getting-started/installation/) `uv`::

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, verify it's working:

```bash
uv --version
```

## Running the Agent

1. Export your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Run the agent:

```bash
uv run main.py
```

### Model Selection

You can specify which model to use with the `--model` argument:

```bash
# Use a reasoning model (default: gpt-4o-mini)
uv run main.py --model gpt-5-mini

# Use a non-reasoning model
uv run main.py --model gpt-4o-mini

```

The agent leverages uv's inline dependencies handling from the script headers, so no manual dependency installation is needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
