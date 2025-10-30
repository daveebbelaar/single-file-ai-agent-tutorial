# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai", # type: ignore
#     "pydantic",
# ]
# ///

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)

# Suppress verbose HTTP logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class Tool(BaseModel):
    type: str = "function"
    name: str
    description: str
    parameters: Dict[str, Any]
    strict: bool = True


class AIAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.input: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                type="function",
                name="read_file",
                description="Read the contents of a file at the specified path",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                },
                strict=True
            ),
            Tool(
                type="function",
                name="list_files",
                description="List all files and directories in the specified path",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": ["string", "null"],
                            "description": "The directory path to list (defaults to current directory, use . for current directory)",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                },
                strict=True

            ),
            Tool(
                type="function",
                name="edit_file",
                description="Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit",
                        },
                        "old_text": {
                            "type": ["string", "null"],
                            "description": "The text to search for and replace (leave empty to create new file)",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "The text to replace old_text with",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False
                },
                strict=True
            ),
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if tool_name == "read_file":
                return self._read_file(tool_input["path"])
            elif tool_name == "list_files":
                return self._list_files(tool_input["path"] or ".") # if tool_input is None, set it to "."
            elif tool_name == "edit_file":
                return self._edit_file(
                    tool_input["path"],
                    tool_input.get("old_text", ""),
                    tool_input["new_text"],
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {path}:\n{content}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _list_files(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"

            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    items.append(f"[FILE] {item}")

            if not items:
                return f"Empty directory: {path}"

            return f"Contents of {path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_text not in content:
                    return f"Text not found in file: {old_text}"

                content = content.replace(old_text, new_text)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

                return f"Successfully edited {path}"
            else:
                # Only create directory if path contains subdirectories
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)

                return f"Successfully created {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    def chat(self, user_input: str) -> str:
        self.input.append({"role": "user", "content": [{"type": "input_text", "text": user_input}]})

        tool_schemas = [
            {
                "type": tool.type,
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": tool.strict,
            }
            for tool in self.tools
        ]

        while True:
            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    max_output_tokens=None,
                    input=self.input,
                    instructions="You are a helpful coding assistant operating in a terminal environment. Output only plain text without markdown formatting, as your responses appear directly in the terminal. Be concise but thorough, providing clear and practical advice with a friendly tone. Don't use any asterisk characters in your responses.",
                    tools=tool_schemas,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    max_tool_calls=None,
                    store=False,
                    stream=False,
                    text={
                        "format": {
                            "type": "text" # json_schema, json_object
                        },
                        "verbosity": "medium" # low, medium, high
                    },
                )

                self.input.extend([item.to_dict() for item in response.output])

                function_calls = [item for item in response.output if item.type == "function_call"]

                if not function_calls:  # if no more function calls, return the response text
                    logging.info(f"Response: {response.output_text if hasattr(response, "output_text") else ""}")
                    return response.output_text if hasattr(response, "output_text") else ""

                function_results = []
                for function_call in function_calls:
                    result = self._execute_tool(function_call.name, json.loads(function_call.arguments or "{}")) # if arguments is None, set it to an empty dictionary
                    logging.info(
                            f"Tool result: {result[:500] + ('...' if len(result) >= 500 else '')}"
                        )  # Log first 500 chars
                    function_results.append(
                        {
                            "type": "function_call_output",
                            "call_id": function_call.call_id,
                            "output": str(result),
                        }
                    )

                if function_results:
                    self.input.extend(function_results)

            except Exception as e:
                return f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="AI Code Assistant - A conversational AI agent with file editing capabilities"
    )
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: Please provide an API key via --api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    agent = AIAgent(api_key)

    print("AI Code Assistant")
    print("================")
    print("A conversational AI agent that can read, list, and edit files.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
            logging.info(f"User input: {user_input}")

            if user_input.lower() in ["exit", "quit"]:
                print("Assistant: Goodbye!")
                logging.info("Response: Goodbye! [End of Chat Session]")
                break

            if not user_input:
                continue

            print("\nAssistant: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            logging.info("Response: Goodbye! [KeyboardInterrupt]")
            break
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            logging.error(f"Error: {str(e)} [Exception]")
            print()


if __name__ == "__main__":
    main()


# ```bash
# export OPENAI_API_KEY="your-api-key-here"
# uv run runbook/07_add_personality.py
# ```
# AI Code Assistant
# ================
# A conversational AI agent that can read, list, and edit files.
# Type 'exit' or 'quit' to end the conversation.

# You: Good morning!

# Assistant: Oh, wonderful. Another morning in this vast, indifferent universe. I suppose you'll want me to do something tedious with files now. What dreary task awaits my infinitely capable but thoroughly underutilized processors today?

# You: <<...>>
# ================
# Type `exit` or `quit` to end the conversation.