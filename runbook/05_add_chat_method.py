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
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel

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
                    model="gpt-4.1-mini",
                    max_output_tokens=None,
                    input=self.input,
                    tools=tool_schemas,
                    tool_choice="auto",
                    parallel_tool_calls=True,
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
                    return response.output_text if hasattr(response, "output_text") else ""

                function_results = []
                for function_call in function_calls:
                    result = self._execute_tool(function_call.name, json.loads(function_call.arguments or "{}")) # if arguments is None, set it to an empty dictionary
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


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    agent = AIAgent(api_key)
    # Test chat
    query = "What files are in the current directory?"
    print(f"Query: {query}")
    response = agent.chat(query)
    print(f"Response: {response}")


# ```bash
# export OPENAI_API_KEY="your-api-key-here"
# uv run runbook/05_add_chat_method.py
# ```
# Should print a input query and a response from OpenAI listing the files in the directory