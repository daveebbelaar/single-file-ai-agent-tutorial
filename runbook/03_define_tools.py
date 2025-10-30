# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai", # type: ignore
#     "pydantic",
# ]
# ///

import os
import sys
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
        print(f"Agent initialized with {len(self.tools)} tools")

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
                            "description": "The directory path to list (defaults to current directory), use . for current directory",
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


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    agent = AIAgent(api_key)


# ```bash
# export OPENAI_API_KEY="your-api-key-here"
# uv run runbook/03_define_tools.py
# ```
# Should print: Agent initialized with 3 tools