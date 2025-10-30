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
        print("Agent initialized")


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    agent = AIAgent(api_key)

# ```bash
# export OPENAI_API_KEY="your-api-key-here"
# uv run runbook/02_agent_class.py
# ```
# Should print: Agent initialized