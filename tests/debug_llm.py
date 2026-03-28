"""Quick script to see what the LLM actually returns."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

import anthropic
from rule_parser import _SYSTEM_PROMPT


async def main():
    client = anthropic.AsyncAnthropicBedrock(
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
    )
    resp = await client.messages.create(
        model="us.anthropic.claude-sonnet-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT.replace("{zones}", "(no zones)"),
        messages=[{"role": "user", "content": "Alert me if a person is detected"}],
    )
    raw = resp.content[0].text.strip()
    print("=== RAW RESPONSE ===")
    print(raw)
    print()

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    parsed = json.loads(raw)
    print("=== PARSED JSON ===")
    print(json.dumps(parsed, indent=2))


asyncio.run(main())
