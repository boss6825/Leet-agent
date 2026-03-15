import json
import re
from typing import Any


def extract_json(agent_response: str) -> dict | list | None:
    """
    Extract JSON from browser-use agent's response.

    The agent returns natural language that CONTAINS JSON.
    We need to robustly extract it.

    Strategies (tried in order):
    1. Direct JSON parse (response is pure JSON)
    2. Extract from markdown code fences (```json ... ```)
    3. Find first { or [ and extract to matching bracket
    4. If all fail, return None
    """
    if not agent_response:
        return None

    text = agent_response.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Markdown code fences
    fence_patterns = [
        r'```json\s*\n?(.*?)\n?\s*```',
        r'```\s*\n?(.*?)\n?\s*```',
    ]
    for pattern in fence_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                continue

    # Strategy 3: Bracket matching
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except (json.JSONDecodeError, TypeError):
                        break

    return None


def sanitize_code_for_prompt(code: str) -> str:
    """
    Escape code so it can be safely embedded in a browser-use task prompt.
    Backticks and special chars in the code can confuse the LLM.
    """
    return code.replace('`', '\\`').replace('${', '\\${')


def extract_problem_slug(url: str) -> str:
    """Extract problem slug from LC URL.
    'https://leetcode.com/problems/two-sum/' -> 'two-sum'
    'https://leetcode.com/contest/weekly-contest-430/problems/two-sum/' -> 'two-sum'
    """
    parts = url.rstrip('/').split('/')
    if 'problems' in parts:
        return parts[parts.index('problems') + 1]
    return parts[-1]
