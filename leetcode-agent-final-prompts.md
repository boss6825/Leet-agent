# LeetCode Contest Auto-Solver Agent — Claude Code Prompts (FINAL V1)

> **Stack**: Python + browser-use + multi-LLM (OpenAI, Anthropic, Gemini)
> **Browser Driver**: browser-use (AI agent that sees & interacts with pages)
> **Code Gen**: Native SDKs (openai, anthropic, google-generativeai) behind a unified wrapper
> **Languages**: C++ first → convert to Python → store both
> **Project Structure**: All code in `src/`

---

## ARCHITECTURE OVERVIEW

```
leetcode-agent/
├── src/
│   ├── __init__.py
│   ├── config.py                # All constants, model tiers, retry configs
│   ├── main.py                  # CLI entry point
│   ├── orchestrator.py          # The brain — full solve pipeline
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract LLM interface
│   │   ├── openai_provider.py   # OpenAI SDK wrapper
│   │   ├── anthropic_provider.py# Anthropic SDK wrapper
│   │   ├── gemini_provider.py   # Google Gemini SDK wrapper
│   │   └── factory.py           # get_llm("openai", "gpt-4o-mini") → LLM instance
│   ├── browser/
│   │   ├── __init__.py
│   │   ├── agent.py             # All browser-use agent tasks
│   │   └── helpers.py           # JSON extraction, response parsing
│   ├── code_gen/
│   │   ├── __init__.py
│   │   ├── solver.py            # Generates solutions (uses llm/ under the hood)
│   │   ├── converter.py         # Converts C++ solution → Python (and future langs)
│   │   └── prompts.py           # All prompt templates in one place
│   ├── storage/
│   │   ├── __init__.py
│   │   └── store.py             # JSON-based solution + attempt storage
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Structured logging with rich
├── solutions/                   # Output directory for solved problems
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

### Why This Structure:

- `llm/` is isolated — adding a new provider means ONE new file + a line in factory.py
- `browser/` is isolated — if browser-use changes API, only this folder changes
- `code_gen/` has prompts.py separate — prompt tuning doesn't touch logic
- `converter.py` is separate from `solver.py` — converting C++ to Python is a
  fundamentally different task than solving a problem from scratch
- Everything in `src/` — standard Python project layout

---

## PROMPT 1: Project Scaffolding, Config & LLM Abstraction Layer

```
You are a senior software engineer building a LeetCode Contest Auto-Solver
agent in Python. The project root already exists with src/config.py, .gitignore,
requirements.txt, and LICENSE. You need to build out the full structure.

== PROJECT CONTEXT ==

This agent:
1. Uses browser-use (AI browser automation) to interact with LeetCode
2. Reads contest problems through the browser (no scraping/selectors)
3. Generates solutions using LLMs (multi-provider: OpenAI, Anthropic, Gemini)
4. Tests solutions by pasting them into LeetCode and running via browser
5. Iterates on failures by feeding failing test cases back to the LLM
6. Falls back to YouTube editorial transcripts as last resort
7. Generates C++ solutions first, then converts to Python

== TASK: Create the LLM abstraction layer and project config ==

=== File: src/config.py ===

Replace/update the existing config.py with:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# LLM Model Configuration
# ──────────────────────────────────────────────
# Three tiers for code generation. Swap models here as you test.
# Each tier has: provider (openai/anthropic/gemini), model name
MODEL_TIERS = {
    "cheap": {"provider": "openai", "model": "gpt-4o-mini"},
    "mid": {"provider": "openai", "model": "gpt-4o"},
    "expensive": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
}

# Browser-use agent model (just drives the browser, keep it cheap)
BROWSER_AGENT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
}

# ──────────────────────────────────────────────
# Retry / Escalation Strategy
# ──────────────────────────────────────────────
# Per difficulty: list of (tier_name, max_retries) in escalation order
ESCALATION_STRATEGY = {
    "Easy": [("cheap", 3), ("mid", 2), ("expensive", 2)],
    "Medium": [("cheap", 2), ("mid", 3), ("expensive", 3)],
    "Hard": [("cheap", 1), ("mid", 2), ("expensive", 4)],
}
# Default if difficulty can't be determined (treat as Medium)
DEFAULT_ESCALATION = [("cheap", 2), ("mid", 3), ("expensive", 3)]

# ──────────────────────────────────────────────
# Language Configuration
# ──────────────────────────────────────────────
# Primary language to solve in (what gets tested first on LC)
PRIMARY_LANGUAGE = "cpp"
# Languages to also convert solutions to and test
CONVERT_TO_LANGUAGES = ["python3"]
# LeetCode language dropdown display names (what you see in the dropdown)
LC_LANGUAGE_NAMES = {
    "cpp": "C++",
    "python3": "Python3",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "rust": "Rust",
}

# ──────────────────────────────────────────────
# Browser Configuration
# ──────────────────────────────────────────────
BROWSER_HEADLESS = False  # False for dev (watch it work), True for prod
BROWSER_PROFILE_PATH = "/tmp/leetcode-browser-profile"

# ──────────────────────────────────────────────
# URLs
# ──────────────────────────────────────────────
LEETCODE_BASE = "https://leetcode.com"
CONTEST_URL_TEMPLATE = "https://leetcode.com/contest/{slug}/"

# ──────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────
SOLUTIONS_DIR = "./solutions"

# ──────────────────────────────────────────────
# Delays & Timeouts (seconds)
# ──────────────────────────────────────────────
DELAY_BETWEEN_AGENT_CALLS = 2     # Pause between browser-use calls
DELAY_BETWEEN_RETRIES = 3         # Pause between LLM retry attempts
DELAY_BETWEEN_QUESTIONS = 5       # Pause between solving different questions
LC_RUN_TIMEOUT = 30               # Max wait for LC "Run" results
LC_SUBMIT_TIMEOUT = 45            # Max wait for LC "Submit" results
YOUTUBE_SEARCH_WAIT = 300         # Wait 5 min if no YT videos found, then retry
MAX_YOUTUBE_VIDEOS_TO_TRY = 3    # How many YT videos to try before giving up
MAX_YOUTUBE_RETRIES_PER_VIDEO = 3 # Retries per video's transcript approach

# ──────────────────────────────────────────────
# API Keys (from .env)
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

=== File: .env.example ===
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
```

=== File: requirements.txt ===
Update to:
```
browser-use
langchain-openai
langchain-anthropic
langchain-google-genai
openai
anthropic
google-generativeai
python-dotenv
rich
```
NOTE: The langchain-* packages are ONLY needed because browser-use
requires them internally. Our code generation uses native SDKs directly.

=== File: src/llm/base.py ===

Create an abstract base class that ALL LLM providers implement:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str           # The actual text response
    model: str             # Model that generated it
    provider: str          # "openai" / "anthropic" / "gemini"
    input_tokens: int      # For cost tracking
    output_tokens: int     # For cost tracking
    cost_usd: float | None # Estimated cost if calculable

class BaseLLM(ABC):
    """Abstract interface for LLM providers."""

    def __init__(self, model: str, provider: str):
        self.model = model
        self.provider = provider

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.0,
                       max_tokens: int = 4096) -> LLMResponse:
        """Generate a completion. All providers must implement this."""
        pass

    def __repr__(self):
        return f"{self.provider}:{self.model}"
```

=== File: src/llm/openai_provider.py ===

```python
from openai import AsyncOpenAI
from .base import BaseLLM, LLMResponse
from src.config import OPENAI_API_KEY

class OpenAILLM(BaseLLM):
    def __init__(self, model: str):
        super().__init__(model=model, provider="openai")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def generate(self, system_prompt, user_prompt,
                       temperature=0.0, max_tokens=4096) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider="openai",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=None  # Can calculate from token counts later
        )
```

=== File: src/llm/anthropic_provider.py ===

Same pattern but using `anthropic.AsyncAnthropic` client.
Use `client.messages.create()` with the Anthropic API format.
Map the system prompt to Anthropic's system parameter.
Map user_prompt to the messages array.

=== File: src/llm/gemini_provider.py ===

Same pattern but using `google.generativeai` SDK.
Use `genai.GenerativeModel(model).generate_content_async()`.
Map system prompt + user prompt appropriately for Gemini's API.

=== File: src/llm/factory.py ===

```python
from .base import BaseLLM
from .openai_provider import OpenAILLM
from .anthropic_provider import AnthropicLLM
from .gemini_provider import GeminiLLM

_PROVIDERS = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "gemini": GeminiLLM,
}

def get_llm(provider: str, model: str) -> BaseLLM:
    """Factory function. Usage: llm = get_llm('openai', 'gpt-4o-mini')"""
    cls = _PROVIDERS.get(provider)
    if not cls:
        raise ValueError(f"Unknown provider: {provider}. "
                        f"Available: {list(_PROVIDERS.keys())}")
    return cls(model=model)

def get_tier_llm(tier_name: str) -> BaseLLM:
    """Get LLM for a specific tier. Usage: llm = get_tier_llm('cheap')"""
    from src.config import MODEL_TIERS
    tier = MODEL_TIERS.get(tier_name)
    if not tier:
        raise ValueError(f"Unknown tier: {tier_name}. "
                        f"Available: {list(MODEL_TIERS.keys())}")
    return get_llm(tier["provider"], tier["model"])
```

=== Create all __init__.py files ===

For src/, src/llm/, src/browser/, src/code_gen/, src/storage/, src/utils/
Keep them minimal — just enough imports for convenient access.

=== EDGE CASES TO HANDLE IN LLM LAYER ===

Each provider wrapper MUST handle these errors gracefully:

1. **Rate limiting**: All three providers return 429 errors.
   Implement exponential backoff: wait 2s, 4s, 8s, 16s, then give up.
   Use a shared retry decorator:
   ```python
   async def retry_with_backoff(func, max_retries=4, base_delay=2):
       for attempt in range(max_retries):
           try:
               return await func()
           except (RateLimitError, APIStatusError) as e:
               if attempt == max_retries - 1:
                   raise
               delay = base_delay * (2 ** attempt)
               logger.warning(f"Rate limited. Retrying in {delay}s...")
               await asyncio.sleep(delay)
   ```

2. **API key missing/invalid**: Check at init time, raise clear error:
   "OPENAI_API_KEY not set in .env file"

3. **Timeout**: Set 60-second timeout on all API calls. If timeout,
   retry once, then raise.

4. **Empty response**: If content is empty string, retry once with
   same prompt. If still empty, raise.

5. **Unexpected response format**: Wrap all response parsing in
   try/except. Log the raw response on failure.

6. **Model not found / deprecated**: Catch the "model not found" error
   from each provider and surface a clear message.

7. **Context length exceeded**: If the prompt is too long (question
   description + code + test cases), truncate the oldest/least relevant
   part (usually the constraints or long descriptions). Each provider
   has different limits — handle this per provider.

8. **Network errors**: ConnectionError, DNS failures. Retry once,
   then raise with "Check your internet connection."

Test this by creating a simple test script:
```python
# test_llm.py
async def test():
    llm = get_llm("openai", "gpt-4o-mini")
    resp = await llm.generate(
        "You are a helpful assistant.",
        "Say hello in exactly 3 words."
    )
    print(f"Response: {resp.content}")
    print(f"Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
```
```

---

## PROMPT 2: Browser Agent — All LeetCode & YouTube Interactions

```
Implement src/browser/agent.py and src/browser/helpers.py.

== CRITICAL UNDERSTANDING OF BROWSER-USE ==

browser-use is an AI agent that:
- Takes a natural language task description
- Opens a real browser (Chrome via Playwright)
- Takes screenshots / reads DOM
- Decides what to click, type, scroll
- Returns a natural language result

We give it INSTRUCTIONS, not selectors. Example:
```python
agent = Agent(
    task="Go to leetcode.com and tell me if I'm logged in",
    llm=llm,    # The LLM that drives browser decisions
    browser=browser
)
result = await agent.run()
# result is a string like "Yes, you are logged in as user123"
```

The browser instance persists between agent calls — so if one call
navigates to a page, the next call starts on that same page.

== FILE: src/browser/helpers.py ==

```python
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
    # Replace backticks that might break markdown formatting
    # Escape any template literal markers
    return code.replace('`', '\\`').replace('${', '\\${')


def extract_problem_slug(url: str) -> str:
    """Extract problem slug from LC URL.
    'https://leetcode.com/problems/two-sum/' → 'two-sum'
    'https://leetcode.com/contest/weekly-contest-430/problems/two-sum/' → 'two-sum'
    """
    parts = url.rstrip('/').split('/')
    # Handle both /problems/slug/ and /contest/.../problems/slug/
    if 'problems' in parts:
        return parts[parts.index('problems') + 1]
    return parts[-1]
```

== FILE: src/browser/agent.py ==

```python
import os
import asyncio
from browser_use import Agent, Browser, BrowserConfig
from src.config import (
    BROWSER_AGENT_CONFIG, BROWSER_HEADLESS,
    BROWSER_PROFILE_PATH, DELAY_BETWEEN_AGENT_CALLS,
    LC_LANGUAGE_NAMES
)
from src.browser.helpers import extract_json, sanitize_code_for_prompt
from src.utils.logger import logger

class BrowserAgent:
    """
    All browser interactions go through this class.
    Each public method = one browser-use agent task.
    """

    def __init__(self):
        self.browser = None
        self.llm = None
        self._initialized = False

    async def initialize(self):
        """
        Set up browser and LLM for browser-use.
        Call this once before using any other method.

        browser-use needs a LangChain-compatible LLM. We use the
        configured BROWSER_AGENT_CONFIG to pick the right one.
        """
        # Import the right LangChain LLM based on config
        provider = BROWSER_AGENT_CONFIG["provider"]
        model = BROWSER_AGENT_CONFIG["model"]

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=model)
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(model=model)
        else:
            raise ValueError(f"Unsupported browser agent provider: {provider}")

        self.browser = Browser(config=BrowserConfig(
            headless=BROWSER_HEADLESS,
        ))
        self._initialized = True
        logger.info(f"Browser agent initialized with {provider}:{model}")

    async def _run_agent_task(self, task: str,
                              max_retries: int = 2) -> str | None:
        """
        Run a single browser-use agent task with retry logic.

        This is the core method all public methods call.
        Returns the agent's string response, or None on failure.

        EDGE CASES HANDLED:
        - Agent raises exception → retry up to max_retries
        - Agent returns empty → retry once
        - Browser crash → attempt to reinitialize browser
        - Timeout → kill agent after reasonable time
        """
        if not self._initialized:
            await self.initialize()

        for attempt in range(max_retries):
            try:
                agent = Agent(
                    task=task,
                    llm=self.llm,
                    browser=self.browser,
                )
                result = await asyncio.wait_for(
                    agent.run(),
                    timeout=120  # 2 minute timeout per agent task
                )

                # browser-use agent.run() returns an AgentHistoryList
                # The final result is typically the last message
                # Extract the actual text result
                result_text = str(result)

                if not result_text or result_text.strip() == "":
                    if attempt < max_retries - 1:
                        logger.warning(f"Empty agent response, retrying... "
                                      f"(attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                        continue
                    return None

                await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                return result_text

            except asyncio.TimeoutError:
                logger.error(f"Agent task timed out (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                    continue
                return None

            except Exception as e:
                logger.error(f"Agent task failed: {e} (attempt {attempt + 1})")

                # If browser crashed, try to reinitialize
                if "browser" in str(e).lower() or "target closed" in str(e).lower():
                    logger.warning("Browser may have crashed. Reinitializing...")
                    try:
                        await self.close()
                        await self.initialize()
                    except Exception as reinit_error:
                        logger.error(f"Reinitialize failed: {reinit_error}")
                        return None

                if attempt < max_retries - 1:
                    await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                    continue
                return None

        return None

    # ──────────────────────────────────────────
    # PUBLIC METHODS — Each is a browser-use task
    # ──────────────────────────────────────────

    async def ensure_logged_in(self) -> str:
        """Check if user is logged into LeetCode. Returns 'LOGGED_IN' or 'NOT_LOGGED_IN'."""
        result = await self._run_agent_task("""
Go to https://leetcode.com

Check if the user is logged in. Signs of being logged in:
- A profile avatar or username in the top-right area of the navigation bar
- NO prominent "Sign In" or "Register" or "Sign up" buttons in the navigation

Signs of NOT being logged in:
- A "Sign In" or "Premium" or "Register" button visible in the navigation
- No user avatar/profile icon

If the user IS logged in, return exactly: LOGGED_IN
If the user is NOT logged in, return exactly: NOT_LOGGED_IN
""")
        if result and "LOGGED_IN" in result and "NOT_LOGGED_IN" not in result:
            return "LOGGED_IN"
        return "NOT_LOGGED_IN"


    async def get_contest_questions(self, contest_slug: str) -> list[dict] | None:
        """
        Get all problem links from a contest page.
        Returns: [{"title": "...", "url": "..."}, ...] or None
        """
        result = await self._run_agent_task(f"""
Go to https://leetcode.com/contest/{contest_slug}/

This is a LeetCode contest page. After the page loads, look for the list of
problems/questions in this contest. There are usually 4 problems.

IMPORTANT: The contest might show a "Contest has ended" message or a
ranking table. The problem links should still be visible — they're usually
listed with their titles and might show point values.

If the page shows "Contest not found" or a 404 error, return: NOT_FOUND

For each problem you can find, get:
- The exact problem title
- The full URL to the problem page

Return as JSON array:
[
  {{"title": "Problem Title", "url": "https://leetcode.com/problems/..."}},
  ...
]

If the contest page redirects to a login page, return: NOT_LOGGED_IN
Return ONLY the JSON array (or NOT_FOUND / NOT_LOGGED_IN).
""")
        if not result:
            return None
        if "NOT_FOUND" in result:
            logger.error(f"Contest not found: {contest_slug}")
            return None
        if "NOT_LOGGED_IN" in result:
            logger.error("Not logged in — contest page redirected to login")
            return None

        parsed = extract_json(result)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
        logger.error(f"Failed to parse contest questions. Raw: {result[:500]}")
        return None


    async def get_question_details(self, problem_url: str) -> dict | None:
        """
        Navigate to a problem page and read the full problem details.
        Returns dict with title, description, examples, constraints, function_signature
        """
        result = await self._run_agent_task(f"""
Go to {problem_url}

This is a LeetCode problem page. Read the ENTIRE problem carefully and extract:

1. **Title**: The problem title (shown at the top)
2. **Difficulty**: Easy, Medium, or Hard (usually shown as a colored tag near the title)
3. **Description**: The COMPLETE problem statement. Include every detail —
   what the function should do, what the inputs/outputs represent, any special
   conditions or definitions. Be thorough.
4. **Examples**: ALL example test cases shown. For each one:
   - Input: the exact input values
   - Output: the exact expected output
   - Explanation: the explanation text (if any)
5. **Constraints**: ALL constraints listed (like "1 <= nums.length <= 10^5",
   "0 <= nums[i] <= 10^9", etc.)
6. **Function Signature**: The function/method signature that needs to be
   implemented. This is visible in the code editor area. It shows the method
   name, parameters, and return type.

IMPORTANT:
- If the problem has images or diagrams, describe what they show in text.
- If there are multiple examples, capture ALL of them, not just the first.
- Read the constraints carefully — they determine the required time complexity.
- For the function signature, look at the code editor on the right side of
  the page. It should show a template like:
  "class Solution {{ public: int twoSum(vector<int>& nums, int target) {{ }} }}"

Return as JSON:
{{
  "title": "Two Sum",
  "difficulty": "Easy",
  "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target...",
  "examples": [
    {{"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]", "explanation": "Because nums[0] + nums[1] == 9"}},
    ...
  ],
  "constraints": ["2 <= nums.length <= 10^4", "-10^9 <= nums[i] <= 10^9"],
  "function_signature": "vector<int> twoSum(vector<int>& nums, int target)"
}}

Return ONLY the JSON.
""")
        if not result:
            return None
        parsed = extract_json(result)
        if isinstance(parsed, dict) and parsed.get('title'):
            return parsed
        logger.error(f"Failed to parse question details. Raw: {result[:500]}")
        return None


    async def paste_and_run(self, problem_url: str, code: str,
                            language: str) -> dict | None:
        """
        Navigate to problem, switch language, paste code, click Run,
        and read results.

        This is the MOST CRITICAL method. It does 5 things in one agent call
        to avoid state issues between calls.

        Args:
            problem_url: Full LC problem URL
            code: The solution code to paste
            language: Language key from config (e.g., "cpp", "python3")
        """
        lc_lang_name = LC_LANGUAGE_NAMES.get(language, language)
        safe_code = sanitize_code_for_prompt(code)

        result = await self._run_agent_task(f"""
Go to {problem_url}

You need to do these steps IN ORDER. Do NOT skip any step.

STEP 1 — SET LANGUAGE:
Look at the code editor area (right side of the page). There's a language
dropdown/selector near the top of the editor. Check what language is currently
selected. If it's not "{lc_lang_name}", click the dropdown and select "{lc_lang_name}".
Wait a moment for the editor to reload with the new language template.

STEP 2 — CLEAR THE EDITOR:
Click inside the code editor to focus it. Select all text (Ctrl+A) and
delete it. The editor should now be empty.

STEP 3 — PASTE THE CODE:
Type this EXACT code into the editor. Do NOT modify it, do NOT add anything,
do NOT remove anything. Type it EXACTLY as shown:

{safe_code}

STEP 4 — VERIFY THE CODE:
After typing, visually confirm the code in the editor looks correct.
If it looks garbled or incomplete, try clearing (Ctrl+A, Delete) and
typing it again.

STEP 5 — RUN THE CODE:
Find and click the "Run" button. It's usually at the bottom of the editor
area. Do NOT click "Submit" — only click "Run".

STEP 6 — WAIT FOR RESULTS:
Wait for the test results to appear. You'll see either:
- A loading spinner (wait for it to finish)
- "Accepted" with green indicators
- "Wrong Answer" with comparison of expected vs actual
- "Runtime Error" with an error message
- "Time Limit Exceeded"
- "Compilation Error" with error details

Wait at least 10-15 seconds for results. Don't read results while
it's still loading.

STEP 7 — READ RESULTS:
Read the test results carefully and return as JSON:
{{
  "status": "Accepted" or "Wrong Answer" or "Runtime Error" or
            "Time Limit Exceeded" or "Compilation Error",
  "all_passed": true/false,
  "test_results_summary": "brief description of what happened",
  "failing_test": {{
    "input": "the input that failed",
    "expected": "expected output",
    "actual": "your code's actual output"
  }} or null if all tests passed,
  "error_message": "compilation or runtime error message" or null
}}

If there's a popup or modal blocking the page (like a contest banner or
premium upsell), dismiss it first by clicking X or pressing Escape.

Return ONLY the JSON.
""", max_retries=3)  # Extra retry for this critical operation

        if not result:
            return None
        parsed = extract_json(result)
        if isinstance(parsed, dict):
            return parsed
        logger.error(f"Failed to parse run results. Raw: {result[:500]}")
        return None


    async def submit_solution(self, problem_url: str) -> dict | None:
        """
        Click Submit on the current problem page (code should already be in editor).
        Wait for submission results and return them.
        """
        result = await self._run_agent_task(f"""
On the current LeetCode problem page, the solution code is already in the editor.

STEP 1: Find and click the "Submit" button. It's near the "Run" button,
usually at the bottom-right of the editor area.

STEP 2: Wait for submission to complete. This can take 5-30 seconds.
You'll see a loading/pending state, then the final result.

STEP 3: Read the FULL submission result. It will show:
- "Accepted" (green) with runtime and memory stats, OR
- "Wrong Answer" with the failing test case details, OR
- "Time Limit Exceeded" / "Memory Limit Exceeded", OR
- "Runtime Error" with error traceback

STEP 4: If the result is "Wrong Answer", CAREFULLY extract:
- How many test cases passed (e.g., "45 / 60 test cases passed")
- The input that failed
- The expected output
- Your code's output
Note: Long inputs might be truncated. Copy whatever is visible.

Return as JSON:
{{
  "status": "Accepted" or "Wrong Answer" or "Runtime Error" or
            "Time Limit Exceeded" or "Memory Limit Exceeded",
  "all_passed": true/false,
  "passed_count": 45,
  "total_count": 60,
  "runtime_ms": 12,
  "memory_mb": 8.5,
  "failing_test": {{
    "input": "the failing test input",
    "expected": "expected output",
    "actual": "your code's output"
  }} or null,
  "error_message": "error details" or null
}}

Return ONLY the JSON.
""", max_retries=3)

        if not result:
            return None
        parsed = extract_json(result)
        if isinstance(parsed, dict):
            return parsed
        logger.error(f"Failed to parse submit results. Raw: {result[:500]}")
        return None


    async def find_youtube_editorial(self, problem_title: str,
                                     contest_slug: str) -> list[str]:
        """Search YouTube for editorial videos for a specific problem."""
        search_query = f"{problem_title} leetcode solution {contest_slug}"
        result = await self._run_agent_task(f"""
Go to https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}

Look at the search results. Find up to 3 videos that:
- Are about solving this specific LeetCode problem: "{problem_title}"
- Were uploaded RECENTLY (within the last few days — look at upload date)
- Are between 3 and 30 minutes long
- Have titles that mention the problem name, "leetcode", or the contest name
- Are NOT YouTube Shorts (Shorts are under 60 seconds)
- Are NOT live streams or multi-hour videos

For each matching video, get the full YouTube URL.

Return as JSON array:
["https://www.youtube.com/watch?v=VIDEO_ID_1", ...]

If no relevant videos found, return: []
Return ONLY the JSON array.
""")
        if not result:
            return []
        parsed = extract_json(result)
        if isinstance(parsed, list):
            return [url for url in parsed if isinstance(url, str)
                    and "youtube.com" in url]
        return []


    async def get_video_transcript(self, video_url: str) -> str | None:
        """
        Get transcript from a YouTube video.
        Tries YouTube's built-in transcript first, then Gemini Ask feature.
        """
        result = await self._run_agent_task(f"""
Go to {video_url}

I need to get the transcript/captions of this video. Try these approaches
in order:

APPROACH 1 — YouTube Transcript:
1. Below the video, look for the video description area
2. Click "...more" or "Show more" to expand the description
3. Look for a "Show transcript" button and click it
4. If a transcript panel opens on the right side, read ALL the transcript text
5. The transcript has timestamps and text — just get the text, ignore timestamps

APPROACH 2 — If no transcript button exists:
1. Look for a Gemini / AI / "Ask" feature on the page
   (Google sometimes shows a Gemini sparkle icon or "Ask about this video")
2. If available, click it
3. Type: "Provide a detailed transcript of this video including all code
   and algorithm explanations"
4. Wait for and read the response

If NEITHER approach works (no transcript button and no Gemini feature),
return exactly: NO_TRANSCRIPT

If you DID get a transcript, return the full transcript text as plain text.
Clean it up:
- Remove [Music], [Applause], and similar markers
- Keep the actual spoken content and any code/algorithm discussion
- Keep it readable and coherent

Return the transcript text (or NO_TRANSCRIPT).
""")
        if not result or "NO_TRANSCRIPT" in result:
            return None
        # Truncate very long transcripts (LLM context limits)
        if len(result) > 15000:
            result = result[:15000] + "\n[TRANSCRIPT TRUNCATED]"
        return result


    async def close(self):
        """Clean up browser resources."""
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self.browser = None
        self._initialized = False
```

== EDGE CASES SUMMARY FOR BROWSER AGENT ==

These are handled in the implementation above, but listing explicitly
so Claude Code doesn't miss any:

1. **Agent returns garbage instead of JSON**: extract_json() tries 3
   parsing strategies. If all fail, method returns None, orchestrator
   handles it.

2. **Browser crashes mid-task**: _run_agent_task() catches exceptions,
   detects browser crash keywords, reinitializes browser, retries.

3. **Agent times out** (page won't load, infinite spinner on LC):
   asyncio.wait_for() with 120s timeout. Retry once, then give up.

4. **LeetCode popup/modal blocks interaction**: Every task prompt
   includes instructions to dismiss popups first.

5. **Code pasting fails** (editor shows wrong content): paste_and_run
   includes a verification step — if code looks wrong, agent retries
   the paste.

6. **LeetCode changes language back**: paste_and_run explicitly checks
   and sets language before pasting.

7. **Results panel shows OLD results** (from previous run): Task prompt
   instructs agent to wait for loading spinner to appear AND disappear.

8. **Contest is still ongoing**: get_contest_questions() handles this —
   if page layout is different (timer, locked problems), agent reports it.

9. **Problem is premium/locked**: get_question_details() — if page says
   "Subscribe to unlock", agent should report it in the response.

10. **YouTube video is age-restricted or region-blocked**: agent reports
    it can't access the video, method returns None, next video is tried.

11. **Code contains backticks or special chars**: sanitize_code_for_prompt()
    escapes them before embedding in the task prompt.

12. **Agent hallucinates results** (says "Accepted" without actually
    running): We mitigate by being explicit in the prompt: "WAIT for
    results to load, do NOT guess." For additional safety, the
    orchestrator can call submit_solution() as a separate verification step.

13. **Session expires mid-workflow**: If any method gets a login redirect,
    it shows up in the agent's response. Orchestrator detects "login" or
    "sign in" in the result and pauses for manual re-login.
```

---

## PROMPT 3: Code Generator & Converter (Solution Generation)

```
Implement src/code_gen/solver.py, src/code_gen/converter.py, and
src/code_gen/prompts.py.

These modules use the LLM abstraction layer (src/llm/) to generate code
solutions and convert between languages.

== FILE: src/code_gen/prompts.py ==

All prompt templates in one file. Easy to tune without touching logic.

```python
"""
All LLM prompt templates for code generation.

DESIGN: Each prompt is a function that takes parameters and returns
(system_prompt, user_prompt) tuple. This makes it easy to:
- A/B test different prompts
- Adjust prompts per language
- Keep all prompt engineering in one place
"""

# ─────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────

def solver_system_prompt(language: str) -> str:
    """System prompt for solving a LeetCode problem from scratch."""

    lang_specific = {
        "cpp": """
- Use modern C++ (C++17 or later). Use STL containers and algorithms.
- Include necessary headers (#include <vector>, #include <algorithm>, etc.)
- Do NOT include `using namespace std;` — use std:: prefix.
  Actually, LeetCode allows `using namespace std;` and most LC solutions use it.
  So: DO use `using namespace std;` for cleaner code.
- Handle integer overflow: use `long long` when values can exceed 2^31.
- For string problems, consider both string and string_view.
- Common patterns: sort + two pointers, unordered_map for O(1) lookup,
  priority_queue for heap, stack/deque for monotonic patterns.
""",
        "python3": """
- Use Python 3.10+ features where helpful (match/case, walrus operator).
- Use type hints in the method signature.
- Python has arbitrary precision integers — no overflow issues.
- Leverage: collections (Counter, defaultdict, deque), heapq,
  bisect, itertools, functools (lru_cache for DP).
- For string manipulation, prefer join/split over concatenation.
- List comprehensions over explicit loops where cleaner.
""",
    }

    return f"""You are an expert competitive programmer. You solve LeetCode
contest problems in {language}.

STRICT RULES:
1. Output ONLY executable code. No explanations, no markdown fences,
   no comments explaining your approach. Just raw code.
2. Your code must define a `class Solution` with the required method(s).
3. Handle ALL edge cases:
   - Empty arrays/strings/graphs
   - Single element inputs
   - Very large inputs (optimize for constraints)
   - Negative numbers, zeros, duplicates
   - Off-by-one errors in indices and boundaries
4. Choose the MOST EFFICIENT algorithm that fits within the constraints.
   Read the constraints to determine required time complexity:
   - n <= 10: O(n!) or O(2^n) brute force OK
   - n <= 20: O(2^n) or O(n * 2^n) OK (bitmask DP)
   - n <= 500: O(n^3) OK
   - n <= 5000: O(n^2) OK
   - n <= 10^5: O(n log n) needed
   - n <= 10^6: O(n) or O(n log n) needed
   - n <= 10^9: O(log n) or O(sqrt(n)) or O(1) needed
5. Consider these algorithmic paradigms:
   - Hash maps/sets for O(1) lookups
   - Two pointers / sliding window for array/string problems
   - Binary search (on answer space too, not just arrays)
   - Dynamic programming (top-down with memoization or bottom-up)
   - Graph: BFS, DFS, Dijkstra, Union-Find, topological sort
   - Trees: DFS, BFS, LCA, segment trees, BIT/Fenwick trees
   - Greedy with proof of correctness
   - Monotonic stack/deque
   - Trie for prefix problems
6. Only use standard library. No external packages.
7. Mentally verify your solution against ALL provided examples before outputting.

{lang_specific.get(language, "")}

OUTPUT: Raw code only. Start with #include or class Solution directly."""


def first_attempt_prompt(question_text: str, language: str,
                         function_signature: str | None = None) -> str:
    """User prompt for first attempt at solving."""
    sig_note = f"\nFunction signature: {function_signature}" if function_signature else ""
    return f"""Solve this LeetCode problem in {language}:

{question_text}
{sig_note}

Output ONLY the code for the Solution class (with any necessary includes/imports)."""


def retry_prompt(question_text: str, language: str,
                 previous_code: str, failed_tests: dict) -> str:
    """User prompt for retrying after a failure."""

    failure_details = []
    if failed_tests.get('status'):
        failure_details.append(f"Status: {failed_tests['status']}")
    if failed_tests.get('input'):
        failure_details.append(f"Failing Input: {failed_tests['input']}")
    if failed_tests.get('expected'):
        failure_details.append(f"Expected Output: {failed_tests['expected']}")
    if failed_tests.get('actual'):
        failure_details.append(f"Your Output: {failed_tests['actual']}")
    if failed_tests.get('error_message'):
        failure_details.append(f"Error: {failed_tests['error_message']}")

    failure_str = '\n'.join(failure_details)

    tle_note = ""
    if failed_tests.get('status') == 'Time Limit Exceeded':
        tle_note = """

CRITICAL: Your solution was TOO SLOW. You MUST use a fundamentally faster
algorithm. Re-read the constraints, determine the required time complexity,
and choose an appropriate algorithm. Do NOT just add micro-optimizations
to the same approach — you need a DIFFERENT algorithmic approach."""

    rte_note = ""
    if failed_tests.get('status') == 'Runtime Error':
        rte_note = """

Your code crashed at runtime. Common causes:
- Array/vector index out of bounds
- Division by zero
- Null/None pointer dereference
- Stack overflow from deep recursion (consider iterative approach)
- Integer overflow (use long long in C++ or check bounds)
Debug carefully."""

    return f"""Your previous solution FAILED. Fix it.

PROBLEM:
{question_text}

YOUR PREVIOUS CODE:
{previous_code}

FAILURE DETAILS:
{failure_str}
{tle_note}{rte_note}

Analyze step by step:
1. What does the failing test case tell you about the bug?
2. What assumption in your code is wrong?
3. What algorithmic change is needed?

Then provide a COMPLETE corrected Solution class.
Output ONLY the corrected code."""


def transcript_prompt(question_text: str, language: str,
                      transcript: str,
                      previous_code: str | None = None) -> str:
    """User prompt for solving with YouTube editorial transcript."""
    prev = ""
    if previous_code:
        prev = f"""

PREVIOUS FAILED ATTEMPTS have not worked. Use the expert's approach instead.
Last failed code:
{previous_code}"""

    return f"""Solve this LeetCode problem in {language}.

A competitive programming expert explained the solution approach in a video.
Use their approach.

PROBLEM:
{question_text}

EXPERT'S EXPLANATION:
{transcript}
{prev}

Based on the expert's approach, implement the solution.
Output ONLY the code for the Solution class."""


def conversion_prompt(source_lang: str, target_lang: str,
                      source_code: str, question_text: str) -> str:
    """User prompt for converting a working solution between languages."""
    return f"""Convert this WORKING {source_lang} solution to {target_lang}.

The solution is CORRECT and ACCEPTED on LeetCode. Convert it faithfully.

PROBLEM (for context):
{question_text}

WORKING {source_lang} CODE:
{source_code}

RULES:
- Maintain the EXACT same algorithm and logic
- Adapt language-specific idioms appropriately
- Keep the same time/space complexity
- The class must be named `Solution` with the same method name
- Include necessary imports/headers for {target_lang}

Output ONLY the {target_lang} code."""
```

== FILE: src/code_gen/solver.py ==

```python
"""
Code generation module. Uses LLM abstraction to generate solutions.
Handles the solve attempt + retry logic per model tier.
"""
import ast
import re
from src.llm.factory import get_tier_llm
from src.llm.base import BaseLLM, LLMResponse
from src.code_gen.prompts import (
    solver_system_prompt, first_attempt_prompt,
    retry_prompt, transcript_prompt
)
from src.config import PRIMARY_LANGUAGE
from src.utils.logger import logger


class Solver:
    """Generates LeetCode solutions using LLMs."""

    async def generate_solution(
        self,
        tier: str,
        question_text: str,
        language: str = PRIMARY_LANGUAGE,
        function_signature: str | None = None,
        failed_tests: dict | None = None,
        previous_code: str | None = None,
        transcript: str | None = None,
    ) -> tuple[str, LLMResponse]:
        """
        Generate a solution using the specified tier's LLM.

        Returns: (cleaned_code, llm_response) tuple

        FLOW:
        - Picks the right prompt based on what's provided
        - Calls the LLM
        - Cleans & validates the response
        - If invalid, retries once with stricter instructions
        """
        llm = get_tier_llm(tier)
        system = solver_system_prompt(language)

        # Pick the right user prompt
        if transcript:
            user = transcript_prompt(question_text, language,
                                    transcript, previous_code)
        elif failed_tests and previous_code:
            user = retry_prompt(question_text, language,
                               previous_code, failed_tests)
        else:
            user = first_attempt_prompt(question_text, language,
                                       function_signature)

        logger.info(f"Generating solution with {llm} "
                   f"(lang={language}, has_failures={failed_tests is not None})")

        response = await llm.generate(system, user)
        code = self._clean_code(response.content, language)

        # Validate
        valid, error = self._validate_code(code, language)
        if not valid:
            logger.warning(f"Generated code invalid: {error}. Retrying...")
            retry_user = (user + f"\n\nCRITICAL: Your previous response was "
                         f"not valid code. Error: {error}\n"
                         f"Return ONLY raw executable code. No markdown, no explanations.")
            response = await llm.generate(system, retry_user)
            code = self._clean_code(response.content, language)
            valid, error = self._validate_code(code, language)
            if not valid:
                logger.error(f"Code still invalid after retry: {error}")
                # Return it anyway — let LC catch the error and we'll retry
                # with failure feedback

        return code, response


    def _clean_code(self, raw: str, language: str) -> str:
        """Extract clean code from LLM response."""
        code = raw.strip()

        # Remove markdown fences
        code = re.sub(r'^```(?:cpp|c\+\+|python3?|java|javascript|typescript|go|rust)?\s*\n?',
                      '', code)
        code = re.sub(r'\n?```\s*$', '', code)

        # Remove any preamble before the actual code starts
        if language == "cpp":
            # Find first #include or class Solution
            markers = ['#include', 'class Solution', 'using namespace']
            for marker in markers:
                idx = code.find(marker)
                if idx > 0:
                    code = code[idx:]
                    break
        elif language == "python3":
            markers = ['class Solution', 'from ', 'import ']
            for marker in markers:
                idx = code.find(marker)
                if idx > 0:
                    code = code[idx:]
                    break

        # Remove trailing explanation text after the code
        # Look for common patterns that indicate explanation
        trail_patterns = [
            '\n\nExplanation:', '\n\nTime complexity:',
            '\n\nThe ', '\n\nThis ', '\n\nNote:',
            '\n\n**', '\n\nComplexity:',
        ]
        for pattern in trail_patterns:
            idx = code.find(pattern)
            if idx > 0:
                code = code[:idx]

        return code.strip()


    def _validate_code(self, code: str, language: str) -> tuple[bool, str]:
        """Basic validation that generated code is structurally correct."""

        if not code:
            return False, "Empty code"

        if "class Solution" not in code:
            return False, "No 'class Solution' found"

        if language == "python3":
            try:
                ast.parse(code)
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"

        if language == "cpp":
            # Basic checks (can't fully compile-check here)
            # Check for unbalanced braces
            if code.count('{') != code.count('}'):
                return False, (f"Unbalanced braces: {code.count('{')} open, "
                             f"{code.count('}')} close")

        return True, "OK"
```

== FILE: src/code_gen/converter.py ==

```python
"""
Converts a working solution from one language to another.
Used after C++ solution is accepted — convert to Python and test.
"""
from src.llm.factory import get_tier_llm
from src.code_gen.prompts import conversion_prompt, solver_system_prompt
from src.utils.logger import logger


class Converter:
    """Convert accepted solutions between programming languages."""

    async def convert(
        self,
        source_lang: str,
        target_lang: str,
        source_code: str,
        question_text: str,
        tier: str = "cheap",  # Conversion is easy — cheap model is fine
    ) -> str:
        """
        Convert a working solution from source_lang to target_lang.
        Returns the converted code.

        Uses the cheap tier by default since converting a known-correct
        solution is much easier than solving from scratch.
        """
        llm = get_tier_llm(tier)
        system = solver_system_prompt(target_lang)
        user = conversion_prompt(source_lang, target_lang,
                                source_code, question_text)

        logger.info(f"Converting {source_lang} → {target_lang} with {llm}")

        response = await llm.generate(system, user)
        code = self._clean_code(response.content, target_lang)

        return code

    def _clean_code(self, raw: str, language: str) -> str:
        """Same cleaning logic as Solver — DRY this up by importing from solver."""
        # For now, duplicate the logic. In production, extract to a shared util.
        import re
        code = raw.strip()
        code = re.sub(r'^```(?:cpp|c\+\+|python3?|java)?\s*\n?', '', code)
        code = re.sub(r'\n?```\s*$', '', code)

        if language == "python3":
            idx = code.find("class Solution")
            if idx > 0:
                code = code[idx:]

        trail_patterns = ['\n\nExplanation:', '\n\nTime complexity:',
                         '\n\nThe ', '\n\nThis ']
        for pattern in trail_patterns:
            idx = code.find(pattern)
            if idx > 0:
                code = code[:idx]

        return code.strip()
```

== EDGE CASES IN CODE GENERATION ==

1. **LLM returns markdown-wrapped code**: _clean_code strips fences.

2. **LLM returns explanation + code**: _clean_code finds code start markers
   and strips preamble.

3. **LLM returns code + explanation**: _clean_code strips trailing text.

4. **LLM returns multiple solution attempts**: (says "Actually, here's a
   better approach..."). _clean_code takes from the LAST "class Solution"
   occurrence? No — that's risky. Better: take from the FIRST occurrence
   and hope it's complete. If it's not, LC will give compilation error
   and we retry.

5. **LLM returns code in wrong language**: _validate_code does basic
   structural checks. If C++ code is returned when Python was requested
   (or vice versa), we catch it:
   - Python code won't have #include → valid check
   - C++ code won't pass ast.parse() → caught

6. **LLM context overflow** (very long problem description + code + test
   cases): Truncate the least important parts. Priority order for what to
   keep: question description > failing test cases > constraints > examples
   > previous code (truncate from middle).

7. **Generated C++ has include issues**: LeetCode provides common includes
   automatically, but some solutions need specific ones. If compilation
   fails, the error message will say which include is missing, and the
   retry prompt includes the error.

8. **Solution class has wrong method name**: This will show up as a
   compilation/runtime error. The retry prompt includes the error, and
   the function_signature helps the LLM get it right next time.
```

---

## PROMPT 4: Solution Storage

```
Implement src/storage/store.py — JSON-based storage for solutions and
attempt history.

```python
"""
Solution storage. Saves all attempts, successes, and failures as JSON.
One JSON file per contest.

DESIGN:
- Atomic writes (write to temp file, then rename) to prevent corruption
- Pretty-printed JSON for human readability
- Tracks every attempt with timestamp, model used, tier, code, and result
- Tracks which language each solution is in
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.config import SOLUTIONS_DIR
from src.utils.logger import logger

console = Console()

class SolutionStore:

    def __init__(self):
        self.base_dir = Path(SOLUTIONS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _contest_file(self, contest_slug: str) -> Path:
        return self.base_dir / f"{contest_slug}.json"

    def _load(self, contest_slug: str) -> dict:
        """Load contest data, or create empty structure."""
        path = self._contest_file(contest_slug)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {
            "contest_slug": contest_slug,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "problems": {}
        }

    def _save(self, contest_slug: str, data: dict):
        """Atomic write: write to temp file then rename."""
        path = self._contest_file(contest_slug)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.base_dir, suffix='.json.tmp'
        )
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _ensure_problem(self, data: dict, problem_slug: str,
                        title: str, url: str, difficulty: str = "Unknown"):
        """Ensure a problem entry exists in the data."""
        if problem_slug not in data["problems"]:
            data["problems"][problem_slug] = {
                "title": title,
                "url": url,
                "difficulty": difficulty,
                "status": "in_progress",
                "solutions": {},     # keyed by language
                "attempts": [],
                "total_attempts": 0,
            }

    def save_attempt(
        self,
        contest_slug: str,
        problem_slug: str,
        title: str,
        url: str,
        difficulty: str,
        attempt_num: int,
        tier: str,
        model_info: str,       # e.g., "openai:gpt-4o-mini"
        language: str,
        code: str,
        result_status: str,
        failing_test: dict | None,
        source: str,           # "llm_direct" | "llm_retry" | "llm_with_transcript"
    ):
        """Save a single attempt (pass or fail)."""
        data = self._load(contest_slug)
        self._ensure_problem(data, problem_slug, title, url, difficulty)

        attempt = {
            "attempt": attempt_num,
            "tier": tier,
            "model": model_info,
            "language": language,
            "source": source,
            "code": code,
            "result_status": result_status,
            "failing_test": failing_test,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        data["problems"][problem_slug]["attempts"].append(attempt)
        data["problems"][problem_slug]["total_attempts"] = attempt_num

        self._save(contest_slug, data)
        logger.debug(f"Saved attempt #{attempt_num} for {problem_slug}")

    def mark_solved(
        self,
        contest_slug: str,
        problem_slug: str,
        language: str,
        final_code: str,
        used_transcript: bool,
    ):
        """Mark a problem as solved in a specific language."""
        data = self._load(contest_slug)
        problem = data["problems"].get(problem_slug)
        if not problem:
            logger.error(f"Cannot mark solved: {problem_slug} not found")
            return

        problem["solutions"][language] = {
            "code": final_code,
            "solved_at": datetime.now(timezone.utc).isoformat(),
            "used_transcript": used_transcript,
        }

        # Update status
        if used_transcript:
            problem["status"] = "solved_with_transcript"
        else:
            problem["status"] = "solved"

        self._save(contest_slug, data)
        logger.info(f"Marked {problem_slug} as solved in {language}")

    def mark_failed(self, contest_slug: str, problem_slug: str):
        """Mark a problem as failed (all retries exhausted)."""
        data = self._load(contest_slug)
        problem = data["problems"].get(problem_slug)
        if problem:
            problem["status"] = "failed"
            self._save(contest_slug, data)
        logger.info(f"Marked {problem_slug} as failed")

    def mark_contest_complete(self, contest_slug: str):
        """Mark the contest as completed."""
        data = self._load(contest_slug)
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save(contest_slug, data)

    def is_solved(self, contest_slug: str, problem_slug: str,
                  language: str = None) -> bool:
        """Check if a problem is already solved (optionally in a specific language)."""
        data = self._load(contest_slug)
        problem = data["problems"].get(problem_slug)
        if not problem:
            return False
        if language:
            return language in problem.get("solutions", {})
        return problem.get("status") in ("solved", "solved_with_transcript")

    def get_attempt_count(self, contest_slug: str, problem_slug: str) -> int:
        data = self._load(contest_slug)
        problem = data["problems"].get(problem_slug)
        return problem.get("total_attempts", 0) if problem else 0

    def print_report(self, contest_slug: str):
        """Print a rich formatted report to terminal."""
        data = self._load(contest_slug)

        table = Table(title=f"Contest: {contest_slug}", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Problem", style="bold")
        table.add_column("Difficulty")
        table.add_column("Status")
        table.add_column("Attempts", justify="right")
        table.add_column("Languages")

        solved = 0
        total = 0
        for i, (slug, problem) in enumerate(data.get("problems", {}).items(), 1):
            total += 1
            status = problem.get("status", "unknown")
            attempts = problem.get("total_attempts", 0)
            langs = ", ".join(problem.get("solutions", {}).keys()) or "-"
            difficulty = problem.get("difficulty", "?")

            if status == "solved":
                status_str = "[green]✅ Solved[/green]"
                solved += 1
            elif status == "solved_with_transcript":
                status_str = "[yellow]✅ Solved (YT)[/yellow]"
                solved += 1
            elif status == "failed":
                status_str = "[red]❌ Failed[/red]"
            else:
                status_str = "[dim]⏳ In Progress[/dim]"

            diff_colors = {"Easy": "green", "Medium": "yellow", "Hard": "red"}
            diff_str = f"[{diff_colors.get(difficulty, 'white')}]{difficulty}[/]"

            table.add_row(str(i), problem.get("title", slug), diff_str,
                         status_str, str(attempts), langs)

        console.print(table)
        console.print(Panel(f"Solved: {solved}/{total}",
                           style="bold cyan"))
```

EDGE CASES:
1. **Concurrent writes**: Unlikely in V1 (single process), but atomic
   writes prevent corruption if user Ctrl+C's mid-write.
2. **Disk full**: Let the OS error propagate — not much we can do.
3. **Corrupt JSON file**: Wrap _load in try/except, if JSON is corrupt,
   log error and start fresh (losing history — add a backup mechanism
   in V2).
4. **Same contest run twice**: Existing data is loaded and new attempts
   are appended. Already-solved problems are skipped.
5. **Unicode in problem titles**: ensure_ascii=False in json.dump handles this.
```

---

## PROMPT 5: Orchestrator — The Brain

```
Implement src/orchestrator.py — the main solve pipeline that coordinates
everything.

This is the most important file. It implements the escalation strategy,
the retry loop, the language conversion, and the YouTube fallback.

```python
"""
Orchestrator: The brain of the agent.

PIPELINE PER QUESTION:
1. Read question details from LeetCode (via browser agent)
2. Attempt to solve in C++ using escalating model tiers:
   Easy:   cheap(3) → mid(2) → expensive(2) → YouTube
   Medium: cheap(2) → mid(3) → expensive(3) → YouTube
   Hard:   cheap(1) → mid(2) → expensive(4) → YouTube
3. For each attempt:
   a. Generate code using current tier's LLM
   b. Paste into LC and Run
   c. If Run passes → Submit
   d. If Submit accepted → solved! Convert to Python, test that too.
   e. If fail → feed failing test cases back, retry within same tier
   f. If tier exhausted → escalate to next tier
4. If all tiers fail → YouTube fallback
5. Store everything (attempts, solutions, failures)
"""
import asyncio
from rich.console import Console
from rich.panel import Panel
from src.browser.agent import BrowserAgent
from src.browser.helpers import extract_problem_slug
from src.code_gen.solver import Solver
from src.code_gen.converter import Converter
from src.storage.store import SolutionStore
from src.config import (
    ESCALATION_STRATEGY, DEFAULT_ESCALATION,
    PRIMARY_LANGUAGE, CONVERT_TO_LANGUAGES,
    DELAY_BETWEEN_RETRIES, DELAY_BETWEEN_QUESTIONS,
    MAX_YOUTUBE_VIDEOS_TO_TRY, MAX_YOUTUBE_RETRIES_PER_VIDEO,
    YOUTUBE_SEARCH_WAIT, MODEL_TIERS
)
from src.utils.logger import logger

console = Console()


class Orchestrator:

    def __init__(self):
        self.agent = BrowserAgent()
        self.solver = Solver()
        self.converter = Converter()
        self.store = SolutionStore()

    async def solve_contest(self, contest_slug: str,
                            skip_youtube: bool = False,
                            only_question: int | None = None):
        """
        Main entry point. Solve all (or one) questions from a contest.
        """
        console.print(Panel(
            f"🚀 LeetCode Contest Solver\n"
            f"Contest: {contest_slug}\n"
            f"Primary language: {PRIMARY_LANGUAGE}\n"
            f"YouTube fallback: {'disabled' if skip_youtube else 'enabled'}",
            style="bold green"
        ))

        try:
            await self.agent.initialize()

            # ── Check login ──
            console.print("\n[yellow]Checking LeetCode login...[/yellow]")
            login = await self.agent.ensure_logged_in()
            if login != "LOGGED_IN":
                console.print("[bold red]Not logged in to LeetCode![/bold red]")
                console.print("The browser is open. Please log into LeetCode now.")
                console.print("After logging in, press Enter to continue...")
                await asyncio.get_event_loop().run_in_executor(
                    None, input, ""
                )
                login = await self.agent.ensure_logged_in()
                if login != "LOGGED_IN":
                    console.print("[bold red]Still not logged in. Aborting.[/bold red]")
                    return
            console.print("[green]✓ Logged in[/green]")

            # ── Get contest questions ──
            console.print(f"\n[cyan]Fetching contest questions...[/cyan]")
            questions = await self.agent.get_contest_questions(contest_slug)
            if not questions:
                console.print("[bold red]Failed to get contest questions![/bold red]")
                return

            console.print(f"[green]Found {len(questions)} questions:[/green]")
            for i, q in enumerate(questions, 1):
                console.print(f"  Q{i}: {q.get('title', 'Unknown')}")

            # ── Filter if --question flag used ──
            if only_question:
                if only_question > len(questions):
                    console.print(f"[red]Question {only_question} doesn't exist "
                                 f"(contest has {len(questions)} questions)[/red]")
                    return
                questions = [questions[only_question - 1]]
                console.print(f"\n[yellow]Solving only Q{only_question}[/yellow]")

            # ── Solve each question ──
            for i, q in enumerate(questions, 1):
                qnum = only_question or i
                console.print(f"\n{'━'*60}")
                console.print(f"[bold cyan]Q{qnum}: {q.get('title', 'Unknown')}[/bold cyan]")
                console.print(f"{'━'*60}")

                await self._solve_question(contest_slug, q, skip_youtube)

                if i < len(questions):
                    console.print(f"\n[dim]Waiting {DELAY_BETWEEN_QUESTIONS}s "
                                 f"before next question...[/dim]")
                    await asyncio.sleep(DELAY_BETWEEN_QUESTIONS)

            # ── Report ──
            self.store.mark_contest_complete(contest_slug)
            console.print(f"\n{'━'*60}")
            self.store.print_report(contest_slug)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted! Saving current state...[/yellow]")
            self.store.mark_contest_complete(contest_slug)
            self.store.print_report(contest_slug)

        finally:
            await self.agent.close()


    async def _solve_question(self, contest_slug: str,
                              question: dict, skip_youtube: bool):
        """Solve a single question through the full pipeline."""

        problem_url = question['url']
        problem_slug = extract_problem_slug(problem_url)
        title = question.get('title', problem_slug)

        # ── Skip if already solved ──
        if self.store.is_solved(contest_slug, problem_slug, PRIMARY_LANGUAGE):
            console.print(f"[green]Already solved in {PRIMARY_LANGUAGE}. Skipping.[/green]")
            return

        # ── Read problem details ──
        console.print("[cyan]Reading problem details...[/cyan]")
        details = await self.agent.get_question_details(problem_url)
        if not details:
            console.print("[red]Failed to read problem. Skipping.[/red]")
            self.store.mark_failed(contest_slug, problem_slug)
            return

        difficulty = details.get('difficulty', 'Medium')  # Default to Medium
        question_text = self._build_question_text(details)
        function_sig = details.get('function_signature')

        console.print(f"[dim]Difficulty: {difficulty}[/dim]")

        # ── Get escalation strategy ──
        escalation = ESCALATION_STRATEGY.get(difficulty, DEFAULT_ESCALATION)
        console.print(f"[dim]Escalation plan: "
                     f"{' → '.join(f'{t}({r})' for t, r in escalation)}[/dim]")

        # ═══════════════════════════════════════
        # PHASE 1: LLM Solve with Tier Escalation
        # ═══════════════════════════════════════

        global_attempt = 0
        previous_code = None
        failed_tests = None

        for tier_name, max_retries in escalation:
            tier_config = MODEL_TIERS[tier_name]
            model_info = f"{tier_config['provider']}:{tier_config['model']}"
            console.print(f"\n[bold]Tier: {tier_name} ({model_info}) — "
                         f"up to {max_retries} attempts[/bold]")

            for retry in range(1, max_retries + 1):
                global_attempt += 1
                console.print(f"\n  [yellow]Attempt {global_attempt} "
                             f"(tier={tier_name}, retry={retry}/{max_retries})[/yellow]")

                # ── Generate solution ──
                try:
                    code, llm_resp = await self.solver.generate_solution(
                        tier=tier_name,
                        question_text=question_text,
                        language=PRIMARY_LANGUAGE,
                        function_signature=function_sig,
                        failed_tests=failed_tests if retry > 1 or previous_code else None,
                        previous_code=previous_code if retry > 1 or previous_code else None,
                    )
                    console.print(f"  [dim]Generated {len(code)} chars, "
                                 f"tokens: {llm_resp.input_tokens}in/"
                                 f"{llm_resp.output_tokens}out[/dim]")
                except Exception as e:
                    console.print(f"  [red]LLM error: {e}[/red]")
                    await asyncio.sleep(DELAY_BETWEEN_RETRIES)
                    continue

                # ── Paste & Run on LeetCode ──
                console.print(f"  Pasting & running on LeetCode...")
                run_result = await self.agent.paste_and_run(
                    problem_url, code, PRIMARY_LANGUAGE
                )

                if not run_result:
                    console.print(f"  [red]Browser agent failed to run code[/red]")
                    self.store.save_attempt(
                        contest_slug, problem_slug, title, problem_url,
                        difficulty, global_attempt, tier_name, model_info,
                        PRIMARY_LANGUAGE, code, "Browser Error", None,
                        "llm_retry" if retry > 1 else "llm_direct"
                    )
                    previous_code = code
                    await asyncio.sleep(DELAY_BETWEEN_RETRIES)
                    continue

                status = run_result.get('status', 'Unknown')
                console.print(f"  Run result: {status}")

                # ── If example tests pass → Submit ──
                if run_result.get('all_passed', False):
                    console.print(f"  [green]Example tests passed! Submitting...[/green]")
                    submit_result = await self.agent.submit_solution(problem_url)

                    if submit_result and submit_result.get('all_passed', False):
                        # ✅ ACCEPTED!
                        console.print(f"  [bold green]✅ ACCEPTED![/bold green]")
                        self.store.save_attempt(
                            contest_slug, problem_slug, title, problem_url,
                            difficulty, global_attempt, tier_name, model_info,
                            PRIMARY_LANGUAGE, code, "Accepted", None,
                            "llm_retry" if retry > 1 else "llm_direct"
                        )
                        self.store.mark_solved(
                            contest_slug, problem_slug,
                            PRIMARY_LANGUAGE, code, False
                        )

                        # ── Convert to other languages ──
                        await self._convert_and_test(
                            contest_slug, problem_slug, title,
                            problem_url, difficulty, question_text,
                            code, global_attempt
                        )
                        return  # Done with this question!

                    else:
                        # Submission failed on hidden tests
                        status = submit_result.get('status', 'Unknown') if submit_result else 'Unknown'
                        console.print(f"  [red]Submission failed: {status}[/red]")
                        failed_tests = self._extract_failure(submit_result)
                        self.store.save_attempt(
                            contest_slug, problem_slug, title, problem_url,
                            difficulty, global_attempt, tier_name, model_info,
                            PRIMARY_LANGUAGE, code, status, failed_tests,
                            "llm_retry"
                        )
                else:
                    # Example tests failed
                    console.print(f"  [red]Tests failed: {status}[/red]")
                    failed_tests = self._extract_failure(run_result)
                    self.store.save_attempt(
                        contest_slug, problem_slug, title, problem_url,
                        difficulty, global_attempt, tier_name, model_info,
                        PRIMARY_LANGUAGE, code, status, failed_tests,
                        "llm_retry" if retry > 1 else "llm_direct"
                    )

                previous_code = code
                await asyncio.sleep(DELAY_BETWEEN_RETRIES)

            # Tier exhausted — carry forward the last failure to next tier
            console.print(f"  [yellow]Tier '{tier_name}' exhausted.[/yellow]")

        # ═══════════════════════════════════════
        # PHASE 2: YouTube Fallback
        # ═══════════════════════════════════════

        if skip_youtube:
            console.print("[yellow]YouTube fallback disabled. Giving up.[/yellow]")
            self.store.mark_failed(contest_slug, problem_slug)
            return

        console.print(f"\n[magenta]All model tiers exhausted. "
                     f"Trying YouTube editorials...[/magenta]")

        videos = await self.agent.find_youtube_editorial(title, contest_slug)

        if not videos:
            console.print(f"[yellow]No videos yet. Waiting "
                         f"{YOUTUBE_SEARCH_WAIT}s...[/yellow]")
            await asyncio.sleep(YOUTUBE_SEARCH_WAIT)
            videos = await self.agent.find_youtube_editorial(title, contest_slug)

        if not videos:
            console.print("[red]No YouTube editorials found. Giving up.[/red]")
            self.store.mark_failed(contest_slug, problem_slug)
            return

        for vid_idx, video_url in enumerate(videos[:MAX_YOUTUBE_VIDEOS_TO_TRY], 1):
            console.print(f"\n[magenta]Video {vid_idx}/{min(len(videos), MAX_YOUTUBE_VIDEOS_TO_TRY)}: "
                         f"{video_url}[/magenta]")

            transcript = await self.agent.get_video_transcript(video_url)
            if not transcript:
                console.print("[yellow]No transcript. Next video...[/yellow]")
                continue

            console.print(f"[green]Got transcript ({len(transcript)} chars)[/green]")

            # Use expensive tier for transcript-based solving
            yt_previous_code = None
            yt_failed_tests = None

            for yt_retry in range(1, MAX_YOUTUBE_RETRIES_PER_VIDEO + 1):
                global_attempt += 1
                console.print(f"\n  [magenta]YT attempt {yt_retry}/"
                             f"{MAX_YOUTUBE_RETRIES_PER_VIDEO}[/magenta]")

                try:
                    code, llm_resp = await self.solver.generate_solution(
                        tier="expensive",  # Use best model for transcript
                        question_text=question_text,
                        language=PRIMARY_LANGUAGE,
                        function_signature=function_sig,
                        transcript=transcript,
                        failed_tests=yt_failed_tests,
                        previous_code=yt_previous_code,
                    )
                except Exception as e:
                    console.print(f"  [red]LLM error: {e}[/red]")
                    continue

                run_result = await self.agent.paste_and_run(
                    problem_url, code, PRIMARY_LANGUAGE
                )

                if run_result and run_result.get('all_passed', False):
                    submit_result = await self.agent.submit_solution(problem_url)
                    if submit_result and submit_result.get('all_passed', False):
                        console.print(f"  [bold green]✅ ACCEPTED (via transcript)![/bold green]")
                        self.store.save_attempt(
                            contest_slug, problem_slug, title, problem_url,
                            difficulty, global_attempt, "expensive",
                            f"{MODEL_TIERS['expensive']['provider']}:{MODEL_TIERS['expensive']['model']}",
                            PRIMARY_LANGUAGE, code, "Accepted", None,
                            "llm_with_transcript"
                        )
                        self.store.mark_solved(
                            contest_slug, problem_slug,
                            PRIMARY_LANGUAGE, code, True
                        )
                        await self._convert_and_test(
                            contest_slug, problem_slug, title,
                            problem_url, difficulty, question_text,
                            code, global_attempt
                        )
                        return
                    else:
                        yt_failed_tests = self._extract_failure(submit_result)
                else:
                    yt_failed_tests = self._extract_failure(run_result)

                yt_previous_code = code
                self.store.save_attempt(
                    contest_slug, problem_slug, title, problem_url,
                    difficulty, global_attempt, "expensive",
                    f"{MODEL_TIERS['expensive']['provider']}:{MODEL_TIERS['expensive']['model']}",
                    PRIMARY_LANGUAGE, code,
                    yt_failed_tests.get('status', 'Unknown') if yt_failed_tests else 'Unknown',
                    yt_failed_tests, "llm_with_transcript"
                )
                await asyncio.sleep(DELAY_BETWEEN_RETRIES)

        # ═══════════════════════════════════════
        # PHASE 3: Give up
        # ═══════════════════════════════════════
        console.print(f"\n[bold red]❌ Failed to solve: {title}[/bold red]")
        self.store.mark_failed(contest_slug, problem_slug)


    async def _convert_and_test(
        self, contest_slug, problem_slug, title, problem_url,
        difficulty, question_text, cpp_code, base_attempt
    ):
        """
        After C++ solution is accepted, convert to Python and test.
        """
        for target_lang in CONVERT_TO_LANGUAGES:
            if self.store.is_solved(contest_slug, problem_slug, target_lang):
                console.print(f"  [dim]Already solved in {target_lang}[/dim]")
                continue

            console.print(f"\n  [cyan]Converting to {target_lang}...[/cyan]")
            try:
                converted = await self.converter.convert(
                    PRIMARY_LANGUAGE, target_lang, cpp_code, question_text
                )
            except Exception as e:
                console.print(f"  [red]Conversion failed: {e}[/red]")
                continue

            # Test the converted solution
            console.print(f"  Testing {target_lang} solution on LeetCode...")
            run_result = await self.agent.paste_and_run(
                problem_url, converted, target_lang
            )

            if run_result and run_result.get('all_passed', False):
                submit_result = await self.agent.submit_solution(problem_url)
                if submit_result and submit_result.get('all_passed', False):
                    console.print(f"  [green]✅ {target_lang} solution accepted![/green]")
                    self.store.mark_solved(
                        contest_slug, problem_slug,
                        target_lang, converted, False
                    )
                    continue

            # If conversion fails, try once more with error feedback
            console.print(f"  [yellow]{target_lang} conversion failed. "
                         f"Retrying with feedback...[/yellow]")
            failed = self._extract_failure(run_result)
            try:
                converted2 = await self.converter.convert(
                    PRIMARY_LANGUAGE, target_lang, cpp_code, question_text,
                    tier="mid"  # Use slightly better model for retry
                )
                run2 = await self.agent.paste_and_run(
                    problem_url, converted2, target_lang
                )
                if run2 and run2.get('all_passed', False):
                    sub2 = await self.agent.submit_solution(problem_url)
                    if sub2 and sub2.get('all_passed', False):
                        console.print(f"  [green]✅ {target_lang} accepted (2nd try)![/green]")
                        self.store.mark_solved(
                            contest_slug, problem_slug,
                            target_lang, converted2, False
                        )
                        continue
            except Exception:
                pass

            console.print(f"  [yellow]Could not convert to {target_lang}. "
                         f"C++ solution saved.[/yellow]")


    def _extract_failure(self, result: dict | None) -> dict | None:
        """Extract failure details from a run/submit result."""
        if not result:
            return {"status": "Browser Error"}

        failure = {"status": result.get("status", "Unknown")}

        failing_test = result.get("failing_test")
        if failing_test and isinstance(failing_test, dict):
            failure.update({
                "input": failing_test.get("input"),
                "expected": failing_test.get("expected"),
                "actual": failing_test.get("actual"),
            })

        if result.get("error_message"):
            failure["error_message"] = result["error_message"]

        return failure


    def _build_question_text(self, details: dict) -> str:
        """Format question details into clean text for code generation."""
        parts = [f"TITLE: {details.get('title', 'Unknown')}"]

        if details.get('difficulty'):
            parts.append(f"DIFFICULTY: {details['difficulty']}")

        parts.append(f"\nDESCRIPTION:\n{details.get('description', 'No description')}")

        examples = details.get('examples', [])
        if examples:
            parts.append("\nEXAMPLES:")
            for i, ex in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                if ex.get('input'):
                    parts.append(f"  Input: {ex['input']}")
                if ex.get('output'):
                    parts.append(f"  Output: {ex['output']}")
                if ex.get('explanation'):
                    parts.append(f"  Explanation: {ex['explanation']}")

        constraints = details.get('constraints', [])
        if constraints:
            parts.append("\nCONSTRAINTS:")
            for c in constraints:
                parts.append(f"  • {c}")

        return '\n'.join(parts)
```

== EDGE CASES HANDLED BY ORCHESTRATOR ==

1. **CAPTCHA / anti-bot detection**: If paste_and_run or submit returns
   None repeatedly, and the agent reports login/captcha issues, the
   orchestrator should detect this pattern:
   ```python
   consecutive_browser_failures = 0
   # In the retry loop:
   if not run_result:
       consecutive_browser_failures += 1
       if consecutive_browser_failures >= 3:
           console.print("[bold yellow]⚠️ Multiple browser failures. "
                        "LeetCode may be blocking automation.[/bold yellow]")
           console.print("Check the browser window. If there's a CAPTCHA "
                        "or verification, solve it manually.")
           console.print("Press Enter to continue...")
           await asyncio.get_event_loop().run_in_executor(None, input, "")
           consecutive_browser_failures = 0
   else:
       consecutive_browser_failures = 0
   ```

2. **LeetCode rate limiting ("too many submissions")**: Detect "rate"
   or "too many" in result status/error_message:
   ```python
   if any(phrase in str(run_result).lower()
          for phrase in ["rate limit", "too many", "try again later"]):
       console.print("[yellow]Rate limited. Waiting 60 seconds...[/yellow]")
       await asyncio.sleep(60)
   ```

3. **Carrying failure context across tiers**: When we escalate from
   cheap → mid tier, the `previous_code` and `failed_tests` from the
   last cheap attempt carry forward. The mid model gets to see what
   the cheap model tried and why it failed.

4. **TLE detection for algorithm escalation**: When status is TLE, the
   prompt template in prompts.py already includes special instructions
   to use a fundamentally different algorithm. But additionally, when
   escalating tiers after TLE, we should note in the prompt that
   previous attempts were all too slow.

5. **Keyboard interrupt (Ctrl+C)**: Caught at the solve_contest level.
   Saves current state and prints report before exiting.

6. **Browser crash mid-question**: _run_agent_task in browser agent
   handles this, but orchestrator should also be resilient:
   ```python
   try:
       run_result = await self.agent.paste_and_run(...)
   except Exception as e:
       logger.error(f"Unexpected error: {e}")
       # Try to reinitialize and continue
       try:
           await self.agent.close()
           await self.agent.initialize()
       except:
           pass
       continue
   ```

7. **Contest page returns no questions** (wrong slug, contest doesn't
   exist, or future contest): Caught early with clear error message.

8. **One question fails but others can still be solved**: Each question
   is independent. If Q2 fails, Q3 still gets attempted.

9. **Resuming a partially completed run**: is_solved() check at the
   start of _solve_question skips already-solved problems. User can
   re-run the same contest and it picks up where it left off.

10. **LLM returns same broken solution repeatedly**: The retry prompt
    includes the previous code AND why it failed, so the model should
    try something different. But if it's stuck in a loop (same output
    twice in a row), we could detect this and force escalation:
    ```python
    if code.strip() == previous_code.strip():
        console.print("  [yellow]LLM returned same code. "
                     "Forcing tier escalation.[/yellow]")
        break  # Break inner retry loop, escalate to next tier
    ```

11. **Very long problem descriptions that exceed context**: The prompt
    construction prioritizes: description > examples > constraints.
    If total text exceeds ~12000 chars, truncate constraints first,
    then examples (keeping first 2).

12. **Problem has interactive format** (some LC problems require
    interactive I/O): These are rare in contests but exist. The agent
    should still try — worst case it fails and we move on.
```

---

## PROMPT 6: Main Entry Point & Logger

```
Implement src/main.py (CLI entry point) and src/utils/logger.py.

== FILE: src/utils/logger.py ==

```python
"""Simple structured logger using rich."""
import logging
from rich.logging import RichHandler
from src.config import LOG_LEVEL  # Add LOG_LEVEL = "INFO" to config.py

def setup_logger(name: str = "leetcode-agent") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    return logging.getLogger(name)

logger = setup_logger()
```

Add `LOG_LEVEL = "INFO"` to src/config.py.

== FILE: src/main.py ==

```python
"""
LeetCode Contest Auto-Solver Agent — CLI Entry Point

Usage:
  python -m src.main weekly-contest-430
  python -m src.main weekly-contest-430 --question 3
  python -m src.main biweekly-contest-150 --skip-youtube
  python -m src.main weekly-contest-430 --debug
"""
import asyncio
import argparse
import sys
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="LeetCode Contest Auto-Solver Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main weekly-contest-430
  python -m src.main weekly-contest-430 -q 3
  python -m src.main weekly-contest-430 --skip-youtube
  python -m src.main weekly-contest-430 --debug
        """
    )
    parser.add_argument(
        "contest",
        help="Contest slug (e.g., weekly-contest-430, biweekly-contest-150)"
    )
    parser.add_argument(
        "-q", "--question", type=int, choices=[1, 2, 3, 4],
        help="Solve only question N (1-4)"
    )
    parser.add_argument(
        "--skip-youtube", action="store_true",
        help="Disable YouTube transcript fallback"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging and slow mode"
    )
    parser.add_argument(
        "--language", default=None,
        help="Override primary language (cpp, python3, java, etc.)"
    )

    args = parser.parse_args()

    # Debug mode
    if args.debug:
        import src.config as cfg
        cfg.LOG_LEVEL = "DEBUG"
        cfg.BROWSER_HEADLESS = False
        cfg.DELAY_BETWEEN_RETRIES = 5
        cfg.DELAY_BETWEEN_AGENT_CALLS = 4

    # Language override
    if args.language:
        import src.config as cfg
        cfg.PRIMARY_LANGUAGE = args.language

    # Run
    from src.orchestrator import Orchestrator
    orch = Orchestrator()

    try:
        asyncio.run(orch.solve_contest(
            contest_slug=args.contest,
            skip_youtube=args.skip_youtube,
            only_question=args.question,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/bold red]")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Also create a top-level convenience script:

== FILE: run.py (project root) ==
```python
"""Convenience runner. `python run.py weekly-contest-430`"""
from src.main import main
main()
```
```

---

## PROMPT 7: Integration Test Script

```
Create test_pipeline.py at the project root for testing each module
independently and as an integrated pipeline.

```python
"""
Integration test for the LeetCode agent.
Tests each module step by step against a known easy problem (Two Sum).

Usage:
  python test_pipeline.py              # Run all tests
  python test_pipeline.py --step 2     # Run only step 2
  python test_pipeline.py --step 1-3   # Run steps 1 through 3
"""
import asyncio
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

async def test_step_1_llm():
    """Test: Can we call each LLM provider?"""
    console.print(Panel("Step 1: Testing LLM Providers", style="bold"))

    from src.llm.factory import get_llm

    # Test OpenAI
    console.print("  Testing OpenAI...")
    try:
        llm = get_llm("openai", "gpt-4o-mini")
        resp = await llm.generate("Say hello.", "Say hello in 5 words.")
        console.print(f"  [green]✅ OpenAI: {resp.content[:50]}[/green]")
    except Exception as e:
        console.print(f"  [red]❌ OpenAI failed: {e}[/red]")

    # Test Anthropic (if key available)
    console.print("  Testing Anthropic...")
    try:
        llm = get_llm("anthropic", "claude-sonnet-4-20250514")
        resp = await llm.generate("Say hello.", "Say hello in 5 words.")
        console.print(f"  [green]✅ Anthropic: {resp.content[:50]}[/green]")
    except Exception as e:
        console.print(f"  [yellow]⚠️ Anthropic: {e}[/yellow]")

    # Test Gemini (if key available)
    console.print("  Testing Gemini...")
    try:
        llm = get_llm("gemini", "gemini-2.0-flash")
        resp = await llm.generate("Say hello.", "Say hello in 5 words.")
        console.print(f"  [green]✅ Gemini: {resp.content[:50]}[/green]")
    except Exception as e:
        console.print(f"  [yellow]⚠️ Gemini: {e}[/yellow]")


async def test_step_2_code_gen():
    """Test: Can we generate a valid C++ solution?"""
    console.print(Panel("Step 2: Testing Code Generation", style="bold"))

    from src.code_gen.solver import Solver

    solver = Solver()
    question = """
TITLE: Two Sum
DESCRIPTION: Given an array of integers nums and an integer target,
return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution,
and you may not use the same element twice.
EXAMPLES:
Example 1: Input: nums = [2,7,11,15], target = 9  Output: [0,1]
Example 2: Input: nums = [3,2,4], target = 6  Output: [1,2]
CONSTRAINTS: 2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9
"""
    code, resp = await solver.generate_solution(
        tier="cheap", question_text=question, language="cpp"
    )
    console.print(f"  Generated code ({len(code)} chars):")
    console.print(f"  [dim]{code[:200]}...[/dim]")
    assert "class Solution" in code, "No Solution class!"
    console.print(f"  [green]✅ Valid C++ code generated[/green]")


async def test_step_3_browser():
    """Test: Can browser agent read a LeetCode problem?"""
    console.print(Panel("Step 3: Testing Browser Agent", style="bold"))

    from src.browser.agent import BrowserAgent

    agent = BrowserAgent()
    try:
        await agent.initialize()

        console.print("  Checking login...")
        status = await agent.ensure_logged_in()
        console.print(f"  Login: {status}")
        if status != "LOGGED_IN":
            console.print("  [yellow]Log into LeetCode in the browser, "
                         "then press Enter[/yellow]")
            input()

        console.print("  Reading Two Sum problem...")
        details = await agent.get_question_details(
            "https://leetcode.com/problems/two-sum/"
        )
        if details and details.get('title'):
            console.print(f"  [green]✅ Got problem: {details['title']}[/green]")
            console.print(f"  [dim]Examples: {len(details.get('examples', []))}[/dim]")
        else:
            console.print(f"  [red]❌ Failed to read problem[/red]")
    finally:
        await agent.close()


async def test_step_4_paste_and_run():
    """Test: Can we paste code and run it on LeetCode?"""
    console.print(Panel("Step 4: Testing Paste & Run", style="bold"))

    from src.browser.agent import BrowserAgent

    # Known correct C++ solution for Two Sum
    correct_code = """
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> m;
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (m.count(complement)) {
                return {m[complement], i};
            }
            m[nums[i]] = i;
        }
        return {};
    }
};
"""
    agent = BrowserAgent()
    try:
        await agent.initialize()
        console.print("  Pasting correct solution and running...")
        result = await agent.paste_and_run(
            "https://leetcode.com/problems/two-sum/",
            correct_code, "cpp"
        )
        console.print(f"  Result: {result}")
        if result and result.get('all_passed'):
            console.print(f"  [green]✅ Code ran and passed![/green]")
        else:
            console.print(f"  [yellow]⚠️ Unexpected result[/yellow]")
    finally:
        await agent.close()


async def test_step_5_full_pipeline():
    """Test: Full pipeline on Two Sum (not via contest, direct problem)."""
    console.print(Panel("Step 5: Full Pipeline Test (Two Sum)", style="bold"))
    console.print("  [dim]This tests: generate → paste → run → submit[/dim]")

    from src.browser.agent import BrowserAgent
    from src.code_gen.solver import Solver

    agent = BrowserAgent()
    solver = Solver()

    try:
        await agent.initialize()

        # Read problem
        details = await agent.get_question_details(
            "https://leetcode.com/problems/two-sum/"
        )
        if not details:
            console.print("[red]Failed to read problem[/red]")
            return

        # Build question text
        from src.orchestrator import Orchestrator
        orch = Orchestrator()
        question_text = orch._build_question_text(details)

        # Generate C++ solution
        code, _ = await solver.generate_solution(
            tier="cheap", question_text=question_text, language="cpp"
        )
        console.print(f"  Generated solution, pasting...")

        # Run on LC
        result = await agent.paste_and_run(
            "https://leetcode.com/problems/two-sum/", code, "cpp"
        )
        console.print(f"  Run result: {result}")

        if result and result.get('all_passed'):
            submit = await agent.submit_solution(
                "https://leetcode.com/problems/two-sum/"
            )
            console.print(f"  Submit result: {submit}")
            if submit and submit.get('all_passed'):
                console.print("[bold green]✅ FULL PIPELINE WORKS![/bold green]")
            else:
                console.print("[yellow]Run passed but submit failed[/yellow]")
        else:
            console.print("[yellow]Run didn't pass — but pipeline mechanics work[/yellow]")

    finally:
        await agent.close()


# ── Runner ──
STEPS = {
    1: test_step_1_llm,
    2: test_step_2_code_gen,
    3: test_step_3_browser,
    4: test_step_4_paste_and_run,
    5: test_step_5_full_pipeline,
}

async def run_tests(steps):
    for step_num in steps:
        if step_num in STEPS:
            await STEPS[step_num]()
            console.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", help="Step number or range (e.g., 2 or 1-3)")
    args = parser.parse_args()

    if args.step:
        if '-' in args.step:
            start, end = map(int, args.step.split('-'))
            steps = range(start, end + 1)
        else:
            steps = [int(args.step)]
    else:
        steps = sorted(STEPS.keys())

    asyncio.run(run_tests(steps))
```
```

---

## EXECUTION ORDER

Feed prompts into Claude Code sequentially. After EACH prompt, test:

| # | Prompt | Test Command | What to Verify |
|---|--------|--------------|----------------|
| 1 | Config + LLM layer | `python test_pipeline.py --step 1` | API keys work, all providers respond |
| 2 | Browser agent | `python test_pipeline.py --step 3` | Browser opens, reads LC problem |
| 3 | Code generator | `python test_pipeline.py --step 2` | Valid C++ code generated |
| 4 | Storage | (tested inline) | JSON saved/loaded correctly |
| 5 | Orchestrator | `python test_pipeline.py --step 5` | Full generate→paste→run→submit works |
| 6 | Main CLI + Logger | `python -m src.main weekly-contest-420 -q 1 --debug` | Real contest solve |
| 7 | Integration tests | `python test_pipeline.py` | All steps green |

---

## COMPREHENSIVE EDGE CASE REFERENCE

These are ALL the edge cases built into the prompts above, compiled as
a checklist for verification:

### LLM Layer
- [ ] API key missing → clear error message
- [ ] API key invalid → clear error message
- [ ] Rate limit (429) → exponential backoff (2s, 4s, 8s, 16s)
- [ ] Request timeout → retry once, then raise
- [ ] Empty response → retry once
- [ ] Model not found → clear error message
- [ ] Context length exceeded → truncate intelligently
- [ ] Network error → retry once, then clear message

### Browser Agent
- [ ] Agent returns text instead of JSON → 3-strategy JSON extractor
- [ ] Browser crashes → detect, reinitialize, retry
- [ ] Agent task times out (120s) → retry once
- [ ] LeetCode popup/modal blocks page → prompt says dismiss first
- [ ] Code paste fails (wrong content in editor) → prompt says verify & retry
- [ ] Language dropdown doesn't switch → prompt explicitly checks
- [ ] Old results shown in panel → prompt says wait for loading spinner
- [ ] Contest doesn't exist → detect "not found" in response
- [ ] Problem is premium-locked → detect in response, skip
- [ ] CAPTCHA / anti-bot → 3 consecutive failures trigger human intervention
- [ ] Session expires → detect login redirect, pause for manual login
- [ ] YouTube video has no transcript → return None, try next video
- [ ] YouTube video is region-blocked → same
- [ ] Code contains backticks → sanitize before embedding in prompt
- [ ] Agent hallucinates results → submit_solution as separate verification

### Code Generation
- [ ] LLM returns markdown-wrapped code → strip fences
- [ ] LLM returns explanation + code → find code start, strip preamble
- [ ] LLM returns code + trailing explanation → strip trailing text
- [ ] LLM returns code in wrong language → structural validation catches it
- [ ] Python code has syntax error → ast.parse catches it
- [ ] C++ code has unbalanced braces → brace count check
- [ ] LLM returns same broken code twice → detect, force tier escalation
- [ ] Very long problem description → truncate constraints/examples first

### Orchestrator
- [ ] Difficulty not detected → default to Medium escalation
- [ ] All tiers exhausted → YouTube fallback (if enabled)
- [ ] YouTube has no videos yet → wait 5 min, retry once
- [ ] All YouTube videos have no transcript → give up gracefully
- [ ] LeetCode rate limits submissions → detect, wait 60s
- [ ] Ctrl+C during execution → save state, print report, clean exit
- [ ] Run same contest twice → skip already-solved problems
- [ ] C++ conversion to Python fails → retry with mid-tier model
- [ ] One question fails → continue to next question
- [ ] Browser dies mid-question → reinitialize, continue
- [ ] LLM throws unexpected exception → catch, log, continue to next retry

### Storage
- [ ] First run (no existing data) → create fresh JSON
- [ ] Re-run (data exists) → append new attempts
- [ ] Ctrl+C during write → atomic write prevents corruption
- [ ] Unicode in problem titles → json ensure_ascii=False
- [ ] Solutions directory doesn't exist → create it
