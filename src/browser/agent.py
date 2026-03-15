import asyncio
from browser_use import Agent, BrowserSession
from src.config import (
    BROWSER_AGENT_CONFIG,
    BROWSER_HEADLESS,
    DELAY_BETWEEN_AGENT_CALLS,
    LC_LANGUAGE_NAMES,
    LEETCODE_USERNAME,
    LEETCODE_PASSWORD,
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

        browser-use has its own LLM wrappers. We use the
        configured BROWSER_AGENT_CONFIG to pick the right one.
        """
        provider = BROWSER_AGENT_CONFIG["provider"]
        model = BROWSER_AGENT_CONFIG["model"]

        # browser-use 0.12+ has its own LLM wrappers under browser_use.llm
        if provider == "openai":
            from browser_use.llm.openai.chat import ChatOpenAI
            self.llm = ChatOpenAI(model=model)
        elif provider == "anthropic":
            from browser_use.llm.anthropic.chat import ChatAnthropic
            self.llm = ChatAnthropic(model=model)
        elif provider == "gemini":
            from browser_use.llm.google.chat import ChatGoogle
            self.llm = ChatGoogle(model=model)
        else:
            raise ValueError(f"Unsupported browser agent provider: {provider}")

        self.browser = BrowserSession(headless=BROWSER_HEADLESS)
        await self.browser.start()
        self._initialized = True
        logger.info(f"Browser agent initialized with {provider}:{model}")

    async def _run_agent_task(self, task: str, max_retries: int = 2) -> str | None:
        """
        Run a single browser-use agent task with retry logic.

        This is the core method all public methods call.
        Returns the agent's string response, or None on failure.
        """
        if not self._initialized:
            await self.initialize()

        for attempt in range(max_retries):
            try:
                agent = Agent(
                    task=task,
                    llm=self.llm,
                    browser_session=self.browser,
                )
                result = await asyncio.wait_for(
                    agent.run(), timeout=120  # 2 minute timeout per agent task
                )

                # browser-use agent.run() returns an AgentHistoryList
                # Extract the actual text result
                result_text = str(result)

                if not result_text or result_text.strip() == "":
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Empty agent response, retrying... "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
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
        """Check if user is logged into LeetCode. Returns 'LOGGED_IN' or 'NOT_LOGGED_IN'.
        If not logged in and credentials are available, logs in automatically."""
        result = await self._run_agent_task(
            """
Go to https://leetcode.com

Check if the user is logged in. Signs of being logged in:
- A profile avatar or username in the top-right area of the navigation bar
- NO prominent "Sign In" or "Register" or "Sign up" buttons in the navigation

Signs of NOT being logged in:
- A "Sign In" or "Premium" or "Register" button visible in the navigation
- No user avatar/profile icon

If the user IS logged in, return exactly: LOGGED_IN
If the user is NOT logged in, return exactly: NOT_LOGGED_IN
"""
        )
        if result and "LOGGED_IN" in result and "NOT_LOGGED_IN" not in result:
            return "LOGGED_IN"

        # Auto-login if credentials are available
        if LEETCODE_USERNAME and LEETCODE_PASSWORD:
            logger.info("Not logged in. Attempting auto-login...")
            login_ok = await self._login()
            if login_ok:
                return "LOGGED_IN"

        return "NOT_LOGGED_IN"

    async def _login(self) -> bool:
        """Log into LeetCode using credentials from .env.
        Uses sensitive_data so credentials never appear in prompts/logs."""
        result = await self._run_agent_task_with_secrets(
            task="""
Go to https://leetcode.com/accounts/login/

STEP 1: Wait for the login page to fully load. You should see email/username
and password input fields.

STEP 2: Click on the username/email input field and type the value of
x_username (use the secret placeholder).

STEP 3: Click on the password input field and type the value of
x_password (use the secret placeholder).

STEP 4: Click the "Sign In" button to submit the login form.

STEP 5: Wait for the page to redirect after login (up to 15 seconds).
Check if login was successful:
- If you see a profile avatar or username in the nav bar: return LOGGED_IN
- If you see an error message like "Invalid credentials": return LOGIN_FAILED
- If you see a CAPTCHA or verification challenge: return CAPTCHA_REQUIRED

Return exactly one of: LOGGED_IN, LOGIN_FAILED, CAPTCHA_REQUIRED
""",
            secrets={
                "x_username": LEETCODE_USERNAME,
                "x_password": LEETCODE_PASSWORD,
            },
            max_retries=2,
        )
        if result and "LOGGED_IN" in result:
            logger.info("Auto-login successful!")
            return True
        if result and "CAPTCHA" in result:
            logger.warning("CAPTCHA detected during login. Manual intervention needed.")
        else:
            logger.error(f"Auto-login failed: {result}")
        return False

    async def _run_agent_task_with_secrets(
        self, task: str, secrets: dict[str, str], max_retries: int = 2
    ) -> str | None:
        """Like _run_agent_task but passes sensitive_data to the agent
        so credentials are injected securely without appearing in the prompt."""
        if not self._initialized:
            await self.initialize()

        for attempt in range(max_retries):
            try:
                agent = Agent(
                    task=task,
                    llm=self.llm,
                    browser_session=self.browser,
                    sensitive_data=secrets,
                )
                result = await asyncio.wait_for(agent.run(), timeout=120)
                result_text = str(result)

                if not result_text or result_text.strip() == "":
                    if attempt < max_retries - 1:
                        await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                        continue
                    return None

                await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                return result_text

            except asyncio.TimeoutError:
                logger.error(f"Login task timed out (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                    continue
                return None

            except Exception as e:
                logger.error(f"Login task failed: {e} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(DELAY_BETWEEN_AGENT_CALLS)
                    continue
                return None

        return None

    async def get_contest_questions(self, contest_slug: str) -> list[dict] | None:
        """
        Get all problem links from a contest page.
        Returns: [{"title": "...", "url": "..."}, ...] or None
        """
        result = await self._run_agent_task(
            f"""
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
"""
        )
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
        result = await self._run_agent_task(
            f"""
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
"""
        )
        if not result:
            return None
        parsed = extract_json(result)
        if isinstance(parsed, dict) and parsed.get("title"):
            return parsed
        logger.error(f"Failed to parse question details. Raw: {result[:500]}")
        return None

    async def paste_and_run(
        self, problem_url: str, code: str, language: str
    ) -> dict | None:
        """
        Navigate to problem, switch language, paste code, click Run,
        and read results.
        """
        lc_lang_name = LC_LANGUAGE_NAMES.get(language, language)
        safe_code = sanitize_code_for_prompt(code)

        result = await self._run_agent_task(
            f"""
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
""",
            max_retries=3,
        )

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
        result = await self._run_agent_task(
            """
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
{
  "status": "Accepted" or "Wrong Answer" or "Runtime Error" or
            "Time Limit Exceeded" or "Memory Limit Exceeded",
  "all_passed": true/false,
  "passed_count": 45,
  "total_count": 60,
  "runtime_ms": 12,
  "memory_mb": 8.5,
  "failing_test": {
    "input": "the failing test input",
    "expected": "expected output",
    "actual": "your code's output"
  } or null,
  "error_message": "error details" or null
}

Return ONLY the JSON.
""",
            max_retries=3,
        )

        if not result:
            return None
        parsed = extract_json(result)
        if isinstance(parsed, dict):
            return parsed
        logger.error(f"Failed to parse submit results. Raw: {result[:500]}")
        return None

    async def find_youtube_editorial(
        self, problem_title: str, contest_slug: str
    ) -> list[str]:
        """Search YouTube for editorial videos for a specific problem."""
        search_query = f"{problem_title} leetcode solution {contest_slug}"
        result = await self._run_agent_task(
            f"""
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
"""
        )
        if not result:
            return []
        parsed = extract_json(result)
        if isinstance(parsed, list):
            return [
                url for url in parsed if isinstance(url, str) and "youtube.com" in url
            ]
        return []

    async def get_video_transcript(self, video_url: str) -> str | None:
        """
        Get transcript from a YouTube video.
        Tries YouTube's built-in transcript first, then Gemini Ask feature.
        """
        result = await self._run_agent_task(
            f"""
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
"""
        )
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
                await self.browser.stop()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self.browser = None
        self._initialized = False
