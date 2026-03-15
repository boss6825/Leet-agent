import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# LLM Model Configuration
# ──────────────────────────────────────────────
MODEL_TIERS = {
    "cheap": {"provider": "openai", "model": "gpt-4o-mini"},
    "mid": {"provider": "openai", "model": "gpt-4o"},
    "expensive": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
}

BROWSER_AGENT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
}

# ──────────────────────────────────────────────
# Retry / Escalation Strategy
# ──────────────────────────────────────────────
ESCALATION_STRATEGY = {
    "Easy": [("cheap", 3), ("mid", 2), ("expensive", 2)],
    "Medium": [("cheap", 2), ("mid", 3), ("expensive", 3)],
    "Hard": [("cheap", 1), ("mid", 2), ("expensive", 4)],
}
DEFAULT_ESCALATION = [("cheap", 2), ("mid", 3), ("expensive", 3)]

# ──────────────────────────────────────────────
# Language Configuration
# ──────────────────────────────────────────────
PRIMARY_LANGUAGE = "cpp"
CONVERT_TO_LANGUAGES = ["python3"]
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
BROWSER_HEADLESS = False
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
DELAY_BETWEEN_AGENT_CALLS = 2
DELAY_BETWEEN_RETRIES = 3
DELAY_BETWEEN_QUESTIONS = 5
LC_RUN_TIMEOUT = 30
LC_SUBMIT_TIMEOUT = 45
YOUTUBE_SEARCH_WAIT = 300
MAX_YOUTUBE_VIDEOS_TO_TRY = 3
MAX_YOUTUBE_RETRIES_PER_VIDEO = 3

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ──────────────────────────────────────────────
# API Keys (from .env)
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ──────────────────────────────────────────────
# LeetCode Credentials (from .env)
# ──────────────────────────────────────────────
LEETCODE_USERNAME = os.getenv("LEETCODE_USERNAME")
LEETCODE_PASSWORD = os.getenv("LEETCODE_PASS")
