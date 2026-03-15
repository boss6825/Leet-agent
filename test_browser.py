# test_browser.py — save and run: python test_browser.py
import asyncio
from src.browser.agent import BrowserAgent

async def test():
    agent = BrowserAgent()
    try:
        await agent.initialize()
        print("Checking login...")
        status = await agent.ensure_logged_in()
        print(f"Login status: {status}")
        
        if status == "LOGGED_IN":
            print("Reading Two Sum problem...")
            details = await agent.get_question_details("https://leetcode.com/problems/two-sum/")
            if details:
                print(f"Got: {details.get('title')} ({details.get('difficulty')})")
                print(f"Examples: {len(details.get('examples', []))}")
            else:
                print("Failed to read problem")
    finally:
        await agent.close()

asyncio.run(test())