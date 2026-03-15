"""Quick test for LLM providers. Run: python test_llm.py"""
import asyncio
from rich.console import Console

console = Console()


async def test():
    from src.llm.factory import get_llm

    # Test OpenAI
    console.print("\n[bold]Testing OpenAI (gpt-4o-mini)...[/bold]")
    try:
        llm = get_llm("openai", "gpt-4o-mini")
        resp = await llm.generate("You are a helpful assistant.", "Say hello in exactly 3 words.")
        console.print(f"  [green]Response: {resp.content}[/green]")
        console.print(f"  Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
    except Exception as e:
        console.print(f"  [red]Failed: {e}[/red]")

    # Test Anthropic
    console.print("\n[bold]Testing Anthropic (claude-sonnet-4-20250514)...[/bold]")
    try:
        llm = get_llm("anthropic", "claude-sonnet-4-20250514")
        resp = await llm.generate("You are a helpful assistant.", "Say hello in exactly 3 words.")
        console.print(f"  [green]Response: {resp.content}[/green]")
        console.print(f"  Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
    except Exception as e:
        console.print(f"  [yellow]Skipped/Failed: {e}[/yellow]")

    # Test Gemini
    console.print("\n[bold]Testing Gemini (gemini-2.0-flash)...[/bold]")
    try:
        llm = get_llm("gemini", "gemini-2.5-flash")
        resp = await llm.generate("You are a helpful assistant.", "Say hello in exactly 3 words.")
        console.print(f"  [green]Response: {resp.content}[/green]")
        console.print(f"  Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
    except Exception as e:
        console.print(f"  [yellow]Skipped/Failed: {e}[/yellow]")

    console.print("\n[bold green]LLM layer test complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(test())
