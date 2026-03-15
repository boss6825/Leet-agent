import asyncio
import anthropic
from .base import BaseLLM, LLMResponse
from src.config import ANTHROPIC_API_KEY
from src.utils.logger import logger


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str):
        super().__init__(model=model, provider="anthropic")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env file")
        self.client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY, timeout=60.0
        )

    async def generate(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.0,
                       max_tokens: int = 4096) -> LLMResponse:
        return await self._retry_with_backoff(
            system_prompt, user_prompt, temperature, max_tokens
        )

    async def _retry_with_backoff(self, system_prompt, user_prompt,
                                   temperature, max_tokens,
                                   max_retries=4, base_delay=2):
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.content[0].text if response.content else ""

                if not content.strip() and attempt < max_retries - 1:
                    logger.warning("Empty response from Anthropic, retrying...")
                    await asyncio.sleep(base_delay)
                    continue

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="anthropic",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=None,
                )

            except anthropic.RateLimitError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Anthropic rate limited. Retrying in {delay}s...")
                await asyncio.sleep(delay)

            except anthropic.APITimeoutError as e:
                last_error = e
                if attempt < 1:
                    logger.warning("Anthropic timeout. Retrying once...")
                    await asyncio.sleep(base_delay)
                    continue
                raise

            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < 1:
                    logger.warning("Connection error. Retrying once...")
                    await asyncio.sleep(base_delay)
                    continue
                raise RuntimeError("Check your internet connection.") from e

            except anthropic.AuthenticationError as e:
                raise ValueError(f"Invalid Anthropic API key") from e

            except anthropic.APIStatusError as e:
                if "model_not_found" in str(e).lower() or "does not exist" in str(e).lower():
                    raise ValueError(f"Model not found: {self.model}") from e
                last_error = e
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Anthropic API error: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise last_error
