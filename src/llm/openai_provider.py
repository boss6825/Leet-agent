import asyncio
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
from .base import BaseLLM, LLMResponse
from src.config import OPENAI_API_KEY
from src.utils.logger import logger


class OpenAILLM(BaseLLM):
    def __init__(self, model: str):
        super().__init__(model=model, provider="openai")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

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
                content = choice.message.content or ""

                if not content.strip() and attempt < max_retries - 1:
                    logger.warning("Empty response from OpenAI, retrying...")
                    await asyncio.sleep(base_delay)
                    continue

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="openai",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    cost_usd=None,
                )

            except RateLimitError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"OpenAI rate limited. Retrying in {delay}s...")
                await asyncio.sleep(delay)

            except APITimeoutError as e:
                last_error = e
                if attempt < 1:
                    logger.warning("OpenAI timeout. Retrying once...")
                    await asyncio.sleep(base_delay)
                    continue
                raise

            except APIConnectionError as e:
                last_error = e
                if attempt < 1:
                    logger.warning("Connection error. Retrying once...")
                    await asyncio.sleep(base_delay)
                    continue
                raise RuntimeError("Check your internet connection.") from e

            except APIStatusError as e:
                if "model_not_found" in str(e).lower() or "does not exist" in str(e).lower():
                    raise ValueError(f"Model not found: {self.model}") from e
                last_error = e
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"OpenAI API error: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise last_error
