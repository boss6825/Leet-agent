import asyncio
from google import genai
from google.genai import types
from .base import BaseLLM, LLMResponse
from src.config import GEMINI_API_KEY
from src.utils.logger import logger


class GeminiLLM(BaseLLM):
    def __init__(self, model: str):
        super().__init__(model=model, provider="gemini")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env file")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

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
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )

                content = response.text if response.text else ""

                if not content.strip() and attempt < max_retries - 1:
                    logger.warning("Empty response from Gemini, retrying...")
                    await asyncio.sleep(base_delay)
                    continue

                # Extract token counts from usage metadata
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="gemini",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=None,
                )

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                if "429" in str(e) or "resource exhausted" in error_str:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini rate limited. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                if "not found" in error_str or "does not exist" in error_str:
                    raise ValueError(f"Model not found: {self.model}") from e

                if "deadline" in error_str or "timeout" in error_str:
                    if attempt < 1:
                        logger.warning("Gemini timeout. Retrying once...")
                        await asyncio.sleep(base_delay)
                        continue
                    raise

                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Gemini error: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise last_error
