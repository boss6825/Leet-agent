from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


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
