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
