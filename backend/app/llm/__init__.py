from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.interface import LLMMessage, LLMProvider, LLMResponse

_gemini_provider: GeminiProvider | None = None
_groq_provider: GroqProvider | None = None


def get_gemini_provider() -> GeminiProvider:
    global _gemini_provider
    if _gemini_provider is None:
        _gemini_provider = GeminiProvider()
    return _gemini_provider


def get_groq_provider() -> GroqProvider:
    global _groq_provider
    if _groq_provider is None:
        _groq_provider = GroqProvider()
    return _groq_provider


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "get_gemini_provider",
    "get_groq_provider",
]

