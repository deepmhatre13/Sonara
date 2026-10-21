from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str = Field(..., description="Role of message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class LLMResponse(BaseModel):
    content: str = Field(..., description="Generated text response")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_name: str
    provider_name: str
    raw_response: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract interface for LLM synthesis (Gemini, Groq fallback)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Generates completion from chat messages with optional Pydantic structured output."""
        pass
