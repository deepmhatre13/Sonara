import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.llm.interface import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger("sonara.llm.groq")


class GroqProvider(LLMProvider):
    """
    Optional Groq LLM provider fallback using `groq-python` SDK.
    Provides fast, resilient inference when Gemini encounters transient rate limits or outages.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self._model_name = model_name or settings.GROQ_MODEL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.is_configured:
                raise ProviderUnavailableError(
                    provider_name="groq",
                    detail="GROQ_API_KEY is not configured.",
                )
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError as e:
                logger.error("groq package is not installed.")
                raise ProviderUnavailableError(
                    provider_name="groq",
                    detail="groq package is not installed in the environment.",
                ) from e
        return self._client

    def _generate_sync(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        groq_messages = [{"role": m.role, "content": m.content} for m in messages]

        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": groq_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_schema:
            request_kwargs["response_format"] = {"type": "json_object"}

        try:
            chat_completion = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Groq API request failed: {err_msg}")
            raise ProviderUnavailableError(
                provider_name="groq",
                detail=f"Groq API call failed: {err_msg}",
            ) from e

        content = chat_completion.choices[0].message.content or ""
        prompt_tokens = None
        completion_tokens = None
        if hasattr(chat_completion, "usage") and chat_completion.usage:
            prompt_tokens = getattr(chat_completion.usage, "prompt_tokens", None)
            completion_tokens = getattr(chat_completion.usage, "completion_tokens", None)

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=self._model_name,
            provider_name=self.provider_name,
        )

    async def generate_response(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as e:
            logger.error(f"Groq request timed out after {self.timeout_seconds}s")
            raise ProviderUnavailableError(
                provider_name="groq",
                detail=f"Groq API timed out after {self.timeout_seconds} seconds.",
            ) from e

