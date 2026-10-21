import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.llm.interface import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger("sonara.llm.gemini")


class GeminiProvider(LLMProvider):
    """
    Production Google Gemini LLM provider using the modern `google.genai` SDK.
    Supports structured output validation, token usage tracking, and bounded timeouts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._model_name = model_name or settings.GEMINI_MODEL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

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
                    provider_name="gemini",
                    detail="GEMINI_API_KEY is not configured.",
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as e:
                logger.error("google-genai SDK is not installed.")
                raise ProviderUnavailableError(
                    provider_name="gemini",
                    detail="google-genai package is not installed.",
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
        from google.genai import types

        # Build contents from messages: system prompt extracted if first message
        system_instruction = None
        user_parts = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                user_parts.append(f"{msg.role.upper()}: {msg.content}")

        prompt_text = "\n\n".join(user_parts)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt_text,
                config=config,
            )
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Gemini API request failed: {err_msg}")
            raise ProviderUnavailableError(
                provider_name="gemini",
                detail=f"Gemini API call failed: {err_msg}",
            ) from e

        content = response.text or ""
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

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
            logger.error(f"Gemini request timed out after {self.timeout_seconds}s")
            raise ProviderUnavailableError(
                provider_name="gemini",
                detail=f"Gemini API timed out after {self.timeout_seconds} seconds.",
            ) from e

