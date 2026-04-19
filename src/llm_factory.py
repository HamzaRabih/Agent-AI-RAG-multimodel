from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


GEMINI_MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-latest": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-1.5-pro-latest": "gemini-2.5-pro",
    "gemini-1.0-pro": "gemini-2.5-flash",
}


def _normalize_gemini_model_name(model_name: str) -> str:
    return GEMINI_MODEL_ALIASES.get(model_name.strip(), model_name.strip())


def create_llm(
    model_name: str,
    temperature: float,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    provider: str= "opensource-gemini",
) -> BaseChatModel:
    provider = provider.lower()

    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY manquante.")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key,
        )

    if provider == "opensource-gemini":
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY manquante.")
        normalized_model_name = _normalize_gemini_model_name(model_name)
        return ChatGoogleGenerativeAI(
            model=normalized_model_name,
            temperature=temperature,
            api_key=gemini_api_key,
        )

    raise ValueError(f"Provider non supporte: {provider}")
