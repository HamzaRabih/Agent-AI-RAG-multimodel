from __future__ import annotations

import streamlit as st

from src.config import get_default_gemini_api_key, get_default_openai_api_key

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"]
OSS_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
]


def render_sidebar() -> dict:
    st.sidebar.title("Configuration")

    provider = st.sidebar.selectbox(
        "Provider LLM",
        options=["opensource-gemini", "openai"],
        format_func=lambda p: "OpenAI (payant)" if p == "openai" else "Google Gemini",
    )

    model_options = OPENAI_MODELS if provider == "openai" else GEMINI_MODELS
    model_name = st.sidebar.selectbox("Modele", options=model_options)

    st.sidebar.markdown("### API Keys")
    openai_api_key = st.sidebar.text_input(
        "OPENAI_API_KEY",
        value=get_default_openai_api_key(),
        type="password",
    )
    gemini_api_key = st.sidebar.text_input(
        "GEMINI_API_KEY",
        value=get_default_gemini_api_key(),
        type="password",
    )

    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1)

    st.sidebar.markdown("### Base vectorielle")
    create_db = st.sidebar.button("Creer / Mettre a jour la base")
    reset_db = st.sidebar.button("Reinitialiser la base")

    return {
        "provider": provider,
        "model_name": model_name,
        "openai_api_key": openai_api_key,
        "gemini_api_key": gemini_api_key,
        "temperature": temperature,
        "create_db": create_db,
        "reset_db": reset_db,
    }
