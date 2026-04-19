from __future__ import annotations

import streamlit as st


def init_chat_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def render_history() -> None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def push_user_message(content: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": content})


def push_assistant_message(content: str) -> None:
    st.session_state.chat_history.append({"role": "assistant", "content": content})
