from __future__ import annotations

import streamlit as st

from src.chains import generate_rag_answer
from src.config import CHROMA_DB_DIR, RAW_DOCS_DIR, ensure_project_dirs
from src.document_loader import load_documents, split_documents, validate_pdf
from src.embedding_model import get_embedding_model
from src.llm_factory import create_llm
from src.vector_store import get_retriever, get_vector_store, index_documents, is_store_empty, reset_vector_store
from ui.chat import init_chat_state, push_assistant_message, push_user_message, render_history
from ui.sidebar import render_sidebar
from ui.uploader import render_uploader


def run_app() -> None:
    st.set_page_config(page_title="Agentic-RAG-Pro", page_icon="💬", layout="wide")
    st.title("Agentic-RAG-Pro")
    st.caption("RAG avec Streamlit, LangChain, ChromaDB, BAAI/bge-m3 et switch de modeles LLM")

    ensure_project_dirs()
    init_chat_state()

    settings = render_sidebar()

    if settings["reset_db"]:
        reset_vector_store(CHROMA_DB_DIR)
        st.session_state.chat_history = []
        st.success("Base vectorielle reinitialisee.")

    st.markdown("## 1) Upload des documents")
    saved_files = render_uploader(RAW_DOCS_DIR)
    if saved_files:
        invalid_files = []
        for path in saved_files:
            valid, message = validate_pdf(path)
            if not valid:
                invalid_files.append((path.name, message))

        if invalid_files:
            for file_name, error in invalid_files:
                st.error(f"PDF invalide: {file_name} ({error})")

    st.markdown("## 2) Creation / mise a jour de la base vectorielle")
    if settings["create_db"]:
        with st.spinner("Indexation des documents en cours..."):
            docs = load_documents(RAW_DOCS_DIR)
            chunks = split_documents(docs)
            embeddings = get_embedding_model()
            store = get_vector_store(CHROMA_DB_DIR, embeddings)
            total = index_documents(store, chunks)
        st.success(f"Indexation terminee. {total} chunks ajoutes.")

    st.markdown("## 3) Chat RAG")
    render_history()

    user_question = st.chat_input("Pose ta question sur les documents")
    if user_question:
        push_user_message(user_question)
        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            llm = create_llm(
                provider=settings["provider"],
                model_name=settings["model_name"],
                temperature=settings["temperature"],
                openai_api_key=settings["openai_api_key"],
                gemini_api_key=settings["gemini_api_key"],
            )

            embeddings = get_embedding_model()
            store = get_vector_store(CHROMA_DB_DIR, embeddings)
            if is_store_empty(store):
                raise ValueError("La base vectorielle est vide. Ajoutez des documents puis creez la base.")

            retriever = get_retriever(store, k=4)
            answer, sources = generate_rag_answer(
                question=user_question,
                llm=llm,
                retriever=retriever,
                chat_history=st.session_state.chat_history,
            )

            full_answer = answer
            if sources:
                preview = "\n\n".join(f"- {s[:180]}..." for s in sources[:3])
                full_answer += f"\n\nSources (extrait):\n{preview}"

            with st.chat_message("assistant"):
                st.markdown(full_answer)
            push_assistant_message(full_answer)

        except Exception as exc:
            error_msg = f"Erreur: {exc}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            push_assistant_message(error_msg)


if __name__ == "__main__":
    run_app()
