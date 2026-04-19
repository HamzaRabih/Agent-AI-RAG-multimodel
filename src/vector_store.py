from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

COLLECTION_NAME = "agentic_rag_docs"


def get_vector_store(persist_dir: Path, embeddings: Embeddings) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def reset_vector_store(persist_dir: Path) -> None:
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)


def index_documents(store: Chroma, chunks: list[Document]) -> int:
    if not chunks:
        return 0
    store.add_documents(chunks)#chaque chunk est transformé en embedding
                               # il est stocké dans Chroma
                               # indexé pour la recherche vectorielle
    return len(chunks)


def get_retriever(store: Chroma, k: int = 4,ty: str='similarity') -> VectorStoreRetriever:
    return store.as_retriever(
        search_kwargs={"k": k},#retourne les k documents les plus similaires
        search_type=str(ty)
        )


def is_store_empty(store: Chroma) -> bool:
    try:
        return store._collection.count() == 0
    except Exception:
        return True
