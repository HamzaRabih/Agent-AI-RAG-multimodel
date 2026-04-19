from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
    )
