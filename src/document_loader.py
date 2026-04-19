from __future__ import annotations

from pathlib import Path # gestion propre des chemins

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from PyPDF2 import PdfReader


def validate_pdf(path: Path) -> tuple[bool, str]:
    try:
        reader = PdfReader(str(path))
        _ = len(reader.pages) #_ varible spéciale par convention ,Je stocke la valeur, mais je ne vais pas l’utiliser
        return True, "OK"
    except Exception as exc:  # pragma: no cover - defensive path
        return False, str(exc)


def load_documents(pdf_directory: Path) -> list[Document]:
    loader = PyPDFDirectoryLoader(str(pdf_directory))
    return loader.load()


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    encoding:str="o200k_base"
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
