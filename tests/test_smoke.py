from src.config import CHROMA_DB_DIR, RAW_DOCS_DIR


def test_paths_are_defined() -> None:
    assert RAW_DOCS_DIR.name == "raw_docs"
    assert CHROMA_DB_DIR.name == "chroma_db"
