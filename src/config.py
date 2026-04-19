from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

#Charge les variables depuis un fichier .env
load_dotenv()

"""
dossier racine du projet
.parent.parent :remonte de 2 niveaux pour atteindre le dossier racine
"""

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"


#Créer une structure propre pour config runtime
"""
Exemple utilisation :settings = RuntimeSettings(
    provider="openai",
    model_name="gpt-4",
    temperature=0.2,
    openai_api_key="sk-xxx",
    gemini_api_key=None
)
"""
@dataclass
class RuntimeSettings:
    provider: str
    model_name: str
    temperature: float
    openai_api_key: str | None
    gemini_api_key: str | None


"""
Créer automatiquement les dossiers s ils n existent pas
parents=True: crée les dossiers parents si besoin
exist_ok=True: évite une erreur si le dossier existe déjà
"""
def ensure_project_dirs() -> None:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_default_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def get_default_gemini_api_key() -> str:
    # Prefer the standard key name, keep legacy token name as fallback.
    return os.getenv("GEMINI_API_KEY", os.getenv("GEMINI_API_TOKEN", ""))
