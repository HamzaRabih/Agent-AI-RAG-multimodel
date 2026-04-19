# Agentic-RAG-Pro

Application RAG basee sur:
- Streamlit
- LangChain
- ChromaDB
- Embeddings BAAI/bge-m3
- LLM payant: OpenAI Chat
- LLM open source: via Hugging Face Inference

## Structure

```text
Agentic-RAG-Pro/
├── .env
├── requirements.txt
├── main.py
├── src/
├── data/
│   ├── raw_docs/
│   └── chroma_db/
├── ui/
└── tests/
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Renseigner les cles API dans `.env` (optionnel), ou directement dans la sidebar Streamlit:

```env
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
```

## Lancement

```bash
streamlit run main.py
```

## Workflow

1. Uploader un ou plusieurs PDF.
2. Cliquer sur Creer / Mettre a jour la base pour construire l'index vectoriel.
3. Poser des questions dans le chat RAG.
4. Utiliser Reinitialiser la base en cas de mise a jour documentaire majeure.

## Notes

- La base Chroma est persistante dans `data/chroma_db`.
- Les PDF uploades sont stockes dans `data/raw_docs`.
- Le modele d'embedding est fixe sur `BAAI/bge-m3`.
