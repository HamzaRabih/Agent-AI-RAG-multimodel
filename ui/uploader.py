from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_uploader(raw_docs_dir: Path) -> list[Path]:
    st.subheader("Charger des PDF")
    uploaded_files = st.file_uploader(
        "Selectionnez un ou plusieurs fichiers PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    saved_paths: list[Path] = []
    if uploaded_files:
        for uploaded in uploaded_files:
            target = raw_docs_dir / uploaded.name
            target.write_bytes(uploaded.getvalue())
            saved_paths.append(target)

        st.success(f"{len(saved_paths)} fichier(s) enregistre(s) dans data/raw_docs.")

    return saved_paths
