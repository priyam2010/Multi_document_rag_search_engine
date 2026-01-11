from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WikipediaLoader
)

from langchain_core.documents import Document
from pathlib import Path
import tempfile
import os
import pdfplumber


# ------------------------------
# 1️⃣ Folder-based loader (optional, unchanged logic)
# ------------------------------
def load_local_documents(data_dir):
    documents = []

    if not Path(data_dir).exists():
        return documents  # fail safely

    for file in Path(data_dir).iterdir():
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file))
        else:
            continue

        docs = loader.load()
        for d in docs:
            d.metadata["source_type"] = "local"
            d.metadata["source_id"] = file.name

        documents.extend(docs)

    return documents


# ------------------------------
# 2️⃣ Streamlit Upload Loader (THIS FIXES YOUR ERROR)
# ------------------------------
def load_uploaded_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:

        # ---- PDF ----
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "upload",
                        "source_id": uploaded_file.name
                    }
                )
            )

        # ---- TXT ----
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "upload",
                        "source_id": uploaded_file.name
                    }
                )
            )

    return documents


# ------------------------------
# 3️⃣ Wikipedia Loader (unchanged)
# ------------------------------
def load_wikipedia(topic: str):
    loader = WikipediaLoader(query=topic, load_max_docs=2)
    docs = loader.load()

    for d in docs:
        d.metadata["source_type"] = "wikipedia"
        d.metadata["source_id"] = d.metadata.get("title", topic)

    return docs
