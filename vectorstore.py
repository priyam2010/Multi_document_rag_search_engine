import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_DIR = "faiss_index"


def get_embeddings():
    """
    Returns HuggingFace embeddings safely on CPU
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def index_documents(chunks):
    """
    Index document chunks into FAISS
    """

    # ✅ HARD STOP if no chunks
    if not chunks:
        raise ValueError("No chunks received for indexing.")

    # ✅ Extract valid text only
    texts = []
    metadatas = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if content:
            texts.append(content)
            metadatas.append(chunk.get("metadata", {}))

    # ✅ HARD STOP if all chunks are empty
    if not texts:
        raise ValueError("All document chunks are empty. Check PDF text extraction.")

    embeddings = get_embeddings()

    # ✅ FAISS indexing
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local(FAISS_DIR)


def load_faiss_index():
    """
    Load FAISS index safely
    """

    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(
            "FAISS index not found. Please index documents first."
        )

    embeddings = get_embeddings()

    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )






