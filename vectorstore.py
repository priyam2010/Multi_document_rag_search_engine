import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import FAISS_DIR, EMBEDDING_MODEL

def _embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def index_documents(chunks):
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=_embeddings(),
        metadatas=metadatas
    )

    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)

def load_faiss_index():
    if not os.path.exists(FAISS_DIR):
        return None

    return FAISS.load_local(
        FAISS_DIR,
        _embeddings(),
        allow_dangerous_deserialization=True
    )


