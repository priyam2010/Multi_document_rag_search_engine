from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from config import FAISS_DIR


def index_documents(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(FAISS_DIR)


def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if not os.path.exists(FAISS_DIR):
        return None
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
