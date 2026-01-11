import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from config import FAISS_DIR

# Ensure your HuggingFace API key is set in env
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HF_API_KEY:
    raise ValueError("Please set HUGGINGFACE_API_KEY in .env")

def get_embeddings():
    """Return HuggingFace embeddings object using Hub API."""
    return HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_API_KEY
    )

def index_documents(chunks):
    """Index document chunks into FAISS."""
    embeddings = get_embeddings()

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    vectorstore.save_local(FAISS_DIR)

def load_faiss_index():
    """Load FAISS index if it exists."""
    if not os.path.exists(FAISS_DIR):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(FAISS_DIR, embeddings)
