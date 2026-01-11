# vectorstore.py
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings  # ✅ Use this instead
from config import FAISS_DIR
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get HuggingFace API key
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HF_API_KEY:
    raise ValueError("Please set HUGGINGFACE_API_KEY in your .env file")

def get_embeddings():
    """
    Returns HuggingFace Hub Embeddings configured for CPU.
    """
    return HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # 🔑 ensures no GPU usage
        huggingfacehub_api_token=HF_API_KEY
    )

def index_documents(chunks):
    """
    Indexes document chunks into FAISS vectorstore.
    """
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
    """
    Loads FAISS vectorstore if it exists, else returns None.
    """
    if not os.path.exists(FAISS_DIR):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(FAISS_DIR, embeddings)

