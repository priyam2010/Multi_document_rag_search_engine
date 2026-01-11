import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Paths  ✅ THIS WAS MISSING
DATA_DIR = "data/documents"
FAISS_DIR = os.getenv("FAISS_DIR", "data/faiss_index")

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))
