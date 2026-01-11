import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Models
# Recommended: gemini-1.5-flash for speed or gemini-1.5-pro for complex reasoning
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Paths
DATA_DIR = "data/documents"
FAISS_DIR = os.getenv("FAISS_DIR", "data/faiss_index")

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))

