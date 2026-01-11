import os

# ---------------- API KEYS ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

# ---------------- MODELS ----------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- CHUNKING ----------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "documents")
FAISS_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# ---------------- RETRIEVAL ----------------
TOP_K = int(os.getenv("TOP_K", 4))
