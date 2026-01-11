from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceEmbeddings
import os
from config import FAISS_DIR

# ✅ Streamlit Free SAFE embeddings (NO torch, NO sentence-transformers)
def get_embeddings():
    return HuggingFaceInferenceEmbeddings(
        api_key=os.environ["HUGGINGFACE_API_KEY"],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def index_documents(chunks):
    embeddings = get_embeddings()

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    if not texts:
        raise ValueError("No text chunks found to index")

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local(FAISS_DIR)

def load_faiss_index():
    if not os.path.exists(FAISS_DIR):
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )






