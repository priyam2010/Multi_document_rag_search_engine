from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import os
from config import FAISS_DIR


def get_embeddings():
    return HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_KEY")  # ✅ MUST BE SET
    )


def index_documents(chunks):
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
    if not os.path.exists(FAISS_DIR):
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
