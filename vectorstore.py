from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from config import FAISS_DIR
import streamlit as st


def get_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HUGGINGFACE_API_KEY"],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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






