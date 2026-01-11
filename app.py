import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from loaders import load_uploaded_documents
from text_utils import chunk_documents
from vectorstore import index_documents, load_faiss_index
from rag_pipeline import generate_answer

st.set_page_config(page_title="Hybrid RAG Search Engine", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("📄 Document Manager")
use_web = st.sidebar.checkbox("Enable Tavily Web Search", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload documents first.")
    else:
        docs = load_uploaded_documents(uploaded_files)
        chunks = chunk_documents(docs)
        index_documents(chunks)
        st.sidebar.success("Documents indexed successfully!")

vectorstore = load_faiss_index()

# ---------------- Main UI ----------------
st.title("🔍 Hybrid RAG Search Engine")
query = st.text_input("Ask a question")

if st.button("Search") and query:
    if not vectorstore:
        st.warning("Please index documents first.")
    else:
        answer, sources, route = generate_answer(query, vectorstore, use_web)
        icon = "📄" if route == "document" else "🌐" if route == "web" else "🔀"

        tabs = st.tabs(["Answer", "Sources"])
        with tabs[0]:
            st.markdown(f"### {icon} Answer")
            st.write(answer)

        with tabs[1]:
            st.markdown("### 📚 Evidence")
            for s in sources:
                st.write(s)
