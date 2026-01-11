# app.py
import streamlit as st
from dotenv import load_dotenv
from loaders import load_uploaded_documents
from text_utils import chunk_documents
from vectorstore import index_documents, load_faiss_index
from rag_pipeline import generate_answer

load_dotenv()

st.set_page_config(page_title="Hybrid RAG Search Engine", layout="wide")

# ---------------- Session State ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- Sidebar ----------------
st.sidebar.title("📄 Document Manager")

groq_api_key = st.sidebar.text_input(
    "Enter your GROQ API Key",
    type="password"
)

use_web = st.sidebar.checkbox("Enable Tavily Web Search", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------- Index Documents ----------------
if st.sidebar.button("Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload documents first.")
    else:
        try:
            docs = load_uploaded_documents(uploaded_files)
            chunks = chunk_documents(docs)

            index_documents(chunks)
            st.session_state.vectorstore = load_faiss_index()
            st.sidebar.success("✅ Documents Indexed Successfully")
        except Exception as e:
            st.sidebar.error("❌ Indexing failed")
            st.sidebar.exception(e)

# ---------------- Main UI ----------------
st.title("🔍 Hybrid RAG Search Engine")
query = st.text_input("Ask a question:")

if st.button("Search") and query:
    if not groq_api_key:
        st.warning("Please enter your GROQ API key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.warning("Please index documents first.")
    else:
        try:
            answer, sources, route = generate_answer(
                query=query,
                vectorstore=st.session_state.vectorstore,
                groq_api_key=groq_api_key,
                use_web=use_web
            )

            icon = "📄" if route == "document" else "🌐" if route == "web" else "🔀"

            tabs = st.tabs(["Answer", "Sources"])
            with tabs[0]:
                st.markdown(f"### {icon} Answer")
                st.write(answer)

            with tabs[1]:
                st.markdown("### 📚 Evidence")
                for s in sources:
                    st.write(s)

        except Exception as e:
            st.error("❌ Failed to generate answer")
            st.exception(e)
