# 🚀 GA02: Multi-Document Hybrid RAG Search Engine

## 📌 Project Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) Search Engine** that combines:

- Semantic search over multiple local documents
- Real-time web search using Tavily
- Citation-aware answer generation
- An interactive Streamlit-based chatbot UI

The system mirrors real-world **enterprise AI copilots** that intelligently blend **private knowledge bases** with **live internet data**, while maintaining **source transparency** and **answer grounding**.

---

## 🎯 Objectives

The primary goals of this project are to:

- Build a searchable knowledge base from multiple unstructured documents
- Perform semantic retrieval using FAISS
- Integrate real-time web search via Tavily
- Dynamically route queries between:
  - Document-based search
  - Web-based search
  - Hybrid search
- Generate grounded answers with clear citations
- Provide a clean, user-friendly Streamlit UI

---

## 🧠 System Architecture

User Query
│
Query Classification (Document / Web / Hybrid)
│
├── FAISS Vector Search (Local Docs)
├── Tavily Web Search (Real-Time)
├──Context Assembly
├──LLM (Groq via LangChain)
├──Answer + Citations
├──Streamlit UI


---

## 📂 Project Structure

GA02_Hybrid_RAG/
│
├── app.py # Streamlit UI
├── config.py # Configuration & constants
├── loaders.py # PDF / TXT / Wikipedia loaders
├── models.py # Unified document schemas
├── text_utils.py # Text cleaning & chunking
├── vectorstore.py # FAISS indexing & loading
├── web_search.py # Tavily integration
├── rag_pipeline.py # Hybrid RAG logic
├── requirements.txt
├── .env
│
├── data/
│ ├── documents/ # Uploaded files
│ └── faiss_index/ # Saved FAISS index
│
└── venv/


---

## 📥 Data Sources

### Local Knowledge Base
- PDF documents
- Text files
- Wikipedia pages (LangChain loader)

### Real-Time Knowledge
- Tavily web search results:
  - News
  - Current events
  - Recent research
  - Live statistics

---

## 🧩 Key Features

### ✅ Multi-Document Ingestion
- Supports PDFs, TXT files, and Wikipedia pages
- Unified metadata schema for traceability

### ✅ Semantic Search
- Recursive chunking with overlap
- FAISS-based vector similarity search

### ✅ Hybrid RAG Pipeline
- Intelligent query routing
- Document-only, web-only, or hybrid context assembly

### ✅ Citation-Aware Answers
- Distinguishes between:
  - `[Doc]` document sources
  - `[Web]` Tavily search sources

### ✅ Streamlit UI
- Upload & index documents
- Toggle web search ON/OFF
- Answer & source tabs
- Visual route indicators:
  - 📄 Document-based
  - 🌐 Web-based
  - 🔀 Hybrid

---

## 🔒 Document Grounding Behavior

When **Tavily Web Search is OFF**, the system:

- Answers **only if the information exists in uploaded documents**
- Otherwise responds:

> **“The answer is not available in the provided documents.”**

This ensures **strict document grounding** and prevents hallucinations.

---

## 🧪 Evaluation Scenarios

| Scenario | Expected Behavior |
|--------|-------------------|
| Static knowledge query | Retrieved from documents |
| Real-time factual query | Retrieved via Tavily |
| Hybrid reasoning query | Combined document + web context |

---

## 📊 Quality Assessment

### Strengths
- Modular and scalable architecture
- Clear source attribution
- Real-time + private knowledge fusion
- Production-style UI

### Limitations
- Rule-based query classification
- No re-ranking of retrieved chunks
- No automatic top-N document summarization

### Future Enhancements
- ML-based query classifier
- Chunk re-ranking (Cross-Encoder)
- Document-level summarization
- Conversation memory
- Authentication for enterprise use

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

2️⃣ Install Dependencies
-pip install -r requirements.txt

3️⃣ Configure .env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

4️⃣ Run the App
streamlit run app.py


