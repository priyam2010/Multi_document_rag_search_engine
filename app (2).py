import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports (Matches your uploaded logic)
#

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import Tool
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agentic AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .chat-bubble { padding: 10px; border-radius: 15px; margin-bottom: 10px; }
    .user-bubble { background-color: #1e3a8a; align-self: flex-end; }
    .bot-bubble { background-color: #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Agent Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password", value=os.getenv("groq_api_key", ""))
    st.info(
            "***Available Tools ***"
            "📚 Wikipedia Search,"
            "📄 Arxiv Research Papers,"
            "🌐 Web Content Retrieval,"
            "📋 PDF Document Search,"
            "📝 Personal Information Retrieval")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- INITIALIZE AGENT (Caching to avoid re-loading FAISS every click) ---
@st.cache_resource
def get_agent_executor(groq_key):
    # 1. Setup Tools
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(wiki_client=None,top_k_results=1, doc_content_chars_max=250))
    
    # 2 arxiv retriever tool
    api_wrapper_arxiv=ArxivAPIWrapper(arxiv_search=None,
    arxiv_exceptions=None,top_k_results=1,doc_content_chars_max=250)
    arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

    # 3 url retriever tool
    loader = WebBaseLoader('https://www.msn.com/en-in/news/world/sorry-everyone-iit-kanpur-student-found-dead-in-hostel-note-recovered-authorities-launch-probe/ar-AA1TfDGT?uxmode=ruby&ocid=edgntpruby&pc=W099&cvid=69540543931741e0b69f668967302972&ei=14')
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, HuggingFaceEmbeddings())
    retriever = vectordb.as_retriever()
    url_retriever_tool = create_retriever_tool(retriever, "URL_search_tool", "Search information from the given URL about IIT Kanpur student found dead in hostel")

    # 4 RAG Tool from PDF Upload
    loader = PyPDFLoader("C:\\Agentic_AI_Projects\\Multi_Source_Rag_App\\Agent Quality Whitepaper.pdf")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb1 = FAISS.from_documents(documents, HuggingFaceEmbeddings())
    retriever = vectordb1.as_retriever()
    pdf_retriever_tool = create_retriever_tool(retriever, "pdf_retriever_tool", "Search information about agent quality whitepaper which discusses the profound challenge of quality assurance for autonomous, goal-oriented AI agents.")

    # 5.Text retriever tool 
    loader1 = TextLoader(r"C:\Agentic_AI_Workshop\1-Langchain\about_me.txt")
    docs1 = loader1.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs1)
    embedding_mode2 = HuggingFaceEmbeddings()
    vectordb2 = FAISS.from_documents(documents, embedding_mode2)
    text_retriever_tool = vectordb2.as_retriever()
    text_retriever_tool = create_retriever_tool(text_retriever_tool, "text_retriever_tool", "Search information about Abhiram kumar soni")

    tools = [wiki, arxiv, url_retriever_tool, pdf_retriever_tool, text_retriever_tool]

    # 2. Setup LLM & Agent
    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools,verbose=True,max_iterations=2)

# --- MAIN CHAT INTERFACE ---c
st.title("🤖 Multi-Tool Agentic AI")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate Response
    if not api_key:
        st.error("Please provide an API key in the sidebar.")
    else:
        with st.chat_message("assistant"):
            try:
                executor = get_agent_executor(api_key)
                response = executor.invoke({"input": prompt_input})
                output = response["output"]
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                st.error(f"Error: {str(e)}")

