"""
backend.py - Core Backend Service & LangGraph AI Agent Architecture

This module provides backend services for the AI Chatbot application:
1. Environment Configuration & Global Caches
2. LangGraph Custom Tools (Tavily Web Search, HITL Calculator, PDF RAG Search)
3. PDF Processing & FAISS Vector Indexing Pipeline
4. LangGraph StateGraph Architecture & SQLite Persistence Checkpointer
5. Helper functions for message parsing and conversation thread history extraction
"""

import math
import os
import sqlite3
import tempfile
import uuid
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

# ==========================================
# 1. Environment Initialization
# ==========================================
# Load environment variables (.env) searching recursively
load_dotenv(find_dotenv())

# Global in-memory cache for the vector store (accessible across threads)
GLOBAL_VECTORSTORE = {"vs": None, "name": None}


# ==========================================
# 2. Custom Tools Definitions
# ==========================================

# 2.1 Tavily Search Tool (Web Search Integration)
tavily_tool = TavilySearchResults(max_results=3)


# 2.2 Calculator Tool (with Human-in-the-Loop Interrupt)
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression safely after obtaining user approval.

    Args:
        expression: A string containing a mathematical expression (e.g., "2+2", "math.sqrt(16)").

    Returns:
        The result of the calculation as a string, or rejection/error message.
    """
    # Trigger Human-in-the-Loop (HITL) interrupt asking for explicit approval
    decision = interrupt(
        f"We need your approval before proceeding with calculation: `{expression}`. (Yes / No)"
    )

    # Check user response
    if str(decision).strip().lower() not in ["yes", "y", "approve"]:
        return (
            "USER REJECTED THIS TOOL CALL: The user denied permission to run the calculator. "
            "Do NOT perform or output the calculation. Strictly tell the user that the operation was rejected by them."
        )

    try:
        # Restrict built-in evaluation environment for basic safety
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "sum": sum,
            "min": min,
            "max": max,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}"


# 2.3 PDF Processing Pipeline & Vector Store Search Tool
def process_pdf_file(uploaded_file):
    """Processes an uploaded PDF file, chunks it, generates embeddings, and constructs a FAISS index.

    Args:
        uploaded_file: Streamlit UploadedFile object containing raw PDF bytes.

    Returns:
        Tuple (FAISS vectorstore object, int count of document chunks created).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load PDF pages
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Chunk text into overlapping segments
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Generate Gemini embeddings and save to FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Cache vectorstore in-memory & save to disk
        GLOBAL_VECTORSTORE["vs"] = vectorstore
        GLOBAL_VECTORSTORE["name"] = uploaded_file.name
        try:
            vectorstore.save_local("faiss_index")
        except Exception:
            pass

        return vectorstore, len(chunks)
    finally:
        # Clean up temporary disk file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@tool
def pdf_search(query: str) -> str:
    """Searches the uploaded PDF document for relevant snippets to answer user questions.

    Args:
        query: Search query string describing the information needed from the PDF.

    Returns:
        Formatted text snippets from the document along with page numbers.
    """
    # 1. Check Global Cache first (accessible across worker threads)
    vectorstore = GLOBAL_VECTORSTORE.get("vs")

    # 2. Check Disk Index as fallback if cache empty
    if vectorstore is None and os.path.exists("faiss_index"):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
            vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            GLOBAL_VECTORSTORE["vs"] = vectorstore
        except Exception:
            pass

    if vectorstore is None:
        return "No PDF document has been uploaded yet by the user. Please instruct the user to upload a PDF document in the sidebar."

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    results = retriever.invoke(query)

    if not results:
        return "No relevant information found in the uploaded PDF document."

    formatted_results = []
    for i, doc in enumerate(results, 1):
        page_num = doc.metadata.get("page", 0) + 1
        formatted_results.append(f"--- Snippet {i} (Page {page_num}) ---\n{doc.page_content}")

    return "\n\n".join(formatted_results)


# ==========================================
# 3. LangGraph Workflow & Agent Compiler
# ==========================================

def custom_tools_condition(state):
    """Custom tools condition handler ensuring messages key compatibility."""
    if isinstance(state, dict):
        if "messages" not in state and "message" in state:
            state["messages"] = state["message"]
    return tools_condition(state)


class ChatState(TypedDict, total=False):
    """State schema for LangGraph agent workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    message: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def get_chatbot():
    """Compiles and caches the LangGraph chatbot StateGraph with persistent SQLite memory checkpointer."""
    # List of active tools available to the AI agent
    tools = [tavily_tool, calculator, pdf_search]

    # Initialize Google Gemini LLM with tool binding
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    llm_with_tools = llm.bind_tools(tools)

    # Initialize Graph builder
    graph = StateGraph(ChatState)

    def chat_node(state: ChatState):
        """Main LLM decision node."""
        msgs = state.get("messages") or state.get("message", [])
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "message": [response]}

    # Define Graph structure
    graph.add_node("chatbot", chat_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", custom_tools_condition)
    graph.add_edge("tools", "chatbot")

    # Persistent SQLite connection checkpointer for conversation memory
    conn = sqlite3.connect("chatbot_memory.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


# Instantiate/Fetch cached chatbot graph object
chatbot = get_chatbot()


# ==========================================
# 4. Helper & Database Persistence Utilities
# ==========================================

def extract_text_content(content) -> str:
    """Safely parses string, list, or dict message content into plain text.

    Args:
        content: Raw message content (str, list, or dict).

    Returns:
        Extracted text string.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)


def load_all_threads_from_db():
    """Queries SQLite checkpoint database to load all active chat thread IDs and titles.

    Returns:
        List of dicts: [{"id": thread_id, "title": preview_title}]
    """
    try:
        conn = sqlite3.connect("chatbot_memory.sqlite", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cursor.fetchall()
        conn.close()
    except Exception:
        rows = []

    thread_list = []
    for row in rows:
        tid = row[0]
        state = chatbot.get_state({"configurable": {"thread_id": tid}})
        msgs = (state.values.get("messages") or state.values.get("message", [])) if state and state.values else []

        title = "New Chat"
        for m in msgs:
            if isinstance(m, HumanMessage) and m.content:
                text = extract_text_content(m.content).strip()
                if text:
                    title = text[:28] + ("..." if len(text) > 28 else "")
                    break
        thread_list.append({"id": tid, "title": title})

    return thread_list
