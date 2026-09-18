"""
app.py - Streamlit Frontend Application for LangGraph AI Chatbot

This module handles the Streamlit user interface, including:
1. Page configuration and layout setup.
2. Session state management for multi-thread chat history and uploaded PDF documents.
3. Sidebar navigation (PDF document uploader & active conversation threads list).
4. Main Chat Interface with support for:
   - Dynamic message rendering (User, Assistant, Tool Execution status cards).
   - Human-in-the-Loop (HITL) approval/rejection interactive widgets.
   - Real-time streaming assistant response generation via LangGraph backend.
"""

import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

# Import backend services and functions
from backend import (
    GLOBAL_VECTORSTORE,
    chatbot,
    extract_text_content,
    load_all_threads_from_db,
    process_pdf_file,
)

# ==========================================
# 1. Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="LangGraph AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. Session State Initialization
# ==========================================
if "thread_list" not in st.session_state:
    saved_threads = load_all_threads_from_db()
    if saved_threads:
        st.session_state.thread_list = saved_threads
        st.session_state.current_thread_id = saved_threads[-1]["id"]
    else:
        initial_id = str(uuid.uuid4())[:8]
        st.session_state.thread_list = [{"id": initial_id, "title": "New Chat"}]
        st.session_state.current_thread_id = initial_id

# Fallback: ensure current_thread_id exists in thread_list
if not any(t["id"] == st.session_state.current_thread_id for t in st.session_state.thread_list):
    st.session_state.current_thread_id = st.session_state.thread_list[0]["id"]


# ==========================================
# 3. Sidebar UI - PDF RAG & Conversation Threads
# ==========================================
with st.sidebar:
    st.title("📄 PDF RAG Assistant")
    uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader")

    # Handle PDF Upload & Indexing
    if uploaded_pdf is not None:
        if st.session_state.get("uploaded_pdf_name") != uploaded_pdf.name:
            with st.status("⚙️ Processing PDF & building FAISS index...", state="running") as pdf_status:
                try:
                    vs, chunk_count = process_pdf_file(uploaded_pdf)
                    st.session_state["faiss_vectorstore"] = vs
                    st.session_state["uploaded_pdf_name"] = uploaded_pdf.name
                    GLOBAL_VECTORSTORE["vs"] = vs
                    GLOBAL_VECTORSTORE["name"] = uploaded_pdf.name
                    pdf_status.update(
                        label=f"✅ Indexed '{uploaded_pdf.name}' ({chunk_count} chunks)",
                        state="complete",
                    )
                except Exception as e:
                    pdf_status.update(label=f"❌ PDF Processing Error: {e}", state="error")
        else:
            st.caption(f"📌 Active Document: `{uploaded_pdf.name}`")
    elif "uploaded_pdf_name" in st.session_state and st.session_state["uploaded_pdf_name"]:
        st.caption(f"📌 Active Document: `{st.session_state['uploaded_pdf_name']}`")
    elif GLOBAL_VECTORSTORE.get("name"):
        st.caption(f"📌 Active Document: `{GLOBAL_VECTORSTORE['name']}`")

    st.divider()

    # Conversation Threads Navigation
    st.title("💬 Conversations")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.thread_list.append({"id": new_id, "title": "New Chat"})
        st.session_state.current_thread_id = new_id
        st.rerun()

    st.divider()

    # Render List of Chat Thread Selection Buttons
    for thread in reversed(st.session_state.thread_list):
        is_active = thread["id"] == st.session_state.current_thread_id
        btn_label = f"📍 {thread['title']}" if is_active else f"💬 {thread['title']}"
        btn_type = "primary" if is_active else "secondary"

        if st.button(btn_label, key=f"btn_{thread['id']}", use_container_width=True, type=btn_type):
            st.session_state.current_thread_id = thread["id"]
            st.rerun()

# Retrieve current active thread object
current_thread_obj = next(t for t in st.session_state.thread_list if t["id"] == st.session_state.current_thread_id)


# ==========================================
# 4. Main Chat Interface Header
# ==========================================
st.title(f"🤖 {current_thread_obj['title']}")
st.caption("Powered by LangGraph, LangChain Google GenAI, & Streamlit")


# ==========================================
# 5. Render Message History & Human-in-the-Loop Widget
# ==========================================
config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
current_state = chatbot.get_state(config)
messages = (
    (current_state.values.get("messages") or current_state.values.get("message", []))
    if current_state and current_state.values
    else []
)

for msg in messages:
    text_content = extract_text_content(msg.content)
    if isinstance(msg, HumanMessage):
        if text_content:
            with st.chat_message("user"):
                st.markdown(text_content)

    elif isinstance(msg, AIMessage):
        # 1. Render tool calls requested by the assistant
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                t_name = tool_call.get("name", "tool")
                t_args = tool_call.get("args", {})
                icon = (
                    "🧮"
                    if "calc" in t_name.lower()
                    else "🔍"
                    if "tavily" in t_name.lower() or "search" in t_name.lower()
                    else "📄"
                    if "pdf" in t_name.lower()
                    else "🛠️"
                )
                with st.status(f"{icon} **Tool Called:** `{t_name}`", state="complete", expanded=False):
                    st.write("**Arguments:**")
                    st.json(t_args)

        # 2. Render AI assistant text response
        if text_content:
            with st.chat_message("assistant"):
                st.markdown(text_content)

    elif isinstance(msg, ToolMessage):
        t_name = getattr(msg, "name", None) or "Tool Output"
        icon = (
            "🧮"
            if "calc" in t_name.lower()
            else "🔍"
            if "tavily" in t_name.lower() or "search" in t_name.lower()
            else "📄"
            if "pdf" in t_name.lower()
            else "📦"
        )
        with st.status(f"{icon} **Tool Result (`{t_name}`)**", state="complete", expanded=False):
            st.code(
                text_content,
                language="json" if text_content.strip().startswith(("{", "[")) else "text",
            )


# 5.5 Render Human-in-the-Loop (HITL) Approval Widget if execution is paused by interrupt
has_active_interrupt = False
if current_state and current_state.tasks:
    for task in current_state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            for interrupt_item in task.interrupts:
                has_active_interrupt = True
                prompt_text = interrupt_item.value
                with st.chat_message("assistant"):
                    st.warning(f"⚠️ **Human Approval Required:** {prompt_text}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "✅ Approve (Yes)",
                            key=f"approve_{st.session_state.current_thread_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            with st.spinner("🔄 Resuming execution with approval..."):
                                for _ in chatbot.stream(Command(resume="Yes"), config=config, stream_mode="updates"):
                                    pass
                            st.rerun()
                    with col2:
                        if st.button(
                            "❌ Reject (No)",
                            key=f"reject_{st.session_state.current_thread_id}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            # Extract tool_call_id to cleanly close tool execution
                            last_msg = (
                                current_state.values["messages"][-1]
                                if (current_state and current_state.values)
                                else None
                            )
                            tool_call_id = "call_rejected"
                            if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                tool_call_id = last_msg.tool_calls[0].get("id", "call_rejected")

                            # Directly update graph state with cancellation messages
                            chatbot.update_state(
                                config,
                                {
                                    "messages": [
                                        ToolMessage(
                                            content="Calculation tool execution denied by user.",
                                            tool_call_id=tool_call_id,
                                            name="calculator",
                                        ),
                                        AIMessage(
                                            content="❌ Calculation request was rejected by the user. Operation cancelled."
                                        ),
                                    ]
                                },
                            )
                            st.rerun()


# ==========================================
# 6. User Chat Input & Real-Time Streaming Handler
# ==========================================
user_input = st.chat_input("Type your message here...")

if user_input:
    # Auto-set thread title on first message if still "New Chat"
    if current_thread_obj["title"] == "New Chat":
        clean_title = user_input.strip()
        current_thread_obj["title"] = clean_title[:28] + ("..." if len(clean_title) > 28 else "")

    # Display user input immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response & tool call status updates
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for event in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            if isinstance(event, tuple):
                items = [event]
            elif isinstance(event, dict):
                items = list(event.items())
            else:
                items = []

            for node_name, node_output in items:
                if node_name == "__interrupt__" or not isinstance(node_output, dict):
                    # Execution interrupted by tool (e.g. calculator approval prompt)
                    st.rerun()
                    break

                node_messages = node_output.get("messages", [])
                for msg in node_messages:
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                t_name = tool_call.get("name", "tool")
                                t_args = tool_call.get("args", {})
                                icon = (
                                    "🧮"
                                    if "calc" in t_name.lower()
                                    else "🔍"
                                    if "tavily" in t_name.lower() or "search" in t_name.lower()
                                    else "📄"
                                    if "pdf" in t_name.lower()
                                    else "🛠️"
                                )
                                with st.status(f"{icon} **Tool Called:** `{t_name}`", state="complete", expanded=False):
                                    st.write("**Arguments:**")
                                    st.json(t_args)

                        text = extract_text_content(msg.content)
                        if text:
                            full_response += text
                            message_placeholder.markdown(full_response)

                    elif isinstance(msg, ToolMessage):
                        t_name = getattr(msg, "name", None) or "Tool Output"
                        t_content = extract_text_content(msg.content)
                        icon = (
                            "🧮"
                            if "calc" in t_name.lower()
                            else "🔍"
                            if "tavily" in t_name.lower() or "search" in t_name.lower()
                            else "📄"
                            if "pdf" in t_name.lower()
                            else "📦"
                        )
                        with st.status(f"{icon} **Tool Result (`{t_name}`)**", state="complete", expanded=False):
                            st.code(
                                t_content,
                                language="json" if t_content.strip().startswith(("{", "[")) else "text",
                            )

        st.rerun()
