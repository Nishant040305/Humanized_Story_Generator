import streamlit as st
from db import (
    create_new_chat,
    save_message,
    get_all_chats,
    load_chat,
    delete_chat,
    clear_all_chats,
)


def generate_response(prompt):
    return "Response: " + prompt


st.set_page_config(page_title="Story Spark", layout="centered")

# --- Manage Chat ID via Query Parameters ---
# Retrieve query parameters using the non-experimental API
if "chat_id" not in st.session_state:
    # Use chat_id from query params if available; otherwise, create a new chat.
    if "chat_id" in st.query_params and st.query_params["chat_id"]:
        st.session_state.chat_id = st.query_params["chat_id"]
    else:
        st.session_state.chat_id = create_new_chat()
        st.query_params["chat_id"] = st.session_state.chat_id

if "messages" not in st.session_state:
    st.session_state.messages = load_chat(st.session_state.chat_id)

# === Sidebar ===
st.sidebar.title("💬 Chats")

# New Chat button (creates a new chat and updates the URL)
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = create_new_chat()
    st.session_state.messages = []
    st.query_params["chat_id"] = st.session_state.chat_id

# List chats with load and delete options.
chats = get_all_chats()  # Retrieve chat summaries from MongoDB.
for chat in chats:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    # Chat title button to load the chat.
    if col1.button(chat["title"], key=f"load_{chat['chat_id']}"):
        st.session_state.chat_id = chat["chat_id"]
        st.session_state.messages = load_chat(chat["chat_id"])
        st.query_params["chat_id"] = chat["chat_id"]
    # Delete button for that chat.
    if col2.button("🗑", key=f"delete_{chat['chat_id']}"):
        delete_chat(chat["chat_id"])
        # If the currently loaded chat is deleted, start a new chat.
        if st.session_state.chat_id == chat["chat_id"]:
            st.session_state.chat_id = create_new_chat()
            st.session_state.messages = []
            st.query_params["chat_id"] = st.session_state.chat_id

# Clear All Chats button.
if st.sidebar.button("🧹 Clear All Chats"):
    clear_all_chats()
    st.session_state.chat_id = create_new_chat()
    st.session_state.messages = []
    st.query_params["chat_id"] = st.session_state.chat_id

st.title("🧠 Humanized Story Generator")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input section.
prompt = st.chat_input("Type your prompt...")

if prompt:
    # Process user message.
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.chat_id, "user", prompt)

    # Process assistant response.
    response = generate_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.chat_id, "assistant", response)
