import streamlit as st
from db import (
    create_new_chat,
    save_message,
    get_all_chats,
    load_chat,
    delete_chat,
    clear_all_chats,
)
from models import generate

def generate_response(prompt):
    return generate(prompt)

st.set_page_config(page_title="Story Spark", layout="centered")

# --- Manage Chat ID via Query Parameters ---
query_params = st.experimental_get_query_params()

if "chat_id" not in st.session_state:
    if "chat_id" in query_params and query_params["chat_id"]:
        st.session_state.chat_id = query_params["chat_id"][0]
    else:
        st.session_state.chat_id = create_new_chat()
        st.experimental_set_query_params(chat_id=st.session_state.chat_id)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat(st.session_state.chat_id)

# === Sidebar ===
st.sidebar.title("💬 Chats")

# New Chat button
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = create_new_chat()
    st.session_state.messages = []
    st.experimental_set_query_params(chat_id=st.session_state.chat_id)

# Chat list with options
chats = get_all_chats()
for chat in chats:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    if col1.button(chat["title"], key=f"load_{chat['chat_id']}"):
        st.session_state.chat_id = chat["chat_id"]
        st.session_state.messages = load_chat(chat["chat_id"])
        st.experimental_set_query_params(chat_id=chat["chat_id"])
    if col2.button("🗑", key=f"delete_{chat['chat_id']}"):
        delete_chat(chat["chat_id"])
        if st.session_state.chat_id == chat["chat_id"]:
            st.session_state.chat_id = create_new_chat()
            st.session_state.messages = []
            st.experimental_set_query_params(chat_id=st.session_state.chat_id)

# Clear All Chats button
if st.sidebar.button("🧹 Clear All Chats"):
    clear_all_chats()
    st.session_state.chat_id = create_new_chat()
    st.session_state.messages = []
    st.experimental_set_query_params(chat_id=st.session_state.chat_id)

# === Chat UI ===
st.title("🧠 Humanized Story Generator")

# Display chat history using markdown
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['content']}")

# Chat input section using text_input instead of chat_input
prompt = st.text_input("Type your prompt here...")
if prompt:
    # Display the user input
    st.markdown(f"**User:** {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.chat_id, "user", prompt)

    # Generate and display assistant response
    response = generate_response(prompt)
    st.markdown(f"**Assistant:** {response}")
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.chat_id, "assistant", response)
