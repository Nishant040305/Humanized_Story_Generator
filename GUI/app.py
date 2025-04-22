import streamlit as st
import os
import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast

from db import (
    create_new_chat,
    save_message,
    get_all_chats,
    load_chat,
    delete_chat,
    clear_all_chats,
)
from load_lstm import main as lstm_generate
from load_transformer import tranformer_model


def generate_response(prompt, model_type):
    if model_type == "LSTM":
        return lstm_generate(prompt)
    else:  # Transformer
        return tranformer_model(prompt)


st.set_page_config(page_title="Story Spark", layout="centered")

# Initialize model type in session state if not present
if "model_type" not in st.session_state:
    st.session_state.model_type = "Transformer"

if "chat_id" not in st.session_state:
    if "chat_id" in st.query_params and st.query_params["chat_id"]:
        st.session_state.chat_id = st.query_params["chat_id"]
    else:
        st.session_state.chat_id = create_new_chat()
        st.query_params["chat_id"] = st.session_state.chat_id

if "messages" not in st.session_state:
    st.session_state.messages = load_chat(st.session_state.chat_id)

# === Sidebar ===
st.sidebar.title("💬 Chats")

# Model selection
st.sidebar.subheader("Model Selection")
model_type = st.sidebar.radio(
    "Choose a model:",
    options=["Transformer", "LSTM"],
    index=0 if st.session_state.model_type == "Transformer" else 1,
)
st.session_state.model_type = model_type

# New Chat button
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = create_new_chat()
    st.session_state.messages = []
    st.query_params["chat_id"] = st.session_state.chat_id

# List chats with load and delete options
chats = get_all_chats()
for chat in chats:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    if col1.button(chat["title"], key=f"load_{chat['chat_id']}"):
        st.session_state.chat_id = chat["chat_id"]
        st.session_state.messages = load_chat(chat["chat_id"])
        st.query_params["chat_id"] = chat["chat_id"]
    if col2.button("🗑", key=f"delete_{chat['chat_id']}"):
        delete_chat(chat["chat_id"])
        if st.session_state.chat_id == chat["chat_id"]:
            st.session_state.chat_id = create_new_chat()
            st.session_state.messages = []
            st.query_params["chat_id"] = st.session_state.chat_id

# Clear All Chats button
if st.sidebar.button("🧹 Clear All Chats"):
    clear_all_chats()
    st.session_state.chat_id = create_new_chat()
    st.session_state.messages = []
    st.query_params["chat_id"] = st.session_state.chat_id

# === Main Content ===
st.title("🧠 Humanized Story Generator")

# Display current model
st.caption(f"Currently using: {st.session_state.model_type} model")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input section
prompt = st.chat_input("Type your prompt...")

if prompt:
    # Process user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.chat_id, "user", prompt)

    # Show loading indicator
    with st.spinner(
        f"Generating response using {st.session_state.model_type} model..."
    ):
        # Generate response based on selected model
        response = generate_response(prompt, st.session_state.model_type)

    # Display response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.chat_id, "assistant", response)
