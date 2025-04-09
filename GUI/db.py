from pymongo import MongoClient
from datetime import datetime
import uuid

client = MongoClient("mongodb://localhost:27017/")
db = client["chat_app"]
collection = db["chats"]

def create_new_chat():
    chat_id = str(uuid.uuid4())
    collection.insert_one({
        "chat_id": chat_id,
        "created_at": datetime.now(),
        "messages": []
    })
    return chat_id

def save_message(chat_id, role, content):
    collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": {"role": role, "content": content, "timestamp": datetime.now()}}}
    )

def get_all_chats():
    chats = collection.find({}, {"chat_id": 1, "messages": 1, "created_at": 1})
    summaries = []
    for c in chats:
        title = c["messages"][0]["content"][:40] if c.get("messages") else "New Chat"
        summaries.append({
            "chat_id": c["chat_id"],
            "title": title,
            "created_at": c["created_at"]
        })
    return summaries

def load_chat(chat_id):
    chat = collection.find_one({"chat_id": chat_id})
    return chat["messages"] if chat else []

def delete_chat(chat_id):
    collection.delete_one({"chat_id": chat_id})

def clear_all_chats():
    collection.delete_many({})
