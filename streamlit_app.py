import streamlit as st
import time

# Import model.py file to use RAG function
#from model import retrieve_book_info

st.title("Book Recommendation RAG Model Tester Site")
st.write(
    "This model is currently pulling from these pdfs:\n" \
    "- Archival Favorites"
)

# Simple chat box
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    time.sleep(0.15)
    # Simple echo response
    response = f"You said: {user_input}"
    st.session_state.chat_history.append(("bot", response))

for sender, message in st.session_state.chat_history:
    st.chat_message(sender).write(message)
