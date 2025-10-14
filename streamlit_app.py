import streamlit as st
import time
import random
from model import retrieve_book_info, print_book_results, collection, embedding_model, extracted_book_data

st.title("Book Recommendation RAG Model Tester Site")

# Simple chat box
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.chat_message("bot").write("Hello! I'm here to help you find books that match your mood. How are you feeling today?")

def response_generator():
    response = random.choice([
        "Sure! Here are some book recommendations based on your mood.",
        "Absolutely! Let me find some books that match your feelings.",
        "Of course! I'll look for books that resonate with your current mood."
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

user_input = st.chat_input("Heartwarming, sweet, sad, senual... ")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    relevant_documents, relevant_titles, relevant_ids = retrieve_book_info(user_input, collection, embedding_model)
    response = f"{user_input}"
    # Print the chat message from the user
    with st.chat_message("user"):
        st.write(user_input)
        # Append the user message to the chat history
        st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("bot"):
        filler_response = st.write_stream(response_generator())
        st.session_state.chat_history.append(("bot", filler_response))

#for sender, message in st.session_state.chat_history:
 #   st.chat_message(sender).write(message)
