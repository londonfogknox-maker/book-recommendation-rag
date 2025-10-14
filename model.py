from pypdf import PdfReader
import os
import streamlit as st

def extract_text_and_metadata_from_pdf(pdf_path):
  """Extracts text and specific metadata (title, author, category, mood) from a PDF file."""
  book_data = [] # List to store dictionaries of book information

  try:
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
      if i > 0: # Skip the first page
        page_text = page.extract_text()
        if page_text:
          lines = page_text.splitlines()
          if len(lines) >= 5: # Ensure there are enough lines for all fields
            title = lines[0].strip()
            author = lines[1].strip()
            category = lines[2].strip()
            mood_tags = lines[3].strip()
            description = "\n".join(lines[4:]).strip() # The rest is the description

            book_data.append({
                'title': title,
                'author': author,
                'category': category,
                'mood_tags': mood_tags,
                'description': description
            })
          elif lines: # Handle pages with less than 5 lines, just store the available text
              book_data.append({
                  'title': lines[0].strip() if len(lines) > 0 else "No Title Found",
                  'author': lines[1].strip() if len(lines) > 1 else "No Author Found",
                  'category': lines[2].strip() if len(lines) > 2 else "No Category Found",
                  'mood_tags': lines[3].strip() if len(lines) > 3 else "No Mood Tags Found",
                  'description': "\n".join(lines[4:]).strip() if len(lines) > 4 else "No Description Found"
              })


    #print(f"Extracted data for {len(book_data)} books.")
  except Exception as e:
    print(f"Error reading {pdf_path}: {e}")
    return [] # Return empty list in case of error

  return book_data

# Updated to local workspace path
pdf_path = "Archival Favorites Cleaned (1).pdf"  

if os.path.exists(pdf_path):
  extracted_book_data = extract_text_and_metadata_from_pdf(pdf_path)
  #print(f"Extracted text and metadata from {pdf_path}")
else:
  print(f"File not found: {pdf_path}")
  print("Please upload your PDF file to this location.")

def split_text_into_chunks(text, chunk_size=500, overlap_size=50):
  #"""Splits text into smaller chunks with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

chunked_book_data = []
for book in extracted_book_data:
    description_chunks = split_text_into_chunks(book['description'])
    for chunk in description_chunks:
        chunked_book_data.append({
            'title': book['title'],
            'author': book['author'],
            'category': book['category'],
            'mood_tags': book['mood_tags'],
            'text': chunk # This is the chunked description
        })

from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#print("Embedding model loaded successfully.")

# Extract the 'text' value from each dictionary
text_chunks = [item['text'] for item in chunked_book_data]

# Generate embeddings for each text chunk
chunked_text_embeddings = embedding_model.encode(text_chunks)

# Necessary to use pysqlite3 instead of sqlite3
# to avoid potential compatibility issues with ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Initialize a Chroma client
# Using an in-memory client for this example
client = chromadb.Client()

# Get or create a collection
collection = client.get_or_create_collection(name="book_chunks_with_titles")

# Prepare data for Chroma
# Chroma expects embeddings as a list of lists, documents as a list of strings, and metadata as a list of dictionaries
embeddings_list = chunked_text_embeddings.tolist()
documents_list = [item['text'] for item in chunked_book_data]
metadata_list = [{'title': item['title']} for item in chunked_book_data]
ids_list = [f"chunk_{i}" for i in range(len(chunked_book_data))]


# Add the text chunks, their generated embeddings, and their corresponding book titles to the collection
collection.add(
    embeddings=embeddings_list,
    documents=documents_list,
    metadatas=metadata_list,
    ids=ids_list
)

def retrieve_book_info(user_query, collection, embedding_model, n_results=5):
  """Retrieves the most relevant text chunks, titles, and IDs from Chroma based on a query."""
  # Generate embedding for the query
  query_embedding = embedding_model.encode(user_query).tolist()

  # Query Chroma for similar chunks
  results = collection.query(
      query_embeddings=[query_embedding],
      n_results=n_results,
      include=['documents', 'metadatas'] # Include documents and metadata in the results
  )

  # Extract the relevant documents (text chunks), titles, and IDs
  relevant_documents = results['documents'][0]
  relevant_titles = [metadata['title'] for metadata in results['metadatas'][0]]
  relevant_ids = results['ids'][0]

  return relevant_documents, relevant_titles, relevant_ids

def print_book_results(relevant_documents, relevant_titles, relevant_ids, extracted_book_data, n_display=5):
    """
    Returns a string containing book information for the most relevant results.

    Args:
      relevant_documents: List of relevant text chunks.
      relevant_titles: List of titles corresponding to the chunks.
      relevant_ids: List of IDs corresponding to the chunks.
      extracted_book_data: The original list of dictionaries with full book data.
      n_display: The maximum number of unique books to display.

    Returns:
      A string containing formatted book recommendations.
    """
    recommendations_string = " \n\n"
    displayed_titles = set()
    count = 0
    for i in range(len(relevant_documents)):
      # Use the retrieved title to find the full book data from the original extracted data
      book_info = next((item for item in extracted_book_data if item['title'] == relevant_titles[i]), None)

      if book_info and book_info.get('title') not in displayed_titles:
          recommendations_string += f"\nBook {count+1}:\n"
          recommendations_string += f"  **Title**: {book_info.get('title', 'N/A')}  \n"
          recommendations_string += f"  Author: {book_info.get('author', 'N/A')}  \n"
          recommendations_string += f"  Category: {book_info.get('category', 'N/A')}  \n"
          recommendations_string += f"  Mood Tags: *{book_info.get('mood_tags', 'N/A')}*  \n"
          displayed_titles.add(book_info.get('title'))
          count += 1
          if count >= n_display: # Stop after displaying n_display unique books
              break
      elif not book_info:
          # Fallback if full book info is not found (shouldn't happen if titles match correctly)
          recommendations_string += f"\nRecommendation {count+1}:\n"
          recommendations_string += f"  Title: {relevant_titles[i]}\n"
          recommendations_string += "  Full book details not found.\n"
          count += 1
          if count >= n_display: # Stop after displaying n_display unique books
              break

    return recommendations_string

  # Example of using the function
  #user_query = "Books similar to the Iliad"
  #relevant_documents, relevant_titles, relevant_ids = retrieve_book_info(user_query, collection, embedding_model)

  # Call the new function to get the results string
  #recommendations_output = print_book_results(relevant_documents, relevant_titles, relevant_ids, extracted_book_data)

  # Print the returned string
  #print(recommendations_output)