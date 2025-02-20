import os
import streamlit as st
from groq import Groq
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit App
st.set_page_config(page_title="PDF Query App", layout="wide")
st.title("📄 PDF Query Application")
st.markdown("Upload your **PDF document**, and ask questions to get insights! 💡")

# Option for the user to provide their own Groq API key
user_api_key = st.text_input("Enter your Groq API key (Leave empty to use default)")

# Use the user's API key or default API key from secrets.toml
GROQ_API_KEY = user_api_key if user_api_key else st.secrets["GROQ_API_KEY"]["key"]

# Initialize the Groq client
if not GROQ_API_KEY:
    st.error("Groq API key is missing!")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Define embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Upload PDF file
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

if uploaded_file:
    st.success("PDF file uploaded successfully! 📚")

    # Save uploaded file to a temporary location
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the PDF file
    st.write("🔄 Extracting and splitting the PDF into chunks...")
    loader = PyPDFLoader("temp_uploaded_file.pdf")
    documents = loader.load()

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    st.write(f"✅ Extracted {len(chunks)} chunks from the PDF!")

    # Create embeddings and store in FAISS vector database
    st.write("⚙️ Generating embeddings and storing in FAISS...")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    st.success("FAISS database created successfully! 🚀")

    # Query the database
    st.markdown("### Ask your question about the document ⬇️")
    user_query = st.text_input("💬 Type your query here:")
    if user_query:
        st.write("🔍 Searching the FAISS database...")
        docs = vector_db.similarity_search(user_query, k=3)
        response_context = "\n".join([doc.page_content for doc in docs])

        # Use the Groq API for generating an answer
        st.write("🤖 Generating an intelligent response...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Based on this context: {response_context}\nAnswer this query: {user_query}",
                }
            ],
            model="deepseek-r1-distill-llama-70b",
        )

        response = chat_completion.choices[0].message.content
        st.markdown("### 📋 Response:")
        st.markdown(response)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Streamlit and Groq API**")
