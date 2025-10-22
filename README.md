# RAG Chatbot with Pinecone

A Streamlit-based RAG (Retrieval Augmented Generation) chatbot that uses Pinecone for vector storage and Groq for LLM responses.

## Features

- 📄 Upload PDF and Markdown documents
- 🔍 Semantic search using Pinecone
- 💬 Chat interface with Groq LLM
- 🗂️ Multi-level namespace management
- 🚀 Hosted on Streamlit Cloud

## Setup for Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Set up secrets in Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Create new app
   - Connect your GitHub repository
   - Add these secrets in the "Secrets" section:

```toml
PINECONE_API_KEY = "your_pinecone_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"