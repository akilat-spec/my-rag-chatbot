# RAG Chatbot with Multi-Level Namespaces

A powerful Streamlit-based RAG (Retrieval Augmented Generation) chatbot that allows you to upload documents and chat with them using Pinecone for vector storage and Groq for fast LLM responses.

## Features

- 📚 Upload PDF and Markdown documents
- 🏷️ Multi-level namespace management
- 🔍 Semantic search across documents
- 💬 Real-time chat interface
- 🚀 Fast responses using Groq's LLMs
- ☁️ Cloud-ready deployment

## Setup Instructions

### 1. Get API Keys

- **Pinecone**: Get free API key from [pinecone.io](https://www.pinecone.io/)
- **Groq**: Get free API key from [console.groq.com](https://console.groq.com/)

### 2. Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd my-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py