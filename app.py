import streamlit as st
import os
import tempfile
import json
import uuid
import time
from typing import List

# ==============================
# Set page config FIRST and ONLY ONCE
# ==============================
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ==============================
# Package imports and checks
# ==============================
def check_packages():
    """Check and import all required packages with proper error handling"""
    packages_status = {}
    
    # Check pinecone first
    try:
        import pinecone
        if hasattr(pinecone, 'Pinecone'):
            packages_status['pinecone'] = True
            st.success("✅ Pinecone package: Correct version detected")
        else:
            packages_status['pinecone'] = False
            st.error("❌ Pinecone package: Wrong version detected")
    except ImportError as e:
        packages_status['pinecone'] = False
        st.error(f"❌ Pinecone package: Not installed - {e}")
    except Exception as e:
        packages_status['pinecone'] = False
        st.error(f"❌ Pinecone package: Error - {e}")

    # Check other packages
    other_packages = {
        'groq': 'Groq',
        'langchain_text_splitters': 'RecursiveCharacterTextSplitter',
        'langchain_community': 'PyPDFLoader',
        'sentence_transformers': 'SentenceTransformer',
        'langchain_core': 'Document'
    }
    
    for package, import_name in other_packages.items():
        try:
            if package == 'groq':
                from groq import Groq
            elif package == 'langchain_text_splitters':
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            elif package == 'langchain_community':
                from langchain_community.document_loaders import PyPDFLoader
            elif package == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
            elif package == 'langchain_core':
                from langchain_core.documents import Document
            packages_status[package] = True
            st.success(f"✅ {package}: Import successful")
        except ImportError as e:
            packages_status[package] = False
            st.error(f"❌ {package}: Import failed - {e}")
    
    return packages_status

# Check all packages
st.header("🔧 Package Status")
packages_status = check_packages()

if not all(packages_status.values()):
    st.error("""
    **Missing or incorrect packages detected!**
    
    Please ensure your `requirements.txt` contains:
    ```
    pinecone>=3.0.0
    groq>=0.3.0
    langchain>=0.1.0
    langchain-community>=0.0.10
    langchain-core>=0.1.0
    langchain-text-splitters>=0.0.1
    sentence-transformers>=2.2.0
    ```
    
    If you're on Streamlit Cloud:
    1. Go to your app settings
    2. Click "Clear cache and redeploy"
    3. Wait for the redeployment to complete
    """)
    st.stop()

# Now safely import all packages
import pinecone
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# ==============================
# Constants
# ==============================
CHAT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.1-405b-reasoning", 
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]
UPLOADED_FILES_JSON = "uploaded_files.json"

# ==============================
# Helper Functions
# ==============================
def validate_api_keys(pinecone_key, groq_key):
    """Validate API keys and provide specific error messages"""
    issues = []
    
    if not pinecone_key:
        issues.append("❌ Pinecone API key is missing")
    elif len(pinecone_key) < 20:
        issues.append("❌ Pinecone API key appears too short")
    
    if not groq_key:
        issues.append("❌ Groq API key is missing")
    elif not groq_key.startswith('gsk_'):
        issues.append("❌ Groq API key should start with 'gsk_'")
    elif len(groq_key) < 30:
        issues.append("❌ Groq API key appears too short")
    
    return issues

def get_available_models(groq_client):
    try:
        models = groq_client.models.list()
        available_models = [m.id for m in models.data if m.id in CHAT_MODELS]
        return available_models or CHAT_MODELS
    except Exception as e:
        st.error(f"❌ Error fetching models from Groq: {e}")
        return CHAT_MODELS

def check_environment():
    """Check environment with detailed API key validation"""
    # For Streamlit Cloud, use st.secrets
    pinecone_key = st.secrets.get("PINECONE_API_KEY", "").strip()
    groq_key = st.secrets.get("GROQ_API_KEY", "").strip()
    
    # Validate API keys
    issues = validate_api_keys(pinecone_key, groq_key)
    
    if issues:
        for issue in issues:
            st.error(issue)
        
        st.markdown("""
        **How to fix API key issues:**
        
        1. **Get your Groq API key:**
           - Go to [Groq Cloud Console](https://console.groq.com/)
           - Sign up/login to your account
           - Navigate to API Keys
           - Create a new API key or copy an existing one
           - Keys should start with `gsk_`
        
        2. **Update Streamlit Cloud secrets:**
           - Go to your app settings
           - Click on "Secrets"
           - Make sure your secrets look exactly like this:
        ```toml
        PINECONE_API_KEY = "your_pinecone_key_here"
        GROQ_API_KEY = "gsk_your_groq_key_here"
        ```
        
        3. **Redeploy after updating secrets**
        """)
        return None, None
    
    # Test Groq API key by trying to list models
    try:
        groq_client = Groq(api_key=groq_key)
        models = groq_client.models.list()
        st.success("✅ Groq API key validated successfully")
        st.success("✅ Pinecone API key format looks good")
        return pinecone_key, groq_key
    except Exception as e:
        st.error(f"❌ Groq API key validation failed: {e}")
        st.info("""
        **Possible reasons:**
        - API key is incorrect or expired
        - There might be extra spaces in your secret
        - The key doesn't have proper permissions
        - You've exceeded your rate limit
        
        **Solution:**
        - Get a new API key from [Groq Cloud](https://console.groq.com/)
        - Update your Streamlit Cloud secrets
        - Redeploy the application
        """)
        return None, None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_clients(pinecone_key, groq_key):
    try:
        pc = pinecone.Pinecone(api_key=pinecone_key)
        groq_client = Groq(api_key=groq_key)
        st.success("✅ Clients initialized successfully")
        return pc, groq_client
    except Exception as e:
        st.error(f"❌ Error initializing clients: {e}")
        st.stop()

def get_pinecone_index(pc):
    index_name = "rag-chatbot-index"
    try:
        # List existing indexes
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes]
        
        if index_name not in index_names:
            st.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            with st.spinner("Waiting for index to be ready..."):
                for i in range(30):
                    try:
                        index = pc.Index(index_name)
                        index.describe_index_stats()
                        st.success("✅ Index created and ready")
                        break
                    except Exception:
                        if i == 29:
                            st.error("❌ Timeout waiting for index to be ready")
                            st.stop()
                        time.sleep(1)
        else:
            st.success(f"✅ Using existing index: {index_name}")
        
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"❌ Error with Pinecone index: {e}")
        st.stop()

def process_document(file_path, file_type):
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs = [Document(page_content=text)]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        return [c.page_content for c in chunks]
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        return []

def store_embeddings(index, model, texts, namespace, update_existing=False):
    if not texts:
        st.warning("No text content found in document")
        return 0
        
    try:
        if update_existing:
            try:
                index.delete(delete_all=True, namespace=namespace)
                st.info(f"🔄 Cleared existing data in: {namespace}")
            except Exception as e:
                st.warning(f"Note: Could not clear existing data: {e}")
        
        embeddings = model.encode(texts).tolist()
        vectors = []
        for i, (emb, text) in enumerate(zip(embeddings, texts)):
            vectors.append({
                "id": f"{namespace}_{i}_{uuid.uuid4().hex[:8]}",
                "values": emb,
                "metadata": {"text": text, "chunk_id": i}
            })
        
        # Upsert in batches
        batch_size = 100
        total_uploaded = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            total_uploaded += len(batch)
        
        return total_uploaded
    except Exception as e:
        st.error(f"❌ Error storing embeddings: {e}")
        return 0

def search_documents(index, model, query, namespace):
    try:
        q_emb = model.encode([query]).tolist()[0]
        res = index.query(vector=q_emb, top_k=5, include_metadata=True, namespace=namespace)
        return [{"text": m.metadata.get("text", ""), "score": m.score} for m in res.matches if m.score > 0.3]
    except Exception as e:
        st.error(f"❌ Error searching documents: {e}")
        return []

def generate_response(client, context, question, model):
    try:
        if not context.strip():
            return "I couldn't find relevant information in your documents to answer this question."
            
        prompt = f"""Based on the following context from your documents, please answer the question. If the context doesn't contain relevant information, please say so.

Context:
{context}

Question: {question}

Please provide a helpful answer based only on the context provided:"""
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based only on the provided document context. If the context doesn't contain relevant information, politely say so."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        return resp.choices[0].message.content
    except Exception as e:
        # More specific error handling for Groq API errors
        error_msg = str(e)
        if "401" in error_msg and "Invalid API Key" in error_msg:
            return "❌ **API Key Error**: The Groq API key is invalid. Please check your Streamlit Cloud secrets and ensure you have a valid API key from https://console.groq.com/"
        elif "429" in error_msg:
            return "⚠️ **Rate Limit Exceeded**: You've made too many requests. Please wait a moment and try again."
        elif "500" in error_msg or "503" in error_msg:
            return "🔧 **Service Temporarily Unavailable**: The AI service is currently busy. Please try again in a few moments."
        else:
            return f"❌ **Error generating response**: {error_msg}"

def save_uploaded_files(data):
    try:
        with open(UPLOADED_FILES_JSON, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Note: Could not save file record: {e}")

def load_uploaded_files():
    try:
        if os.path.exists(UPLOADED_FILES_JSON):
            with open(UPLOADED_FILES_JSON) as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def clear_chat():
    if "messages" in st.session_state:
        st.session_state.messages = []

def on_subnamespace_change():
    if "subnamespace_selectbox" in st.session_state:
        st.session_state.selected_subnamespace = st.session_state.subnamespace_selectbox
        if st.session_state.subnamespace_selectbox == "Search All Documents in Parent":
            st.session_state.search_mode = "parent"
        else:
            st.session_state.search_mode = "subnamespace"
        clear_chat()

# ==============================
# Main App
# ==============================
def main():
    st.title("📚 RAG Chatbot - Document Q&A")
    st.markdown("Upload documents and ask questions about their content!")

    # Initialize session state
    session_defaults = {
        "messages": [],
        "uploaded_files": load_uploaded_files(),
        "selected_parent": None,
        "selected_subnamespace": "Create New Subnamespace",
        "parent_namespace_input": "",
        "subnamespace_created": False,
        "search_mode": "subnamespace"
    }
    
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Check environment with detailed validation
    pinecone_key, groq_key = check_environment()
    if not pinecone_key or not groq_key:
        # Show troubleshooting guide
        st.markdown("""
        ## 🔧 API Key Troubleshooting Guide
        
        ### For Groq API Key:
        1. **Get a free API key** from [Groq Cloud](https://console.groq.com/)
        2. **Sign up** for an account (it's free)
        3. **Navigate to API Keys** in the dashboard
        4. **Create a new API key**
        5. **Copy the key** (it should start with `gsk_`)
        
        ### For Streamlit Cloud:
        1. Go to your app's **Settings**
        2. Click on **Secrets**
        3. **Replace** the existing secrets with:
        ```toml
        PINECONE_API_KEY = "your_pinecone_key_here"
        GROQ_API_KEY = "gsk_your_actual_key_here"
        ```
        4. Click **Save**
        5. The app will **automatically redeploy**
        
        ### Common Issues:
        - ❌ **Extra spaces** in the secret value
        - ❌ **Missing quotes** around the API key
        - ❌ **Using the wrong key** (Pinecone key in Groq field or vice versa)
        - ❌ **Key is expired** or revoked
        """)
        return

    # Initialize services
    try:
        embedding_model = load_embedding_model()
        pc, groq_client = initialize_clients(pinecone_key, groq_key)
        index = get_pinecone_index(pc)
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return

    # Rest of your app code remains the same...
    # [The rest of your existing main() function code goes here]
    # ... (keeping the same sidebar, chat interface, etc.)

if __name__ == "__main__":
    main()