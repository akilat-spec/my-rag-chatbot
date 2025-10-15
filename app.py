import streamlit as st
import os
import tempfile
import json
import uuid
import time
from typing import List

# ==============================
# Package imports and checks
# ==============================
try:
    import pinecone
    st.success("✅ Pinecone package detected")
except ImportError as e:
    st.error(f"❌ Pinecone import error: {e}")
    st.info("Please install with: pip install pinecone-client")
    st.stop()

try:
    from groq import Groq
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from sentence_transformers import SentenceTransformer
    from langchain.schema import Document
    PACKAGES_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Missing required package: {e}")
    PACKAGES_AVAILABLE = False

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
# Environment Check
# ==============================
def check_environment():
    # For Streamlit Cloud - secrets are in st.secrets
    try:
        pinecone_key = st.secrets.get("PINECONE_API_KEY")
        groq_key = st.secrets.get("GROQ_API_KEY")
    except:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
    
    if not pinecone_key:
        st.error("""
        ❌ PINECONE_API_KEY not found!
        
        Please add it to your Streamlit secrets:
        1. Go to your app settings
        2. Click on 'Secrets'
        3. Add:
        PINECONE_API_KEY=your_pinecone_key_here
        GROQ_API_KEY=your_groq_key_here
        """)
        
    if not groq_key:
        st.error("""
        ❌ GROQ_API_KEY not found!
        
        Please add it to your Streamlit secrets.
        """)
    
    if not pinecone_key or not groq_key:
        return None, None
        
    return pinecone_key, groq_key

# ==============================
# Helper Functions
# ==============================
def get_available_models(groq_client):
    try:
        models = groq_client.models.list()
        available_models = [m.id for m in models.data if m.id in CHAT_MODELS]
        return available_models or CHAT_MODELS
    except Exception as e:
        st.warning(f"Could not fetch models from Groq: {e}. Using default models.")
        return CHAT_MODELS

@st.cache_resource
def load_embedding_model():
    try:
        # Use a smaller model for Streamlit Cloud compatibility
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

def initialize_clients(pinecone_key, groq_key):
    try:
        pc = pinecone.Pinecone(api_key=pinecone_key)
        groq_client = Groq(api_key=groq_key)
        return pc, groq_client
    except Exception as e:
        st.error(f"Failed to initialize clients: {e}")
        return None, None

def get_pinecone_index(pc):
    try:
        index_name = "rag-chatbot-index"
        existing_indexes = pc.list_indexes()
        
        # Handle different response formats from Pinecone client
        if hasattr(existing_indexes, 'indexes'):
            index_names = [index.name for index in existing_indexes.indexes]
        else:
            index_names = existing_indexes.names()
        
        if index_name not in index_names:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.info("🔄 Creating new Pinecone index... This may take up to 30 seconds.")
            time.sleep(30)
        
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Failed to get Pinecone index: {e}")
        return None

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
        st.error(f"Error processing document: {e}")
        return []

def store_embeddings(index, model, texts, namespace, update_existing=False):
    try:
        if update_existing:
            try:
                index.delete(delete_all=True, namespace=namespace)
                time.sleep(1)
            except Exception:
                pass
        
        if not texts:
            st.warning("No text content found in the document.")
            return 0
            
        embeddings = model.encode(texts).tolist()
        vectors = []
        for i, (emb, text) in enumerate(zip(embeddings, texts)):
            vectors.append({
                "id": f"doc_{i}_{uuid.uuid4().hex[:6]}",
                "values": emb, 
                "metadata": {"text": text}
            })
        
        # Upload in smaller batches for Streamlit Cloud
        batch_size = 20
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            
        return len(vectors)
    except Exception as e:
        st.error(f"Error storing embeddings: {e}")
        return 0

def search_documents(index, model, query, namespace):
    try:
        q_emb = model.encode([query]).tolist()[0]
        res = index.query(vector=q_emb, top_k=3, include_metadata=True, namespace=namespace)
        return [{"text": m.metadata.get("text", ""), "score": m.score} for m in res.matches if m.score > 0.3]
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

def generate_response(client, context, question, model):
    try:
        if context:
            prompt = f"""Based on this context, answer the question. If unsure, say so.

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def save_uploaded_files(data):
    try:
        with open(UPLOADED_FILES_JSON, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_uploaded_files():
    try:
        if os.path.exists(UPLOADED_FILES_JSON):
            with open(UPLOADED_FILES_JSON, "r") as f:
                return json.load(f)
    except:
        pass
    return {}

def clear_chat():
    st.session_state.messages = []

# ==============================
# Main App
# ==============================
def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        layout="wide",
        page_icon="📚"
    )
    
    st.title("📚 RAG Chatbot with Pinecone & Groq")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = load_uploaded_files()
    if "selected_namespace" not in st.session_state:
        st.session_state.selected_namespace = None

    # Check packages
    if not PACKAGES_AVAILABLE:
        st.error("Missing required packages.")
        return

    # Initialize clients
    pinecone_key, groq_key = check_environment()
    if not pinecone_key or not groq_key:
        return

    embedding_model = load_embedding_model()
    pc, groq_client = initialize_clients(pinecone_key, groq_key)
    
    if pc is None or groq_client is None or embedding_model is None:
        st.error("Failed to initialize required services.")
        return
        
    index = get_pinecone_index(pc)
    if index is None:
        return

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    available_models = get_available_models(groq_client)
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    
    if st.sidebar.button("Clear Chat"):
        clear_chat()

    st.sidebar.header("📂 Document Management")
    
    # Get existing namespaces
    try:
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys())
    except:
        namespaces = []

    # Namespace selection
    namespace_options = ["Create New"] + namespaces
    selected_ns = st.sidebar.selectbox("Select Namespace", namespace_options)
    
    if selected_ns == "Create New":
        new_ns = st.sidebar.text_input("New Namespace Name")
        namespace = new_ns.strip().lower() if new_ns else None
    else:
        namespace = selected_ns
        st.session_state.selected_namespace = namespace

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file and namespace:
        if st.sidebar.button("Process Document"):
            with st.spinner("Processing..."):
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process and store
                texts = process_document(tmp_path, "pdf")
                if texts:
                    count = store_embeddings(index, embedding_model, texts, namespace)
                    if count > 0:
                        st.success(f"✅ Added {count} chunks to '{namespace}'")
                        st.session_state.uploaded_files[uploaded_file.name] = namespace
                        save_uploaded_files(st.session_state.uploaded_files)
                    else:
                        st.error("Failed to process document")
                
                # Cleanup
                os.unlink(tmp_path)

    # Chat interface
    st.header("💬 Chat")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not namespace:
            st.warning("Please select or create a namespace first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    # Search for relevant content
                    results = search_documents(index, embedding_model, prompt, namespace)
                    
                    if results:
                        context = "\n".join([r["text"] for r in results])
                        answer = generate_response(groq_client, context, prompt, selected_model)
                        st.write(f"Found {len(results)} relevant sections")
                    else:
                        answer = "No relevant information found. Try uploading documents first."
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display uploaded files
    st.sidebar.header("📁 Your Files")
    for filename, ns in st.session_state.uploaded_files.items():
        st.sidebar.write(f"• {filename} → {ns}")

if __name__ == "__main__":
    main()