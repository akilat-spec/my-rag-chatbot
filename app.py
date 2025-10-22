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
        # Check if it's the correct package by looking for the Pinecone class
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
        'langchain': 'RecursiveCharacterTextSplitter',
        'langchain_community': 'PyPDFLoader',
        'sentence_transformers': 'SentenceTransformer'
    }
    
    for package, import_name in other_packages.items():
        try:
            if package == 'groq':
                from groq import Groq
            elif package == 'langchain':
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            elif package == 'langchain_community':
                from langchain_community.document_loaders import PyPDFLoader
            elif package == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

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
def get_available_models(groq_client):
    try:
        models = groq_client.models.list()
        available_models = [m.id for m in models.data if m.id in CHAT_MODELS]
        return available_models or CHAT_MODELS
    except Exception:
        return CHAT_MODELS

def check_environment():
    # For Streamlit Cloud, use st.secrets
    pinecone_key = st.secrets.get("PINECONE_API_KEY")
    groq_key = st.secrets.get("GROQ_API_KEY")
    
    if not pinecone_key:
        st.error("❌ Missing Pinecone API key")
        st.info("Please add PINECONE_API_KEY to your Streamlit Cloud secrets")
        return None, None
    if not groq_key:
        st.error("❌ Missing Groq API key")
        st.info("Please add GROQ_API_KEY to your Streamlit Cloud secrets")
        return None, None
    
    st.success("✅ API keys found in secrets")
    return pinecone_key, groq_key

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
        return f"I encountered an error while generating a response. Please try again. Error: {str(e)}"

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

    # Check environment
    pinecone_key, groq_key = check_environment()
    if not pinecone_key or not groq_key:
        st.info("Please add your API keys to Streamlit Cloud secrets to continue.")
        return

    # Initialize services
    try:
        embedding_model = load_embedding_model()
        pc, groq_client = initialize_clients(pinecone_key, groq_key)
        index = get_pinecone_index(pc)
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    available_models = get_available_models(groq_client)
    selected_model = st.sidebar.selectbox("Choose AI Model", available_models)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📂 Document Management")

    # Get existing namespaces
    try:
        index_stats = index.describe_index_stats()
        current_stats = index_stats.get("namespaces", {})
        all_namespaces = list(current_stats.keys())
    except Exception as e:
        st.error(f"❌ Could not fetch namespace data: {e}")
        all_namespaces = []
        current_stats = {}

    # Parent namespace management
    parent_namespaces = sorted(list({ns.split("__")[0] for ns in all_namespaces if "__" in ns}))
    parent_options = ["Create New Collection"] + parent_namespaces
    
    if st.session_state.selected_parent is None and parent_namespaces:
        st.session_state.selected_parent = parent_namespaces[0]
    
    current_parent_index = parent_options.index(st.session_state.selected_parent) if st.session_state.selected_parent in parent_options else 0
    
    parent_selected = st.sidebar.selectbox(
        "Select Document Collection",
        options=parent_options,
        index=current_parent_index,
        key="parent_selectbox"
    )

    if parent_selected == "Create New Collection":
        parent_input = st.sidebar.text_input(
            "New Collection Name (letters, numbers, underscores only)",
            value=st.session_state.parent_namespace_input,
            key="parent_namespace_input"
        )
        parent_namespace = parent_input.strip() if parent_input else None
    else:
        parent_namespace = parent_selected
        st.session_state.selected_parent = parent_namespace

    # Document (subnamespace) management
    subnamespaces = []
    if parent_namespace:
        subnamespaces = [ns for ns in all_namespaces if ns.startswith(f"{parent_namespace}__")]
    
    if parent_namespace and subnamespaces:
        sub_options = ["Search All Documents"] + ["Upload New Document"] + subnamespaces
    else:
        sub_options = ["Upload New Document"] + subnamespaces
    
    if st.session_state.selected_subnamespace not in sub_options:
        if subnamespaces:
            st.session_state.selected_subnamespace = "Search All Documents"
            st.session_state.search_mode = "parent"
        else:
            st.session_state.selected_subnamespace = "Upload New Document"
            st.session_state.search_mode = "subnamespace"

    current_sub_index = sub_options.index(st.session_state.selected_subnamespace) if st.session_state.selected_subnamespace in sub_options else 0

    sub_selected = st.sidebar.selectbox(
        "Document Selection",
        options=sub_options,
        index=current_sub_index,
        key="subnamespace_selectbox",
        on_change=on_subnamespace_change
    )

    # File upload
    uploaded_file = st.sidebar.file_uploader("Choose PDF or Markdown file", type=["pdf", "md"])
    tmp_path = None
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        tmp_path = tmp_file.name

    # Document operations
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if uploaded_file and sub_selected == "Upload New Document" and st.button("📤 Upload", use_container_width=True):
            if tmp_path and parent_namespace:
                doc_name = uploaded_file.name.split(".")[0].replace(" ", "_").lower()[:50]
                full_namespace = f"{parent_namespace}__{doc_name}"
                with st.spinner("Processing document..."):
                    texts = process_document(tmp_path, file_ext)
                    if texts:
                        count = store_embeddings(index, embedding_model, texts, full_namespace, False)
                        if count > 0:
                            st.success(f"✅ Uploaded '{doc_name}' with {count} sections")
                            st.session_state.uploaded_files[uploaded_file.name] = full_namespace
                            save_uploaded_files(st.session_state.uploaded_files)
                            st.session_state.selected_subnamespace = full_namespace
                            st.session_state.search_mode = "subnamespace"
                            clear_chat()
                        else:
                            st.error("❌ Failed to store document content")
                    else:
                        st.error("❌ Could not extract text from document")
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                st.rerun()

    with col2:
        if sub_selected not in ["Upload New Document", "Search All Documents"] and st.button("🗑️ Delete", use_container_width=True, type="secondary"):
            if st.session_state.selected_subnamespace:
                try:
                    namespace_to_delete = st.session_state.selected_subnamespace
                    index.delete(delete_all=True, namespace=namespace_to_delete)
                    st.success(f"✅ Deleted document")
                    st.session_state.uploaded_files = {k: v for k, v in st.session_state.uploaded_files.items() if v != namespace_to_delete}
                    save_uploaded_files(st.session_state.uploaded_files)
                    
                    remaining = [ns for ns in subnamespaces if ns != namespace_to_delete]
                    if remaining:
                        st.session_state.selected_subnamespace = "Search All Documents"
                        st.session_state.search_mode = "parent"
                    else:
                        st.session_state.selected_subnamespace = "Upload New Document"
                        st.session_state.search_mode = "subnamespace"
                    
                    clear_chat()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Delete failed: {e}")

    # Main chat area
    st.markdown("### 💬 Chat with Your Documents")
    
    # Search scope info
    if st.session_state.selected_subnamespace:
        if st.session_state.search_mode == "parent" and st.session_state.selected_parent:
            st.info(f"**Searching:** All documents in **{st.session_state.selected_parent}**")
            if subnamespaces:
                st.write(f"**Documents:** {', '.join([ns.split('__')[1] for ns in subnamespaces])}")
        elif st.session_state.search_mode == "subnamespace" and st.session_state.selected_subnamespace:
            doc_name = st.session_state.selected_subnamespace.split('__')[1] if '__' in st.session_state.selected_subnamespace else st.session_state.selected_subnamespace
            st.info(f"**Searching:** Single document - **{doc_name}**")

    # Chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.selected_parent:
            st.warning("👆 Please create or select a document collection first")
        elif st.session_state.search_mode == "parent" and not subnamespaces:
            st.warning("📁 No documents found. Please upload a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    namespaces_to_search = []
                    
                    if st.session_state.search_mode == "parent":
                        namespaces_to_search = subnamespaces
                    else:
                        if st.session_state.selected_subnamespace and st.session_state.selected_subnamespace not in ["Upload New Document", "Search All Documents"]:
                            namespaces_to_search = [st.session_state.selected_subnamespace]
                    
                    all_results = []
                    for ns in namespaces_to_search:
                        results = search_documents(index, embedding_model, prompt, ns)
                        all_results.extend(results)
                    
                    all_results.sort(key=lambda x: x["score"], reverse=True)
                    top_results = all_results[:8]

                    if top_results:
                        context = "\n\n".join([f"[Section {i+1}] {r['text']}" for i, r in enumerate(top_results)])
                        answer = generate_response(groq_client, context, prompt, selected_model)
                        st.write(f"*Found {len(top_results)} relevant sections from {len(namespaces_to_search)} document(s)*")
                    else:
                        answer = "I couldn't find any relevant information in your documents to answer this question."
                        if st.session_state.search_mode == "parent":
                            answer += " You might want to upload more documents or try asking about different topics."
                        else:
                            answer += " This particular document might not contain information about your question."
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.write("**Uploaded Files:**")
    for name, ns in list(st.session_state.uploaded_files.items())[-5:]:  # Show last 5
        st.sidebar.write(f"• {name}")
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Storage Info:**")
    total_vectors = 0
    if st.session_state.selected_parent:
        parent_vectors = 0
        for ns, stats in current_stats.items():
            if ns.startswith(f"{st.session_state.selected_parent}__"):
                vector_count = stats.get('vector_count', 0)
                parent_vectors += vector_count
        st.sidebar.write(f"Vectors in collection: **{parent_vectors}**")
    else:
        for ns, stats in current_stats.items():
            vector_count = stats.get('vector_count', 0)
            total_vectors += vector_count
        st.sidebar.write(f"Total vectors: **{total_vectors}**")

if __name__ == "__main__":
    main()