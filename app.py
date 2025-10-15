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
# Enhanced Environment Check for Deployment
# ==============================
def check_environment():
    # For deployment - check environment variables first
    pinecone_key = os.getenv("PINECONE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    # If not found in environment, try Streamlit secrets
    if not pinecone_key:
        try:
            pinecone_key = st.secrets.get("PINECONE_API_KEY")
        except:
            pass
            
    if not groq_key:
        try:
            groq_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    
    # Display helpful error messages
    if not pinecone_key:
        st.error("""
        ❌ PINECONE_API_KEY not found!
        
        For deployment, please set this as an environment variable:
        - In Streamlit Cloud: Go to App Settings → Secrets
        - Add: PINECONE_API_KEY=your_pinecone_key_here
        """)
        
    if not groq_key:
        st.error("""
        ❌ GROQ_API_KEY not found!
        
        For deployment, please set this as an environment variable:
        - In Streamlit Cloud: Go to App Settings → Secrets  
        - Add: GROQ_API_KEY=your_groq_key_here
        """)
    
    if not pinecone_key or not groq_key:
        st.info("""
        🔧 Setup Instructions:
        1. Get Pinecone API key: https://www.pinecone.io/
        2. Get Groq API key: https://console.groq.com/
        3. Add both as secrets in your deployment platform
        """)
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
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else existing_indexes.names()
        
        if index_name not in index_names:
            # Create index with latest Pinecone client syntax
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
                time.sleep(2)
            except Exception as e:
                st.warning(f"Note: Could not delete existing namespace (might not exist): {e}")
        
        if not texts:
            st.warning("No text content found in the document.")
            return 0
            
        embeddings = model.encode(texts).tolist()
        vectors = []
        for i, (emb, text) in enumerate(zip(embeddings, texts)):
            vectors.append({
                "id": f"doc_{uuid.uuid4().hex[:8]}_{i}",
                "values": emb, 
                "metadata": {"text": text}
            })
        
        # Upload in batches
        batch_size = 50
        total_uploaded = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=namespace)
                total_uploaded += len(batch)
                time.sleep(0.5)
            except Exception as e:
                st.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                
        return total_uploaded
    except Exception as e:
        st.error(f"Error storing embeddings: {e}")
        return 0

def search_documents(index, model, query, namespace):
    try:
        q_emb = model.encode([query]).tolist()[0]
        res = index.query(vector=q_emb, top_k=5, include_metadata=True, namespace=namespace)
        return [{"text": m.metadata.get("text", ""), "score": m.score} for m in res.matches if m.score > 0.3]
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

def generate_response(client, context, question, model):
    try:
        if context:
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Please provide a helpful answer based on the context:"""
        else:
            prompt = f"""Question: {question}

Please provide a helpful answer:"""

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def save_uploaded_files(data):
    try:
        with open(UPLOADED_FILES_JSON, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving uploaded files data: {e}")

def load_uploaded_files():
    try:
        if os.path.exists(UPLOADED_FILES_JSON):
            with open(UPLOADED_FILES_JSON, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def clear_chat():
    if "messages" in st.session_state:
        st.session_state.messages = []

def cleanup_temp_files():
    """Clean up temporary files to save memory"""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.endswith(('.pdf', '.md', '.txt')):
                try:
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.getctime(file_path) < time.time() - 3600:
                        os.unlink(file_path)
                except:
                    pass
    except:
        pass

# Callback function for subnamespace selection
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
    st.set_page_config(
        page_title="RAG Chatbot", 
        layout="wide",
        page_icon="📚"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📚 Multi-Level Namespace RAG Chatbot</h1>', unsafe_allow_html=True)

    # Initialize session_state
    session_defaults = {
        "messages": [],
        "uploaded_files": load_uploaded_files(),
        "selected_parent": None,
        "selected_subnamespace": "Create New Subnamespace",
        "parent_namespace_input": "",
        "subnamespace_created": False,
        "search_mode": "subnamespace"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Check packages
    if not PACKAGES_AVAILABLE:
        st.error("Missing required packages. Please check the requirements.")
        return

    # Check environment and initialize
    pinecone_key, groq_key = check_environment()
    if not pinecone_key or not groq_key:
        return

    # Initialize components with error handling
    embedding_model = load_embedding_model()
    if embedding_model is None:
        return
        
    pc, groq_client = initialize_clients(pinecone_key, groq_key)
    if pc is None or groq_client is None:
        return
        
    index = get_pinecone_index(pc)
    if index is None:
        return

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    available_models = get_available_models(groq_client)
    selected_model = st.sidebar.selectbox("Select Groq Model", available_models, index=0)
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("📂 Namespace Management")

    # Get all namespaces with error handling
    try:
        index_stats = index.describe_index_stats()
        current_stats = index_stats.get("namespaces", {})
        all_namespaces = list(current_stats.keys())
    except Exception as e:
        st.error(f"Error fetching namespaces: {e}")
        current_stats = {}
        all_namespaces = []

    # Parent namespaces
    parent_namespaces = sorted(list({ns.split("__")[0] for ns in all_namespaces if "__" in ns}))

    # Parent namespace selection
    parent_options = ["Create New Parent"] + parent_namespaces
    
    # Handle initial selection
    if st.session_state.selected_parent is None and parent_namespaces:
        st.session_state.selected_parent = parent_namespaces[0]
    
    # Find the current index for the selectbox
    current_parent_index = 0
    if st.session_state.selected_parent in parent_options:
        current_parent_index = parent_options.index(st.session_state.selected_parent)
    
    parent_selected = st.sidebar.selectbox(
        "Select Parent Namespace",
        options=parent_options,
        index=current_parent_index,
        key="parent_selectbox"
    )

    # Handle parent namespace creation/selection
    if parent_selected == "Create New Parent":
        parent_input = st.sidebar.text_input(
            "Enter Parent Namespace Name (no spaces, lowercase)", 
            value=st.session_state.parent_namespace_input,
            key="parent_namespace_input",
            placeholder="e.g., my_documents"
        )
        parent_namespace = parent_input.strip().lower() if parent_input else None
        if parent_namespace and " " in parent_namespace:
            st.sidebar.warning("Please remove spaces from namespace name")
            parent_namespace = None
    else:
        parent_namespace = parent_selected
        st.session_state.selected_parent = parent_namespace

    # Subnamespace handling
    subnamespaces = []
    if parent_namespace:
        subnamespaces = [ns for ns in all_namespaces if ns.startswith(f"{parent_namespace}__")]
    
    # Build subnamespace options
    if parent_namespace and subnamespaces:
        sub_options = ["Search All Documents in Parent"] + ["Create New Subnamespace"] + subnamespaces
    else:
        sub_options = ["Create New Subnamespace"] + subnamespaces
    
    # Handle subnamespace selection persistence
    if st.session_state.selected_subnamespace not in sub_options:
        if subnamespaces:
            st.session_state.selected_subnamespace = "Search All Documents in Parent"
            st.session_state.search_mode = "parent"
        else:
            st.session_state.selected_subnamespace = "Create New Subnamespace"
            st.session_state.search_mode = "subnamespace"

    # Calculate current index for the selectbox
    current_sub_index = 0
    if st.session_state.selected_subnamespace in sub_options:
        current_sub_index = sub_options.index(st.session_state.selected_subnamespace)

    # Use the selectbox with proper key and callback
    sub_selected = st.sidebar.selectbox(
        "Select Search Scope",
        options=sub_options,
        index=current_sub_index,
        key="subnamespace_selectbox",
        on_change=on_subnamespace_change
    )

    # File uploader
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF or Markdown", 
        type=["pdf", "md"], 
        key="file_uploader",
        help="Upload PDF or Markdown files to create or update namespaces"
    )
    
    tmp_path, file_ext = None, None
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        tmp_path = tmp_file.name

    # Upload new subnamespace
    if uploaded_file and sub_selected == "Create New Subnamespace":
        if st.sidebar.button("📤 Upload as New Subnamespace", use_container_width=True):
            if tmp_path and parent_namespace:
                with st.spinner("Processing document..."):
                    subname = uploaded_file.name.split(".")[0].replace(" ", "_").lower()[:50]
                    full_namespace = f"{parent_namespace}__{subname}"
                    texts = process_document(tmp_path, file_ext)
                    if texts:
                        count = store_embeddings(index, embedding_model, texts, full_namespace, update_existing=False)
                        if count > 0:
                            st.success(f"✅ Created subnamespace '{full_namespace}' with {count} chunks.")
                            st.session_state.uploaded_files[uploaded_file.name] = full_namespace
                            save_uploaded_files(st.session_state.uploaded_files)
                            
                            st.session_state.selected_subnamespace = full_namespace
                            st.session_state.search_mode = "subnamespace"
                            st.session_state.subnamespace_created = True
                            clear_chat()
                            st.rerun()
                        else:
                            st.error("Failed to store embeddings. Please try again.")
                    else:
                        st.error("No text content could be extracted from the document.")
                os.unlink(tmp_path)
            else:
                st.error("Please enter a parent namespace name first.")

    # Update existing subnamespace
    if uploaded_file and sub_selected not in ["Create New Subnamespace", "Search All Documents in Parent"]:
        if st.sidebar.button("🔄 Update Selected Subnamespace", use_container_width=True):
            if tmp_path and st.session_state.selected_subnamespace:
                with st.spinner("Updating document..."):
                    texts = process_document(tmp_path, file_ext)
                    if texts:
                        full_namespace = st.session_state.selected_subnamespace
                        count = store_embeddings(index, embedding_model, texts, full_namespace, update_existing=True)
                        if count > 0:
                            st.success(f"✅ Updated '{full_namespace}' with {count} chunks.")
                            st.session_state.uploaded_files[uploaded_file.name] = full_namespace
                            save_uploaded_files(st.session_state.uploaded_files)
                            clear_chat()
                            st.rerun()
                        else:
                            st.error("Failed to update embeddings. Please try again.")
                    else:
                        st.error("No text content could be extracted from the document.")
                os.unlink(tmp_path)

    # Delete subnamespace
    if sub_selected not in ["Create New Subnamespace", "Search All Documents in Parent"]:
        if st.sidebar.button("🗑️ Delete Selected Subnamespace", type="secondary", use_container_width=True):
            if st.session_state.selected_subnamespace:
                try:
                    namespace_to_delete = st.session_state.selected_subnamespace
                    index.delete(delete_all=True, namespace=namespace_to_delete)
                    st.success(f"✅ Deleted subnamespace '{namespace_to_delete}'")
                    
                    st.session_state.uploaded_files = {k: v for k, v in st.session_state.uploaded_files.items() 
                                                     if v != namespace_to_delete}
                    save_uploaded_files(st.session_state.uploaded_files)
                    
                    remaining_subnamespaces = [ns for ns in subnamespaces if ns != namespace_to_delete]
                    if remaining_subnamespaces:
                        st.session_state.selected_subnamespace = "Search All Documents in Parent"
                        st.session_state.search_mode = "parent"
                    else:
                        st.session_state.selected_subnamespace = "Create New Subnamespace"
                        st.session_state.search_mode = "subnamespace"
                    
                    clear_chat()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting subnamespace: {e}")

    # ---------------- Chat Area ----------------
    st.markdown("### 💬 Chat with your document")
    
    # Display current search scope info
    if st.session_state.selected_subnamespace:
        if st.session_state.search_mode == "parent" and st.session_state.selected_parent:
            st.info(f"**Search Scope:** 🔍 All documents in parent namespace `{st.session_state.selected_parent}`")
            if subnamespaces:
                st.write(f"**Included subnamespaces:** {', '.join([ns.split('__')[1] for ns in subnamespaces])}")
        elif st.session_state.search_mode == "subnamespace" and st.session_state.selected_subnamespace:
            st.info(f"**Search Scope:** 📄 Single document `{st.session_state.selected_subnamespace}`")
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.selected_parent:
            st.warning("⚠️ Please select or create a parent namespace first.")
        elif st.session_state.search_mode == "parent" and not subnamespaces:
            st.warning("⚠️ No documents found in this parent namespace. Please upload a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    namespaces_to_search = []
                    
                    if st.session_state.search_mode == "parent":
                        namespaces_to_search = subnamespaces
                        if namespaces_to_search:
                            st.info(f"🔍 Searching across {len(namespaces_to_search)} documents...")
                    else:
                        if st.session_state.selected_subnamespace and st.session_state.selected_subnamespace not in ["Create New Subnamespace", "Search All Documents in Parent"]:
                            namespaces_to_search = [st.session_state.selected_subnamespace]
                            st.info(f"🔍 Searching in single document...")
                    
                    all_results = []
                    for ns in namespaces_to_search:
                        results = search_documents(index, embedding_model, prompt, ns)
                        all_results.extend(results)
                    
                    all_results.sort(key=lambda x: x["score"], reverse=True)
                    top_results = all_results[:10]

                    if top_results:
                        context = "\n\n".join([f"[Score: {r['score']:.3f}] {r['text']}" for r in top_results])
                        answer = generate_response(groq_client, context, prompt, selected_model)
                        
                        st.write(f"**Found {len(top_results)} relevant chunks from {len(namespaces_to_search)} namespace(s)**")
                        
                        with st.expander("View top matches"):
                            for i, result in enumerate(top_results[:3]):
                                st.write(f"**Match {i+1}** (Score: {result['score']:.3f}):")
                                st.write(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
                                st.write("---")
                    else:
                        answer = "No relevant information found in the selected namespace(s)."
                        if st.session_state.search_mode == "parent":
                            answer += " Try searching in a specific document or check if your documents contain relevant information."
                        else:
                            answer += " The document might not contain information related to your question."
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sidebar - uploaded files and namespace info
    st.sidebar.markdown("---")
    st.sidebar.write("**📁 Uploaded Files & Namespaces:**")
    if st.session_state.uploaded_files:
        for name, ns in list(st.session_state.uploaded_files.items())[:10]:
            st.sidebar.write(f"- `{name}` → `{ns}`")
        if len(st.session_state.uploaded_files) > 10:
            st.sidebar.write(f"... and {len(st.session_state.uploaded_files) - 10} more")
    else:
        st.sidebar.write("No files uploaded yet")
    
    # Namespace statistics
    st.sidebar.markdown("---")
    st.sidebar.write("**📊 Namespace Statistics:**")
    total_vectors = 0
    try:
        if st.session_state.selected_parent:
            parent_vectors = 0
            for ns, stats in current_stats.items():
                if ns.startswith(f"{st.session_state.selected_parent}__"):
                    vector_count = stats.get('vector_count', 0)
                    parent_vectors += vector_count
                    st.sidebar.write(f"- `{ns}`: {vector_count} vectors")
            st.sidebar.write(f"**Total in parent:** {parent_vectors} vectors")
        else:
            for ns, stats in current_stats.items():
                vector_count = stats.get('vector_count', 0)
                total_vectors += vector_count
                st.sidebar.write(f"- `{ns}`: {vector_count} vectors")
            st.sidebar.write(f"**Total vectors:** {total_vectors}")
    except Exception as e:
        st.sidebar.write("Unable to load statistics")

    # Cleanup temp files periodically
    cleanup_temp_files()

if __name__ == "__main__":
    main()