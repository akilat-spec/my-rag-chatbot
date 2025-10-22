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
try:
    import pinecone
    if hasattr(pinecone, 'Pinecone'):
        st.success("✅ Correct Pinecone package detected")
    else:
        st.error("❌ Wrong Pinecone package. Please install 'pinecone' not 'pinecone-client'")
        st.stop()
except ImportError as e:
    st.error(f"❌ Pinecone import error: {e}")
    st.info("Please install with: pip install pinecone")
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
    pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not pinecone_key or not groq_key:
        st.error("❌ Missing Pinecone or Groq API key")
        return None, None
    return pinecone_key, groq_key

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_clients(pinecone_key, groq_key):
    pc = pinecone.Pinecone(api_key=pinecone_key)
    groq_client = Groq(api_key=groq_key)
    return pc, groq_client

def get_pinecone_index(pc):
    index_name = "rag-chatbot-index"
    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
    return pc.Index(index_name)

def process_document(file_path, file_type):
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

def store_embeddings(index, model, texts, namespace, update_existing=False):
    if update_existing:
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass
    embeddings = model.encode(texts).tolist()
    vectors = [{"id": f"doc_{uuid.uuid4().hex[:8]}", "values": emb, "metadata": {"text": t}} for emb, t in zip(embeddings, texts)]
    index.upsert(vectors=vectors, namespace=namespace)
    return len(vectors)

def search_documents(index, model, query, namespace):
    q_emb = model.encode([query]).tolist()[0]
    res = index.query(vector=q_emb, top_k=5, include_metadata=True, namespace=namespace)
    return [{"text": m.metadata.get("text", ""), "score": m.score} for m in res.matches if m.score > 0.3]

def generate_response(client, context, question, model):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return resp.choices[0].message.content

def save_uploaded_files(data):
    with open(UPLOADED_FILES_JSON, "w") as f:
        json.dump(data, f, indent=4)

def load_uploaded_files():
    if os.path.exists(UPLOADED_FILES_JSON):
        with open(UPLOADED_FILES_JSON) as f:
            return json.load(f)
    return {}

def clear_chat():
    if "messages" in st.session_state:
        st.session_state.messages = []

# Callback function for subnamespace selection
def on_subnamespace_change():
    if "subnamespace_selectbox" in st.session_state:
        st.session_state.selected_subnamespace = st.session_state.subnamespace_selectbox
        # Update search mode based on selection
        if st.session_state.subnamespace_selectbox == "Search All Documents in Parent":
            st.session_state.search_mode = "parent"
        else:
            st.session_state.search_mode = "subnamespace"
        clear_chat()

# ==============================
# Main App
# ==============================
def main():
    st.title("📚 Multi-Level Namespace Chatbot")

    # Initialize session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = load_uploaded_files()
    if "selected_parent" not in st.session_state:
        st.session_state.selected_parent = None
    if "selected_subnamespace" not in st.session_state:
        st.session_state.selected_subnamespace = "Create New Subnamespace"
    if "parent_namespace_input" not in st.session_state:
        st.session_state.parent_namespace_input = ""
    if "subnamespace_created" not in st.session_state:
        st.session_state.subnamespace_created = False
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "subnamespace"

    if not PACKAGES_AVAILABLE:
        st.error("Missing required packages.")
        return

    pinecone_key, groq_key = check_environment()
    if not pinecone_key or not groq_key:
        return

    embedding_model = load_embedding_model()
    pc, groq_client = initialize_clients(pinecone_key, groq_key)
    index = get_pinecone_index(pc)

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Settings")
    available_models = get_available_models(groq_client)
    selected_model = st.sidebar.selectbox("Select Groq Model", available_models)

    st.sidebar.markdown("---")
    st.sidebar.header("📂 Namespace Management")

    # Get all namespaces
    current_stats = index.describe_index_stats().get("namespaces", {})
    all_namespaces = list(current_stats.keys())

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
            "Enter Parent Namespace Name (no spaces)", 
            value=st.session_state.parent_namespace_input,
            key="parent_namespace_input"
        )
        parent_namespace = parent_input.strip() if parent_input else None
    else:
        parent_namespace = parent_selected
        st.session_state.selected_parent = parent_namespace

    # Subnamespace handling with proper persistence
    subnamespaces = []
    if parent_namespace:
        subnamespaces = [ns for ns in all_namespaces if ns.startswith(f"{parent_namespace}__")]
    
    # Build subnamespace options
    if parent_namespace and subnamespaces:
        sub_options = ["Search All Documents in Parent"] + ["Create New Subnamespace"] + subnamespaces
    else:
        sub_options = ["Create New Subnamespace"] + subnamespaces
    
    # Handle subnamespace selection persistence
    # If current selection is not in options, find the best alternative
    if st.session_state.selected_subnamespace not in sub_options:
        if subnamespaces:
            # If we have subnamespaces, default to search all
            st.session_state.selected_subnamespace = "Search All Documents in Parent"
            st.session_state.search_mode = "parent"
        else:
            # If no subnamespaces, default to create new
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
    uploaded_file = st.sidebar.file_uploader("Upload PDF/Markdown", type=["pdf", "md"], key="file_uploader")
    tmp_path, file_ext = None, None
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        tmp_path = tmp_file.name

    # Upload new subnamespace
    if uploaded_file and sub_selected == "Create New Subnamespace" and st.sidebar.button("📤 Upload as New Subnamespace"):
        if tmp_path and parent_namespace:
            subname = uploaded_file.name.split(".")[0].replace(" ", "_").lower()
            full_namespace = f"{parent_namespace}__{subname}"
            texts = process_document(tmp_path, file_ext)
            count = store_embeddings(index, embedding_model, texts, full_namespace, update_existing=False)
            st.success(f"✅ Created subnamespace '{full_namespace}' with {count} chunks.")
            st.session_state.uploaded_files[uploaded_file.name] = full_namespace
            save_uploaded_files(st.session_state.uploaded_files)
            
            # Update selection to the newly created namespace
            st.session_state.selected_subnamespace = full_namespace
            st.session_state.search_mode = "subnamespace"
            st.session_state.subnamespace_created = True
            clear_chat()
            os.unlink(tmp_path)
            st.rerun()

    # Update existing subnamespace
    if uploaded_file and sub_selected not in ["Create New Subnamespace", "Search All Documents in Parent"] and st.sidebar.button("🔄 Update Selected Subnamespace"):
        if tmp_path and st.session_state.selected_subnamespace:
            texts = process_document(tmp_path, file_ext)
            full_namespace = st.session_state.selected_subnamespace
            count = store_embeddings(index, embedding_model, texts, full_namespace, update_existing=True)
            st.success(f"✅ Updated '{full_namespace}' with {count} chunks.")
            st.session_state.uploaded_files[uploaded_file.name] = full_namespace
            save_uploaded_files(st.session_state.uploaded_files)
            clear_chat()
            os.unlink(tmp_path)
            st.rerun()

    # Delete subnamespace
    if sub_selected not in ["Create New Subnamespace", "Search All Documents in Parent"] and st.sidebar.button("🗑️ Delete Selected Subnamespace", type="secondary"):
        if st.session_state.selected_subnamespace:
            try:
                namespace_to_delete = st.session_state.selected_subnamespace
                index.delete(delete_all=True, namespace=namespace_to_delete)
                st.success(f"✅ Deleted subnamespace '{namespace_to_delete}'")
                # Update the uploaded files record
                st.session_state.uploaded_files = {k: v for k, v in st.session_state.uploaded_files.items() 
                                                 if v != namespace_to_delete}
                save_uploaded_files(st.session_state.uploaded_files)
                
                # Reset selection to a valid option
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
    
    # Debug information (can be removed in production)
    with st.expander("Debug Info"):
        st.write(f"Selected Subnamespace: {st.session_state.selected_subnamespace}")
        st.write(f"Search Mode: {st.session_state.search_mode}")
        st.write(f"Available Subnamespaces: {subnamespaces}")
        st.write(f"Sub Options: {sub_options}")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.selected_parent:
            st.warning("⚠️ Please select or create a parent namespace first.")
        elif st.session_state.search_mode == "parent" and not subnamespaces:
            st.warning("⚠️ No documents found in this parent namespace. Please upload a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching..."):
                    # Determine which namespaces to search based on search mode
                    namespaces_to_search = []
                    
                    if st.session_state.search_mode == "parent":
                        # Search all subnamespaces under the parent
                        namespaces_to_search = subnamespaces
                        st.info(f"🔍 Searching across {len(namespaces_to_search)} documents...")
                    else:
                        # Search only the selected subnamespace
                        if st.session_state.selected_subnamespace and st.session_state.selected_subnamespace not in ["Create New Subnamespace", "Search All Documents in Parent"]:
                            namespaces_to_search = [st.session_state.selected_subnamespace]
                            st.info(f"🔍 Searching in single document...")
                    
                    all_results = []
                    for ns in namespaces_to_search:
                        results = search_documents(index, embedding_model, prompt, ns)
                        all_results.extend(results)
                    
                    # Sort by score and take top results
                    all_results.sort(key=lambda x: x["score"], reverse=True)
                    top_results = all_results[:10]  # Limit to top 10 results

                    if top_results:
                        context = "\n\n".join([f"[Score: {r['score']:.3f}] {r['text']}" for r in top_results])
                        answer = generate_response(groq_client, context, prompt, selected_model)
                        
                        # Display search results summary
                        st.write(f"**Found {len(top_results)} relevant chunks from {len(namespaces_to_search)} namespace(s)**")
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
    st.sidebar.write("**Uploaded Files & Namespaces:**")
    for name, ns in st.session_state.uploaded_files.items():
        st.sidebar.write(f"- {name} → `{ns}`")
    
    # Namespace statistics
    st.sidebar.markdown("---")
    st.sidebar.write("**Namespace Statistics:**")
    total_vectors = 0
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

# ==============================
# App Execution
# ==============================
if __name__ == "__main__":
    main()