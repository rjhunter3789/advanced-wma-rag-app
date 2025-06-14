import streamlit as st
import tempfile
import os
import json
import time
from pathlib import Path
import PyPDF2
from io import BytesIO
from datetime import datetime
import uuid

# Advanced text search using TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    st.error("Please install required packages: scikit-learn")

# Optional: OpenAI for better responses
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Advanced WMA RAG App",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .folder-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .document-item {
        padding: 0.3rem 0.5rem;
        margin: 0.2rem 0;
        border-radius: 3px;
        background-color: #e9ecef;
        border-left: 3px solid #28a745;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .search-scope {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏢 Advanced WMA RAG App</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise Document Management with Intelligent Q&A</p>', unsafe_allow_html=True)

# Initialize session state for advanced features
def init_session_state():
    """Initialize all session state variables"""
    if 'folder_structure' not in st.session_state:
        st.session_state.folder_structure = {
            "root": {
                "type": "folder",
                "name": "Documents",
                "children": {},
                "created": datetime.now().isoformat()
            }
        }
    
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    
    if 'vectorizer' not in st.session_state:
        if SEARCH_AVAILABLE:
            st.session_state.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        else:
            st.session_state.vectorizer = None
    
    if 'document_vectors' not in st.session_state:
        st.session_state.document_vectors = {}
    
    if 'current_folder' not in st.session_state:
        st.session_state.current_folder = "root"
    
    if 'selected_search_scope' not in st.session_state:
        st.session_state.selected_search_scope = "all"
    
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []

init_session_state()

def log_activity(action, details=""):
    """Log user activities"""
    st.session_state.activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    })
    # Keep only last 100 activities
    if len(st.session_state.activity_log) > 100:
        st.session_state.activity_log = st.session_state.activity_log[-100:]

def get_folder_path(folder_id):
    """Get the full path of a folder"""
    def find_path(structure, target_id, current_path=""):
        for key, value in structure.items():
            if key == target_id:
                return current_path + "/" + value["name"] if current_path else value["name"]
            if value.get("type") == "folder" and "children" in value:
                result = find_path(value["children"], target_id, 
                                 current_path + "/" + value["name"] if current_path else value["name"])
                if result:
                    return result
        return None
    
    return find_path(st.session_state.folder_structure, folder_id)

def create_folder(parent_id, folder_name):
    """Create a new folder"""
    if not folder_name.strip():
        return False, "Folder name cannot be empty"
    
    folder_id = str(uuid.uuid4())
    
    def add_to_structure(structure, target_id):
        for key, value in structure.items():
            if key == target_id:
                if "children" not in value:
                    value["children"] = {}
                value["children"][folder_id] = {
                    "type": "folder",
                    "name": folder_name.strip(),
                    "children": {},
                    "created": datetime.now().isoformat()
                }
                return True
            if value.get("type") == "folder" and "children" in value:
                if add_to_structure(value["children"], target_id):
                    return True
        return False
    
    success = add_to_structure(st.session_state.folder_structure, parent_id)
    if success:
        log_activity("Created folder", f"'{folder_name}' in {get_folder_path(parent_id)}")
    return success, "Folder created successfully" if success else "Parent folder not found"

def delete_folder(folder_id):
    """Delete a folder and all its contents"""
    if folder_id == "root":
        return False, "Cannot delete root folder"
    
    def remove_from_structure(structure, target_id):
        for key, value in structure.items():
            if value.get("type") == "folder" and "children" in value:
                if target_id in value["children"]:
                    folder_name = value["children"][target_id]["name"]
                    del value["children"][target_id]
                    return True, folder_name
                success, name = remove_from_structure(value["children"], target_id)
                if success:
                    return success, name
        return False, None
    
    success, folder_name = remove_from_structure(st.session_state.folder_structure, folder_id)
    if success:
        # Remove all documents in this folder
        docs_to_remove = [doc_id for doc_id, doc in st.session_state.documents.items() 
                         if doc.get("folder_id") == folder_id]
        for doc_id in docs_to_remove:
            del st.session_state.documents[doc_id]
            if doc_id in st.session_state.document_vectors:
                del st.session_state.document_vectors[doc_id]
        
        log_activity("Deleted folder", f"'{folder_name}' and {len(docs_to_remove)} documents")
    
    return success, f"Deleted '{folder_name}'" if success else "Folder not found"

def add_document_to_folder(folder_id, file_name, content, file_type="pdf"):
    """Add a document to a specific folder"""
    doc_id = str(uuid.uuid4())
    
    # Process text into chunks
    chunks = chunk_text(content)
    
    # Create document vectors
    if SEARCH_AVAILABLE and chunks:
        try:
            # Create a separate vectorizer for this document
            doc_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            doc_vectors = doc_vectorizer.fit_transform(chunks)
            st.session_state.document_vectors[doc_id] = {
                "vectorizer": doc_vectorizer,
                "vectors": doc_vectors,
                "chunks": chunks
            }
        except Exception as e:
            st.warning(f"Could not create vectors for {file_name}: {str(e)}")
    
    # Store document metadata
    st.session_state.documents[doc_id] = {
        "name": file_name,
        "folder_id": folder_id,
        "content": content,
        "chunks": chunks,
        "type": file_type,
        "uploaded": datetime.now().isoformat(),
        "size": len(content)
    }
    
    log_activity("Uploaded document", f"'{file_name}' to {get_folder_path(folder_id)}")
    return doc_id

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def search_documents(query, scope="all", top_k=5):
    """Search documents with specified scope"""
    if not query.strip():
        return []
    
    relevant_docs = []
    
    # Determine which documents to search
    if scope == "all":
        search_docs = st.session_state.documents
    else:
        # Search only in specific folder
        search_docs = {doc_id: doc for doc_id, doc in st.session_state.documents.items() 
                      if doc.get("folder_id") == scope}
    
    # Search each document
    for doc_id, doc in search_docs.items():
        if doc_id in st.session_state.document_vectors:
            try:
                vec_data = st.session_state.document_vectors[doc_id]
                query_vec = vec_data["vectorizer"].transform([query])
                similarities = cosine_similarity(query_vec, vec_data["vectors"])[0]
                
                # Get top chunks from this document
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Minimum similarity threshold
                        relevant_docs.append({
                            "doc_id": doc_id,
                            "doc_name": doc["name"],
                            "folder_path": get_folder_path(doc["folder_id"]),
                            "chunk": vec_data["chunks"][idx],
                            "similarity": similarities[idx],
                            "chunk_index": idx
                        })
            except Exception as e:
                st.warning(f"Error searching {doc['name']}: {str(e)}")
    
    # Sort by similarity and return top results
    relevant_docs.sort(key=lambda x: x["similarity"], reverse=True)
    return relevant_docs[:top_k]

def generate_answer(query, search_results):
    """Generate answer using OpenAI or simple concatenation"""
    if not search_results:
        return "No relevant information found in the selected documents/folders."
    
    # Combine relevant chunks
    context = "\n\n".join([f"From {result['doc_name']}: {result['chunk']}" 
                          for result in search_results])
    
    # Check if OpenAI is available and configured
    openai_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') else os.getenv("OPENAI_API_KEY")
    
    if OPENAI_AVAILABLE and openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided document context. If the answer isn't in the context, say so. Always mention which documents your answer comes from."},
                    {"role": "user", "content": f"Context from documents:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"OpenAI error: {str(e)}. Using simple retrieval.")
    
    # Fallback: Simple context return
    return f"Based on the documents, here's what I found:\n\n{context[:1000]}{'...' if len(context) > 1000 else ''}"

def display_folder_tree(structure, parent_key="", level=0):
    """Display folder tree with management options"""
    for key, value in structure.items():
        if value.get("type") == "folder":
            indent = "  " * level
            folder_name = value["name"]
            
            # Folder display with actions
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                if st.button(f"📁 {indent}{folder_name}", key=f"folder_{key}"):
                    st.session_state.current_folder = key
                    st.rerun()
            
            with col2:
                if st.button("➕", key=f"add_{key}", help="Add subfolder"):
                    st.session_state.show_add_folder = key
            
            with col3:
                if st.button("📄", key=f"upload_{key}", help="Upload to this folder"):
                    st.session_state.upload_to_folder = key
            
            with col4:
                if key != "root" and st.button("🗑️", key=f"delete_{key}", help="Delete folder"):
                    success, message = delete_folder(key)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            # Show documents in this folder
            folder_docs = [doc for doc_id, doc in st.session_state.documents.items() 
                          if doc.get("folder_id") == key]
            
            if folder_docs:
                for doc in folder_docs:
                    st.markdown(f'{indent}  📄 {doc["name"]} ({len(doc.get("chunks", []))} chunks)')
            
            # Recursively display subfolders
            if "children" in value and value["children"]:
                display_folder_tree(value["children"], key, level + 1)

# Sidebar for folder management and configuration
with st.sidebar:
    st.header("📁 Folder Management")
    
    # Current folder info
    current_path = get_folder_path(st.session_state.current_folder)
    st.info(f"📍 Current: {current_path}")
    
    # Add new folder
    if st.button("Create New Folder"):
        st.session_state.show_add_folder = st.session_state.current_folder
    
    if st.session_state.get("show_add_folder"):
        folder_name = st.text_input("Folder Name:", key="new_folder_name")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create"):
                if folder_name:
                    success, message = create_folder(st.session_state.show_add_folder, folder_name)
                    if success:
                        st.success(message)
                        del st.session_state.show_add_folder
                        st.rerun()
                    else:
                        st.error(message)
        with col2:
            if st.button("Cancel"):
                del st.session_state.show_add_folder
                st.rerun()
    
    st.markdown("---")
    
    # Folder tree
    st.subheader("📂 Folder Structure")
    display_folder_tree(st.session_state.folder_structure)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    total_docs = len(st.session_state.documents)
    total_folders = sum(1 for _ in st.session_state.folder_structure.get("root", {}).get("children", {}))
    
    st.metric("Total Documents", total_docs)
    st.metric("Total Folders", total_folders)
    
    # OpenAI Configuration
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key (optional)", type="password", help="For enhanced responses")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Document Upload")
    
    # Folder selection for upload
    folder_options = {"root": "Documents (Root)"}
    
    def collect_folders(structure, prefix=""):
        for key, value in structure.items():
            if value.get("type") == "folder":
                folder_options[key] = f"{prefix}{value['name']}"
                if "children" in value:
                    collect_folders(value["children"], f"{prefix}{value['name']}/")
    
    collect_folders(st.session_state.folder_structure["root"].get("children", {}), "Documents/")
    
    selected_folder = st.selectbox("Upload to folder:", 
                                  options=list(folder_options.keys()),
                                  format_func=lambda x: folder_options[x],
                                  index=0)
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF files to the selected folder"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            if not SEARCH_AVAILABLE:
                st.error("Text search not available. Please check requirements.")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                
                # Extract text
                text = extract_text_from_pdf(uploaded_file)
                if text.strip():
                    # Add to selected folder
                    doc_id = add_document_to_folder(selected_folder, uploaded_file.name, text, "pdf")
                    st.success(f"✅ {uploaded_file.name} processed")
                else:
                    st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            time.sleep(1)
            st.rerun()

with col2:
    st.header("🔍 Intelligent Search & Q&A")
    
    if st.session_state.documents:
        # Search scope selection
        scope_options = {"all": "🌐 All Documents"}
        collect_folders(st.session_state.folder_structure, "")
        for key, name in folder_options.items():
            if key != "all":
                doc_count = len([d for d in st.session_state.documents.values() if d.get("folder_id") == key])
                if doc_count > 0:
                    scope_options[key] = f"📁 {name} ({doc_count} docs)"
        
        search_scope = st.selectbox("Search scope:", 
                                   options=list(scope_options.keys()),
                                   format_func=lambda x: scope_options[x])
        
        # Display current search scope
        if search_scope != "all":
            st.markdown(f'<div class="search-scope">🎯 Searching in: {folder_options[search_scope]}</div>', 
                       unsafe_allow_html=True)
        
        # Question input
        question = st.text_input("Enter your question:", placeholder="What information do you need?")
        
        col_ask, col_summarize = st.columns(2)
        
        with col_ask:
            ask_button = st.button("🔍 Search & Answer", type="primary")
        
        with col_summarize:
            summarize_button = st.button("📝 Summarize Scope")
        
        if ask_button and question:
            with st.spinner("Searching documents..."):
                # Search documents
                search_results = search_documents(question, search_scope, top_k=5)
                
                if search_results:
                    # Generate answer
                    answer = generate_answer(question, search_results)
                    
                    st.subheader("💡 Answer")
                    st.write(answer)
                    
                    # Show sources
                    with st.expander("📄 Sources", expanded=False):
                        for i, result in enumerate(search_results):
                            st.write(f"**Source {i+1}:** {result['doc_name']}")
                            st.write(f"**Folder:** {result['folder_path']}")
                            st.write(f"**Similarity:** {result['similarity']:.3f}")
                            st.write(f"**Content:** {result['chunk'][:300]}...")
                            st.write("---")
                    
                    log_activity("Asked question", f"'{question}' in scope '{scope_options[search_scope]}'")
                else:
                    st.warning("No relevant information found for your question in the selected scope.")
        
        if summarize_button:
            with st.spinner("Creating summary..."):
                # Get all documents in scope
                if search_scope == "all":
                    docs_to_summarize = list(st.session_state.documents.values())
                else:
                    docs_to_summarize = [doc for doc in st.session_state.documents.values() 
                                       if doc.get("folder_id") == search_scope]
                
                if docs_to_summarize:
                    # Create a summary of all documents
                    combined_text = " ".join([doc["content"][:500] for doc in docs_to_summarize[:5]])  # Limit for performance
                    
                    st.subheader("📋 Summary")
                    st.info(f"Summary of {len(docs_to_summarize)} documents in {scope_options[search_scope]}")
                    
                    # Simple summary (first few sentences from each document)
                    summary_parts = []
                    for doc in docs_to_summarize[:3]:  # Limit to first 3 documents
                        sentences = doc["content"].split('. ')[:2]  # First 2 sentences
                        summary_parts.append(f"**{doc['name']}:** {'. '.join(sentences)}")
                    
                    st.write("\n\n".join(summary_parts))
                    
                    log_activity("Generated summary", f"For scope '{scope_options[search_scope]}'")
                else:
                    st.warning("No documents found in the selected scope.")
    
    else:
        st.info("👆 Upload PDF files first to start searching and asking questions!")

# Activity log (collapsible)
with st.expander("📋 Recent Activity", expanded=False):
    if st.session_state.activity_log:
        for activity in reversed(st.session_state.activity_log[-10:]):  # Show last 10 activities
            timestamp = datetime.fromisoformat(activity["timestamp"]).strftime("%H:%M:%S")
            st.write(f"**{timestamp}** - {activity['action']}: {activity['details']}")
    else:
        st.write("No recent activity")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>Advanced WMA RAG App</strong> - Enterprise Document Management & Intelligent Q&A System 🚀<br>
    Built with Streamlit • Powered by TF-IDF & OpenAI
</div>
""", unsafe_allow_html=True)
