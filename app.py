import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="INQUIRO PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
   
    [data-testid="stHeader"]{
        background-color: white !important;
        color: black !important;
    }
            
    [data-testid="stBaseButton-headerNoPadding"]{
        # background-color: white !important;
        color: black !important;
    }
            
      [data-testid="stMainBlockContainer"]{
        background-color: white !important;
        color: black !important;
    }      
            
    [data-testid="stMain"]{
        background-color: white !important;
        color: black !important;
    }      
            
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p
    {
         color:black !important;
    }
            
    [data-testid="stSidebarContent"]
    {
        background-color: #8d99ae !important;
    }
     
    [data-testid="stFileUploaderDropzone"]
    {
        background-color: rgba(38, 39, 48, 0.2) !important;
    }               
    [data-testid="stMain"] [data-testid="stAlertContainer"]
    {            
        background-color: #edf2f4 !important;
    }

    [data-testid="stFileUploaderFile"]
    {
        color: black !important;
    }
                    
   [data-testid="stAlertContainer"] p
    {            
        color: black !important;
        # font-weight: bold !important;
    }
    
    .main-header 
    {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .user-message {
        background-color: #edf2f4;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def initialize_components():
    """Initialize embedding model and Qdrant client"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        return embedding_model, qdrant_client
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and create vector store"""
    if not uploaded_files:
        return None
    
    with st.spinner("Processing uploaded PDFs..."):
        text = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                text.extend(pages)
                st.success(f"✅ Processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
        
        if not text:
            st.error("No text extracted from PDFs")
            return None
        
        embedding_model, qdrant_client = initialize_components()
        if not embedding_model or not qdrant_client:
            return None
        
        try:
            collection_name = f"RAG_PDF_{int(time.time())}"
            try:
                qdrant_client.delete_collection(collection_name)
            except:
                pass
            
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            vector_store = Qdrant.from_documents(
                documents=text,
                embedding=embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=collection_name,
            )
            
            st.success(f"✅ Successfully created vector store with {len(text)} document chunks")
            return vector_store
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

def get_rag_response(query, vector_store):
    """Get response from RAG system"""
    try:
        llm = Ollama(model="llama3.2:3b")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        with st.spinner("Generating response..."):
            response = rag_chain(query)
        
        return response["result"], response.get("source_documents", [])
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None, []

# Main app
def main():
    st.markdown('<h1 class="main-header">📚 INQUIRO PDF</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.subheader("📁 Upload Your PDF Documents")
        uploaded_files = st.file_uploader(
            "",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to create a knowledge base"
        )
        
        if uploaded_files and st.button("Process Documents", type="primary"):
            vector_store = process_uploaded_files(uploaded_files)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.documents_loaded = True
        
        if st.session_state.documents_loaded:
            st.success("✅ Documents processed and ready for queries!")
        else:
            st.info("📤 Upload and process PDF documents to start asking questions")
                
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("💭 Conversation History")
        
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                <div 
                    style="color: black;
                    font-weight: bold;"
                    class="chat-message user-message">
                    {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(message, unsafe_allow_html=False)
    if st.session_state.documents_loaded:    
            st.subheader("Ask Question")              
            query = st.text_input(
                "What would you like to inquire about?",
                placeholder="What insights are you looking for?",
                key="query_input"
            )
            
            ask_button = st.button("Ask", type="primary")
            
            if ask_button and query:
                if st.session_state.vector_store:
                    st.session_state.chat_history.append(("user", query))
                    response, source_docs = get_rag_response(query, st.session_state.vector_store)
                    
                    if response:
                        st.session_state.chat_history.append(("assistant", response))
                        st.rerun()
                else:
                    st.error("Please process documents first!")
    else:    
        st.markdown("""
        <div style="text-align: center; padding-bottom: 20px">
            <strong>An intelligent PDF question-answering system that transforms your documents into an interactive knowledge base.</strong>
        </div>
        """, unsafe_allow_html=True)
    
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🤖 AI-Powered
            Uses Ollama Llama 3.2 for intelligent text understanding and response generation
            """)
        
        with col2:
            st.markdown("""
            #### 🗃️ Vector Search  
            Powered by Qdrant vector database for fast and accurate document retrieval
            """)
        
        with col3:
            st.markdown("""
            #### 📄 Smart Extraction
            Automatically extracts and indexes content from your PDF documents
            """)
        
        # How it works section
        st.markdown("#### 🚀 How it works:")
        st.markdown("""
        1. **Upload** your PDF documents using the sidebar
        2. **Process** - AI creates searchable embeddings  
        3. **Ask** questions about your documents
        4. **Get** accurate, context-aware answers instantly
        """)
        st.info("Process documents first to enable querying")

if __name__ == "__main__":
    main()
