import streamlit as st
import os
import logging
from dotenv import dotenv_values
from main_system import EnhancedRAGSystem


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_rag_system(api_key, folder_id):
    return EnhancedRAGSystem(api_key, folder_id)

def main():
    st.set_page_config(
        page_title="Multi-Document RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    .user-message {
        background-color: #0288d1;
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #424242;
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        margin-right: 20%;
    }

    .stats-card {
        background-color: #424242;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    config = dotenv_values("csq_project.env")
    gemini_api_key = config.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not gemini_api_key:
        st.error("GOOGLE_API_KEY not found! Please check your environment variables.")
        st.stop()

    folder_id = config.get("FOLDER_ID") or os.getenv("FOLDER_ID")

    if not folder_id:
        st.error("FOLDER_ID not found! Please check your environment variables.")
        st.stop()

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Multi-Document RAG System</h1>
        <p>Advanced document processing with intelligent cross-document reasoning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing RAG System..."):
            st.session_state.rag_system = get_rag_system(gemini_api_key, folder_id)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Document Management")
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            if folder_id:
                with st.spinner("Processing documents..."):
                    success = st.session_state.rag_system.process_documents()
                    if success:
                        st.session_state.documents_processed = True
                        st.success("Documents processed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to process documents")
            else:
                st.error("Please provide a valid Google drive folder ID")
        
        # System statistics
        if st.session_state.documents_processed:
            stats = st.session_state.rag_system.get_system_stats()
            
            st.markdown("## System Statistics")
            st.markdown(f"""
            <div class="stats-card">
                <strong>Documents Processed:</strong> {stats['total_documents']}<br>
                <strong>Total Chunks:</strong> {stats['total_chunks']:,}<br>
                <strong>Status:</strong> {stats['collection_status']}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Document Breakdown"):
                for doc, chunks in stats['processed_documents'].items():
                    st.write(f"• {doc}: {chunks} chunks")
        
        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag_system.clear_memory()
            st.success("Chat history cleared!")
            st.rerun()
    
    # Main chat interface
    if not st.session_state.documents_processed:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #fff3cd; border-radius: 10px; margin: 2rem 0;">
            <h3 style="color: #856404;">No Documents Processed</h3>
            <p style="color: #856404;">Please process documents from the sidebar to start using the system.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f'<div class="user-message">{chat["user"]}</div>', unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f'<div class="assistant-message">{chat["assistant"]}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8f9ff 0%, #e8f2ff 100%); border-radius: 20px; margin: 2rem 0;">
            <h3 style="color: #667eea;">Ready to Answer Your Questions</h3>
            <p style="color: #666; font-size: 1.1rem;">Ask me anything about the processed documents. I can provide cross-document insights with proper source citations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question about the documents:",
            key="user_query",
            placeholder="What would you like to know about the documents?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle query submission
    if submit_button and user_query.strip():
        with st.spinner("Generating response..."):
            answer, sources = st.session_state.rag_system.generate_answer_with_citations(user_query)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "user": user_query,
                "assistant": answer,
                "sources": sources
            })
            
            st.rerun()

if __name__ == "__main__":
    main()