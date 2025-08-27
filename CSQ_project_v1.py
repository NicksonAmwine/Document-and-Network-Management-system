import re
import streamlit as st
import os
import io
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from chromadb.types import Metadata
from pathlib import Path
import logging
from datetime import datetime

# Core libraries
import PyPDF2
import docx
import chromadb
from dotenv import dotenv_values

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from chunking_evaluation.chunking import RecursiveTokenChunker
from pydantic import SecretStr
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Enhanced document chunk with source tracking."""
    content: str
    chunk_id: str
    chunk_index: int
    source_document: str
    document_type: str
    last_modified: float

@dataclass
class RetrievalResult:
    """Result from similarity search with source tracking."""
    content: str
    source_document: str
    relevance_score: float
    chunk_id: str

class MultiDocumentProcessor:
    """Handles processing of multiple document types."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}
    
    def __init__(self):
        self.processed_documents = {}
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                        continue
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text based on file extension."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return ""
    
    def process_documents_from_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Process all supported documents in a directory."""
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return all_chunks
        
        # Find all supported files recursively
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend(directory.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(supported_files)} supported documents")
        
        for file_path in supported_files:
            try:
                text = self.extract_text(str(file_path))
                if text.strip():
                    chunks = self.chunk_document(
                        text, 
                        source_document=file_path.name,
                        document_type=file_path.suffix.lower()
                    )
                    all_chunks.extend(chunks)
                    self.processed_documents[file_path.name] = len(chunks)
                    logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                else:
                    logger.warning(f"No text extracted from {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        return all_chunks
    
    def chunk_document(self, text: str, source_document: str, document_type: str) -> List[DocumentChunk]:
        """Split document into chunks with source tracking."""
        text_splitter = RecursiveTokenChunker(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!"],
        )
        
        chunks = text_splitter.split_text(text)
        
        # Get last modified timestamp of the source document
        try:
            source_path = Path(source_document)
            if not source_path.is_absolute():
                # Try to find the file in the current directory or as a relative path
                source_path = Path.cwd() / source_document
            last_modified = source_path.stat().st_mtime
        except Exception:
            last_modified = datetime.now().timestamp()
        
        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_obj = DocumentChunk(
                content=chunk,
                chunk_id=chunk_id,
                chunk_index=i,
                source_document=source_document,
                document_type=document_type,
                last_modified=last_modified
            )
            document_chunks.append(chunk_obj)
        
        return document_chunks

class EnhancedVectorStore:
    """Enhanced vector store with source tracking and better retrieval."""
    
    def __init__(self, embeddings, chroma_client, collection_name="multi_document_collection"):
        self.embeddings = embeddings
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.collection = self.create_or_get_collection()
        
    def create_or_get_collection(self) -> chromadb.Collection:
        """Create or get the document collection."""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}' with {collection.count()} documents")
            return collection
        except Exception:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"created_at": str(datetime.now())}
            )
            logger.info(f"Created new collection: '{self.collection_name}'")
            return collection
    
    def add_documents(self, document_chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store without clearing existing data."""
        try:
            if not document_chunks:
                logger.warning("No document chunks to add")
                return False
            
            texts = [chunk.content for chunk in document_chunks]
            ids = [chunk.chunk_id for chunk in document_chunks]

            metadatas: List[Metadata] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "source_document": chunk.source_document,
                    "document_type": chunk.document_type,
                    "last_modified": chunk.last_modified 
                }
                for chunk in document_chunks
            ]
            
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Added/updated {len(document_chunks)} chunks in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def delete_documents_by_source(self, source_document: str):
        """Delete all chunks associated with a specific source document."""
        try:
            self.collection.delete(where={"source_document": source_document})
            logger.info(f"Deleted all chunks for source: {source_document}")
        except Exception as e:
            logger.error(f"Error deleting chunks for {source_document}: {e}")

    def get_all_document_metadata(self) -> Dict[str, float]:
        """Fetches all unique documents and their last modified timestamps."""
        try:
            # Fetch a representative set of metadata. ChromaDB doesn't have a "distinct" query.
            # We fetch all metadata and process it in memory.
            all_items = self.collection.get(include=["metadatas"])
            
            doc_timestamps = {}
            metadatas = all_items.get('metadatas')
            if metadatas is not None:
                for metadata in metadatas:
                    source = metadata.get('source_document')
                    timestamp = metadata.get('last_modified')
                    if source:
                        # Keep the latest timestamp if there are multiple entries
                        if source not in doc_timestamps or timestamp > doc_timestamps[source]:
                            doc_timestamps[source] = timestamp
            return doc_timestamps
        except Exception as e:
            logger.error(f"Could not fetch document metadata: {e}")
            return {}


    # def similarity_search(self, query: str, k: int = 8) -> List[RetrievalResult]:
    #     """Search for similar documents with enhanced results."""
    #     try:
    #         if not self.collection:
    #             return []
            
    #         # Generate query embedding
    #         query_embedding = self.embeddings.embed_query(query)
            
    #         # Search in ChromaDB
    #         results = self.collection.query(
    #             query_embeddings=[query_embedding],
    #             n_results=k,
    #             include=['documents', 'metadatas', 'distances']
    #         )
            
    #         if not results['documents'] or not results['documents'][0]:
    #             return []
            
    #         # Convert to RetrievalResult objects
    #         retrieval_results = []
    #         documents = results['documents'][0]
    #         metadatas = results['metadatas'][0]
    #         distances = results['distances'][0]
            
    #         for doc, metadata, distance in zip(documents, metadatas, distances):
    #             retrieval_result = RetrievalResult(
    #                 content=doc,
    #                 source_document=metadata.get('source_document', 'Unknown'),
    #                 relevance_score=1.0 - distance,  # Convert distance to similarity score
    #                 chunk_id=metadata.get('chunk_id', 'Unknown')
    #             )
    #             retrieval_results.append(retrieval_result)
            
    #         return retrieval_results
            
    #     except Exception as e:
    #         logger.error(f"Error searching documents: {str(e)}")
    #         return []

class EnhancedRAGSystem:
    """Complete RAG system with multi-document support and source citations."""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        
        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr(gemini_api_key)
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=SecretStr(gemini_api_key)
        )
        
        db_path = Path(__file__).parent / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.vector_store = EnhancedVectorStore(self.embeddings, self.chroma_client)
        self.document_processor = MultiDocumentProcessor()
        
        self.lc_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.vector_store.collection_name,
            embedding_function=self.embeddings
        )

        self.retriever = self.lc_vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 12}
        )

        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )
        
    def process_documents(self, documents_directory: str) -> bool:
        """Process all documents from directory and update vector store."""
        try:
            # Process documents
            document_chunks = self.document_processor.process_documents_from_directory(
                documents_directory
            )
            
            if not document_chunks:
                logger.error("No documents were processed successfully")
                return False
            
            # Add to vector store
            success = self.vector_store.add_documents(document_chunks)
            
            if success:
                logger.info(f"Successfully processed {len(document_chunks)} chunks from {len(self.document_processor.processed_documents)} documents")
                return True
            else:
                logger.error("Failed to add documents to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def generate_answer_with_citations(self, query: str) -> Tuple[str, List[str]]:
        """Generate answer with source citations."""
        try:
            # Retrieve relevant documents
            retrieval_results = self.retriever.invoke(query)
            
            if not retrieval_results:
                return "I couldn't find relevant information in the document collection to answer your question.", []
            
            # Prepare context and source tracking
            unique_sources = {doc.metadata.get('source_document', 'Unknown'): doc for doc in retrieval_results}
            source_map = {f"source_{i+1}": doc for i, doc in enumerate(unique_sources.values())}
            
            context_parts = []
            for marker, doc in source_map.items():
                context_parts.append(f"[{marker}]:\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Get conversation history
            try:
                memory_vars = self.memory.load_memory_variables({})
                history = memory_vars.get("chat_history", [])
                
                history_str = ""
                if history:
                    formatted_messages = []
                    for msg in history:
                        if hasattr(msg, 'type'):
                            if msg.type == 'human':
                                formatted_messages.append(f"Human: {msg.content}")
                            elif msg.type == 'ai':
                                formatted_messages.append(f"Assistant: {msg.content}")
                    
                    history_str = "\n".join(formatted_messages[-6:])  # Last 3 exchanges
            except:
                history_str = ""
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            You are an expert AI assistant. Your task is to answer the user's question based *only* on the provided context from the document collection.

            **Strict Instructions:**
            1.  Analyze the provided "Context from Documents" thoroughly.
            2.  Use the following context from the uploaded documents to provide a detailed and accurate answer. Be sure to keep your responses relevant to the question.
            3.  You can derive deductions from the provided context to answer questions but be sure to stay within the context of the document. Make the conversation flow naturally and try not to sound like a robot. For example, don't say "Based on the provided documents, ...". or "The text suggests that ...". Instead, just answer the question naturally.
            4.  When you use information from a specific passage, cite it by placing its corresponding source tag (e.g., `[source_1]`) immediately after the statement.
            5.  You can use multiple citations for a single sentence if the information is synthesized from multiple sources.
            6.  If the context does not contain the information needed to answer the question, you **MUST** state: "I could not find sufficient information in the provided documents to answer this question." Do not use outside knowledge or make assumptions.

            **Conversation History:**
            {history}

            **Context from Documents:**
            {context}

            **Question:** {question}

            **Answer (with citations):**
            """)
            
            # Generate response
            prompt = prompt_template.format(
                context=context,
                question=query,
                history=history_str
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer_text = response.content if isinstance(response.content, str) else str(response.content)
            
            used_sources = set(re.findall(r'\[(source_\d+)\]', answer_text))
            
            if not used_sources:
                # If the model didn't cite, just return the answer and a generic source list
                final_answer = answer_text
                sources_list = [str(doc.metadata.get('source_document')) for doc in source_map.values() if doc.metadata.get('source_document') is not None]
            else:
                # Build the footnote section
                footnote_section = ["\n\n---", "**Sources:**"]
                citation_map = {}
                for i, marker in enumerate(sorted(list(used_sources))):
                    citation_num = i + 1
                    citation_map[marker] = citation_num
                    
                    source_doc_name = source_map[marker].metadata.get('source_document', 'Unknown')
                    footnote_section.append(f"[{citation_num}] {source_doc_name}")
                
                # Replace source markers with numbered citations in the text
                final_answer = answer_text
                for marker, num in citation_map.items():
                    final_answer = final_answer.replace(f'[{marker}]', f' [{num}]')
                
                final_answer += "\n" + "\n".join(footnote_section)
                sources_list = [str(source_map[marker].metadata.get('source_document')) for marker in sorted(list(used_sources)) if source_map[marker].metadata.get('source_document') is not None]

            self.memory.save_context({"input": query}, {"output": answer_text})
            
            return final_answer, sources_list
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}", []
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        stats = {
            "total_documents": len(self.document_processor.processed_documents),
            "processed_documents": self.document_processor.processed_documents,
            "total_chunks": self.vector_store.collection.count() if self.vector_store.collection else 0,
            "collection_status": "Ready" if self.vector_store.collection else "Not Initialized"
        }
        return stats

# Streamlit Application

st.cache_resource
def get_rag_system(api_key):
    return EnhancedRAGSystem(api_key)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Multi-Document RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    .source-citation {
        background-color: #f5f5f5;
        border-left: 3px solid #2196f3;
        padding: 8px;
        margin: 5px 0;
        font-size: 0.9em;
        border-radius: 5px;
    }
    .stats-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = dotenv_values("chatbot.env")
    gemini_api_key = config.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not gemini_api_key:
        st.error("GOOGLE_API_KEY not found! Please check your chatbot.env file.")
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
            st.session_state.rag_system = get_rag_system(gemini_api_key)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Document Management")
        
        # Documents directory input
        documents_dir = st.text_input(
            "Documents Directory Path:",
            value="D:\\data\\My docs\\my_projects\\timepledge_projects\\AFTA_chatbot\\uploaded_documents",
            help="Path to directory containing documents to process"
        )
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            if documents_dir and os.path.exists(documents_dir):
                with st.spinner("Processing documents..."):
                    success = st.session_state.rag_system.process_documents(documents_dir)
                    if success:
                        st.session_state.documents_processed = True
                        st.success("Documents processed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to process documents")
            else:
                st.error("Please provide a valid directory path")
        
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
            
            # Source citations
            if chat.get("sources"):
                sources_text = ", ".join(chat["sources"])
                st.markdown(f'<div class="source-citation"><strong>Sources:</strong> {sources_text}</div>', unsafe_allow_html=True)
            
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

    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    
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
            st.session_state.input_counter += 1
            st.session_state.user_query = ""  # Clear input box
            st.rerun()

if __name__ == "__main__":
    main()