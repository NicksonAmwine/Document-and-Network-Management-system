# This script includes use of the google drive API to retrieve and manage documents, and adopts directly from CSQ_project_v2.py.

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

# Google Drive API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

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
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

@dataclass
class DocumentChunk:
    """Enhanced document chunk with source tracking."""
    content: str
    chunk_id: str
    chunk_index: int
    source_document: str
    document_type: str
    last_modified: str

@dataclass
class RetrievalResult:
    """Result from similarity search with source tracking."""
    content: str
    source_document: str
    relevance_score: float
    chunk_id: str

class DocumentHandler:
    """Handles document retrieval from drive and text extraction."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self.service = self.get_drive_service()

    def get_drive_service(self):
        """Authenticates with Google Drive API and returns the service object."""
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        
        try:
            return build("drive", "v3", credentials=creds)
        except HttpError as error:
            logger.error(f"An error occurred building the Drive service: {error}")
            return None
        
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
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
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from Word document."""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX content: {str(e)}")
            return ""

    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from text file."""
        try:
            return file_content.decode('utf-8', errors='replace').strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT content: {str(e)}")
            return ""

    def extract_text(self, file_content: bytes, file_type: str) -> str:
        """Extracts text from various document types."""
        if file_type == 'application/pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self.extract_text_from_docx(file_content)
        elif file_type == 'text/plain':
            return self.extract_text_from_txt(file_content)
        return ""
    
    def list_drive_files(self, service, folder_id: str, page_size=25):
        """Lists files and folders from Google Drive."""
        try:
            query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"

            results = (
                service.files()
                .list(
                    q=query,
                    pageSize=page_size,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)"
                ).execute()
            )
            items = results.get("files", [])
            if not items:
                logger.info(f"No files found in the provided folder.")
                return
            logger.info("--- Files found in folder ---")
            for item in items:
                logger.info(f"Name: {item['name']} (ID: {item['id']})")
            logger.info("----------------------------------------")
            return items
        except HttpError as error:
            logger.error(f"An error occurred listing files: {error}")

    
    def download_and_extract_text(self, service, file_id: str) -> str:
        """Downloads a file from Drive and extracts its text."""
        try:
            # Get file metadata to determine its type
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata.get('name')
            mime_type = file_metadata.get('mimeType')
            modified_time = file_metadata.get('modifiedTime')
            
            logger.info(f"Downloading '{file_name}' (MIME type: {mime_type})...")

            # Handle Google Docs, which need to be exported
            if mime_type.startswith('application/vnd.google-apps'):
                if mime_type == 'application/vnd.google-apps.document':
                    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                    file_content_type = '.txt'
                else:
                    logger.error(f"Unsupported Google App type: {mime_type}")
                    return ""
            # Handle standard files (PDF, DOCX, TXT)
            else:
                request = service.files().get_media(fileId=file_id)
                file_content_type = os.path.splitext(file_name)[1].lower()

            # Download the file content into memory
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}%.")
            
            file_content = fh.getvalue()
            
            # Extract text based on file type
            if file_content_type == '.pdf':
                return self.extract_text_from_pdf(file_content)
            elif file_content_type in ['.docx', '.doc']:
                return self.extract_text_from_docx(file_content)
            elif file_content_type == '.txt':
                return self.extract_text_from_txt(file_content)
            else:
                logger.warning(f"No text extractor for file type: {file_content_type}")
                return ""

        except HttpError as error:
            logger.error(f"An error occurred downloading/extracting file {file_id}: {error}")
            return ""

class MultiDocumentProcessor:
    """Handles processing of multiple document types."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}

    def __init__(self, folder_id: str):
        self.processed_documents = {}
        self.document_handler = DocumentHandler(folder_id=folder_id)

    def process_extracted_text(self, document_handler: DocumentHandler) -> List[DocumentChunk]:
        """Process all supported documents in a directory."""
        all_chunks = []
        service = document_handler.service
        retrieved_files = self.document_handler.list_drive_files(folder_id=document_handler.folder_id, service=service)

        if not retrieved_files:
            logger.error(f"No files found in the provided folder.")
            return all_chunks
        
        # Find all supported files recursively
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend([f for f in retrieved_files if f["name"].lower().endswith(ext)])

        logger.info(f"Found {len(supported_files)} supported documents")
        
        for file in supported_files:
            try:
                text = self.document_handler.download_and_extract_text(service, file_id=file['id'])
                if text.strip():
                    # Pass the dict
                    chunks = self.chunk_document(file, text)
                    all_chunks.extend(chunks)
                    self.processed_documents[file['name']] = len(chunks)
                    logger.info(f"Processed {file['name']}: {len(chunks)} chunks")
                else:
                    logger.warning(f"No text extracted from {file['name']}")
            except Exception as e:
                logger.error(f"Error processing {file['name']}: {str(e)}")
                continue
        
        return all_chunks
    
    def chunk_document(self, file, text: str) -> List[DocumentChunk]:
        """Split document into chunks with source tracking."""
        text_splitter = RecursiveTokenChunker(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!"],
        )
        
        chunks = text_splitter.split_text(text)
        
        # Get last modified timestamp of the source document
        last_modified = file['modifiedTime']
        source_document = file['name']
        document_type = file['mimeType']

        if isinstance(last_modified, (int, float)):
            modified_time = datetime.fromtimestamp(last_modified).isoformat()
        else:
            modified_time = str(last_modified)

        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_obj = DocumentChunk(
                content=chunk,
                chunk_id=chunk_id,
                chunk_index=i,
                source_document=source_document,
                document_type=document_type,
                last_modified=modified_time
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
                metadata={"created_at": datetime.now().isoformat()}
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

    def get_all_document_metadata(self) -> Dict[str, str]:
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
                        if source not in doc_timestamps or timestamp > doc_timestamps[source]:
                            doc_timestamps[source] = timestamp
            return doc_timestamps
        except Exception as e:
            logger.error(f"Could not fetch document metadata: {e}")
            return {}


    def similarity_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Search for similar documents with enhanced results."""
        try:
            if not self.collection:
                logger.warning("Attempted to search an empty collection.")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            documents = results['documents'][0] if results.get('documents') and results['documents'] else []
            metadatas = results['metadatas'][0] if results.get('metadatas') and results['metadatas'] else []
            distances = results['distances'][0] if results.get('distances') and results['distances'] else []
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                similarity = 1 - distance if distance <= 1 else 1 / (1 + distance)  # fallback for other metrics
                retrieval_result = RetrievalResult(
                    content=doc,
                    source_document=str(metadata.get('source_document', 'Unknown')),
                    relevance_score=similarity,
                    chunk_id=str(metadata.get('chunk_id', 'Unknown'))
                )
                retrieval_results.append(retrieval_result)
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

class EnhancedRAGSystem:
    """Complete RAG system with multi-document support and source citations."""
    
    def __init__(self, gemini_api_key: str, folder_id: str):
        self.gemini_api_key = gemini_api_key
        self.folder_id = folder_id

        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr(gemini_api_key)
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=SecretStr(gemini_api_key)
        )
        
        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path.cwd()
        db_path = base_dir / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.vector_store = EnhancedVectorStore(self.embeddings, self.chroma_client)
        self.document_processor = MultiDocumentProcessor(self.folder_id)
        

        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )

    def process_documents(self) -> bool:
        """Process all documents from directory and update vector store."""
        try:
            existing_docs = self.vector_store.get_all_document_metadata()

            # 2. Process all files and get their chunks (the processor is now stateless)
            all_chunks = self.document_processor.process_extracted_text(self.document_processor.document_handler)
            
            # 3. Determine which documents are new or have been updated
            chunks_to_add = []
            files_to_update = set()

            for chunk in all_chunks:
                source_doc = chunk.source_document
                last_modified_in_db = existing_docs.get(source_doc)
                
                if last_modified_in_db is None or chunk.last_modified > last_modified_in_db:
                    chunks_to_add.append(chunk)
                    files_to_update.add(source_doc)

            if not chunks_to_add:
                logger.info("No new or updated documents to process.")
                return True

            # 4. Delete old chunks for any files that were updated
            for file_name in files_to_update:
                self.vector_store.delete_documents_by_source(file_name)
            
            # Add to vector store
            success = self.vector_store.add_documents(chunks_to_add)   #fix error where documents are being added regardless of them already existing.

            if success:
                logger.info(f"Successfully processed {len(chunks_to_add)} chunks from {len(files_to_update)} documents")
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
            retrieval_results = self.vector_store.similarity_search(query, k=12)  
            
            if not retrieval_results:
                return "I couldn't find relevant information in the document collection to answer your question.", []
            
            # Prepare context and source tracking
            source_map = {f"source_{i+1}": result for i, result in enumerate(retrieval_results)}
            
            context_parts = []
            for marker, result in source_map.items():
                context_parts.append(f"[{marker} from {result.source_document}]:\n{result.content}")
            
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
            2.  Use the following provided context to provide a detailed and accurate answer. Be sure to keep your responses relevant to the question.
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
                sources_list = [result.source_document for result in source_map.values()]
            else:
                # Build the footnote section
                footnote_section = ["\n\n---", "**Sources:**"]
                citation_map = {}
                final_sources_list = []
                for i, marker in enumerate(sorted(list(used_sources))):
                    citation_num = i + 1
                    citation_map[marker] = citation_num
                    
                    source_doc_name = source_map[marker].source_document
                    if source_doc_name not in final_sources_list:
                        final_sources_list.append(source_doc_name)

                    footnote_section.append(f"[{citation_num}] {source_doc_name}")
                
                # Replace source markers with numbered citations in the text
                final_answer = answer_text
                for marker, num in citation_map.items():
                    final_answer = final_answer.replace(f'[{marker}]', f' [{num}]')
                
                final_answer += "\n" + "\n".join(footnote_section)
                sources_list = sorted(final_sources_list)

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

@st.cache_resource
def get_rag_system(api_key, folder_id):
    return EnhancedRAGSystem(api_key, folder_id)

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
/*    .source-citation {
        background-color: #f5f5f5;
        border-left: 3px solid #2196f3;
        padding: 8px;
        margin: 5px 0;
        font-size: 0.9em;
        border-radius: 5px;
    } */
    .stats-card {
        background-color: #424242;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load configuration
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
            
            # Source citations
            # if chat.get("sources"):
            #     sources_text = ", ".join(chat["sources"])
            #     st.markdown(f'<div class="source-citation"><strong>Sources:</strong> {sources_text}</div>', unsafe_allow_html=True)
            
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