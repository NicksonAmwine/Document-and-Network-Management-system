# Enhanced RAG System integrating Ollama and advanced retrieval strategies
import re
import streamlit as st
import os
import io
import uuid
import numpy as np
import requests
import torch
from typing import List, Dict, Optional, Tuple, cast, Union
from dataclasses import dataclass
from chromadb.types import Metadata
from pathlib import Path
from pydantic import BaseModel, Field
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
from langchain_core.output_parsers import PydanticOutputParser

# LlamaIndex imports for advanced retrieval
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
    RouterRetriever
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, IndexNode
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Configuration Classes
class ModelSettings(BaseModel):
    """Settings for language models."""
    ollama_model: str = Field(default="llama3:8b-instruct-q8_0", description="Ollama model name")
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", description="HuggingFace embedding model")
    rerank_model: str = Field(default="BAAI/bge-reranker-large", description="Rerank model")
    use_ollama: bool = Field(default=True, description="Use Ollama for LLM")
    use_hf_embeddings: bool = Field(default=True, description="Use HuggingFace embeddings")
    ollama_host: str = Field(default="localhost", description="Ollama host")
    ollama_port: int = Field(default=11434, description="Ollama port")
    temperature: float = Field(default=0.1, description="Model temperature")
    context_window: int = Field(default=8000, description="Context window size")

class RetrieverSettings(BaseModel):
    """Settings for retrieval strategies."""
    similarity_top_k: int = Field(default=20, description="Top k documents for similarity search")
    num_queries: int = Field(default=5, description="Number of generated queries for fusion")
    retriever_weights: List[float] = Field(default=[0.4, 0.6], description="BM25 vs Vector weights")
    top_k_rerank: int = Field(default=6, description="Top k after reranking")
    fusion_mode: str = Field(default="dist_based_score", description="Fusion mode")
    use_hybrid_retrieval: bool = Field(default=True, description="Use hybrid BM25+Vector retrieval")
    use_query_expansion: bool = Field(default=True, description="Use query expansion")
    rerank_model: str = Field(default="BAAI/bge-reranker-large", description="Rerank model")

class QueryExpansion(BaseModel):
    queries: List[str]

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

class OllamaModelManager:
    """Manages Ollama models and connections."""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return any(model["name"] == model_name for model in models)
            return False
        except requests.RequestException:
            return False
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model in Ollama."""
        try:
            payload = {"name": model_name}
            response = requests.post(f"{self.base_url}/api/pull", json=payload, stream=True)
            
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    logger.info(f"Pulling {model_name}: {data}")
            
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

class HuggingFaceEmbeddingManager:
    """Manages HuggingFace embedding models."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", cache_folder: str = "data/huggingface"):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._embedding_model = None
    
    def get_embedding_model(self) -> HuggingFaceEmbedding:
        """Get HuggingFace embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = HuggingFaceEmbedding(
                    model=AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        cache_dir=self.cache_folder
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        cache_dir=self.cache_folder
                    ),
                    trust_remote_code=True,
                    embed_batch_size=8
                )
                logger.info(f"Loaded HuggingFace embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading HuggingFace embedding model: {e}")
                raise
        
        return self._embedding_model

class AdvancedRetriever:
    """Advanced retrieval system with multiple strategies."""
    
    def __init__(self, settings: RetrieverSettings):
        self.settings = settings
        self.rerank_model = SentenceTransformerRerank(
            top_n=settings.top_k_rerank,
            model=settings.rerank_model,
        ) if settings.top_k_rerank > 0 else None
    
    def create_hybrid_retriever(self, vector_index: VectorStoreIndex, llm) -> BaseRetriever:
        """Create hybrid BM25 + Vector retriever."""
        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self.settings.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )
        
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=self.settings.similarity_top_k,
            verbose=True
        )
        
        # Fusion retriever
        if self.settings.use_query_expansion:
            fusion_retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                retriever_weights=self.settings.retriever_weights,
                llm=llm,
                similarity_top_k=self.settings.top_k_rerank,
                num_queries=self.settings.num_queries,
                mode=self.settings.fusion_mode,
                verbose=True
            )
        else:
            # Simple fusion without query generation
            fusion_retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                retriever_weights=self.settings.retriever_weights,
                llm=llm,
                similarity_top_k=self.settings.similarity_top_k,
                num_queries=1,
                mode=self.settings.fusion_mode,
                verbose=True
            )
        
        return fusion_retriever
    
    def create_router_retriever(self, vector_index: VectorStoreIndex, llm) -> RouterRetriever:
        """Create router retriever that selects optimal strategy."""
        # Create different retrieval tools
        vector_tool = RetrieverTool.from_defaults(
            retriever=VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=self.settings.similarity_top_k,
                embed_model=Settings.embed_model
            ),
            description="Use for simple, direct queries where vector similarity is most important.",
            name="Vector Similarity Retriever"
        )
        
        hybrid_tool = RetrieverTool.from_defaults(
            retriever=self.create_hybrid_retriever(vector_index, llm),
            description="Use for complex queries that might benefit from both keyword and semantic search.",
            name="Hybrid BM25+Vector Retriever"
        )
        
        return RouterRetriever.from_defaults(
            selector=LLMSingleSelector.from_defaults(llm=llm),
            retriever_tools=[vector_tool, hybrid_tool],
            llm=llm
        )
    
    def apply_reranking(self, results: List[NodeWithScore], query: QueryBundle) -> List[NodeWithScore]:
        """Apply reranking to results."""
        if self.rerank_model and len(results) > self.settings.top_k_rerank:
            return self.rerank_model.postprocess_nodes(results, query)
        return results[:self.settings.top_k_rerank]

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
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata.get('name')
            mime_type = file_metadata.get('mimeType')
            
            logger.info(f"Downloading '{file_name}' (MIME type: {mime_type})...")

            if mime_type.startswith('application/vnd.google-apps'):
                if mime_type == 'application/vnd.google-apps.document':
                    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                    file_content_type = '.txt'
                else:
                    logger.error(f"Unsupported Google App type: {mime_type}")
                    return ""
            else:
                request = service.files().get_media(fileId=file_id)
                file_content_type = os.path.splitext(file_name)[1].lower()

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}%.")
            
            file_content = fh.getvalue()
            
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
    """Handles processing of multiple document types with LlamaIndex nodes."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}

    def __init__(self, folder_id: str):
        self.processed_documents = {}
        self.document_handler = DocumentHandler(folder_id=folder_id)

    def process_extracted_text(self, document_handler: DocumentHandler) -> List[BaseNode]:
        """Process all supported documents and return LlamaIndex nodes."""
        all_nodes = []
        service = document_handler.service
        retrieved_files = self.document_handler.list_drive_files(folder_id=document_handler.folder_id, service=service)

        if not retrieved_files:
            logger.error(f"No files found in the provided folder.")
            return all_nodes
        
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend([f for f in retrieved_files if f["name"].lower().endswith(ext)])

        logger.info(f"Found {len(supported_files)} supported documents")
        
        for file in supported_files:
            try:
                text = self.document_handler.download_and_extract_text(service, file_id=file['id'])
                if text.strip():
                    nodes = self.create_nodes_from_document(file, text)
                    all_nodes.extend(nodes)
                    self.processed_documents[file['name']] = len(nodes)
                    logger.info(f"Processed {file['name']}: {len(nodes)} nodes")
                else:
                    logger.warning(f"No text extracted from {file['name']}")
            except Exception as e:
                logger.error(f"Error processing {file['name']}: {str(e)}")
                continue
        
        return all_nodes
    
    def create_nodes_from_document(self, file, text: str) -> List[BaseNode]:
        """Create LlamaIndex nodes from document text."""
        
        text_splitter = RecursiveTokenChunker(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!"],
        )
        
        chunks = text_splitter.split_text(text)
        
        nodes = []
        for i, chunk in enumerate(chunks):
            node = TextNode(
                text=chunk,
                id_=str(uuid.uuid4()),
                metadata={
                    "source_document": file['name'],
                    "document_type": file['mimeType'],
                    "chunk_index": i,
                    "last_modified": file['modifiedTime']
                }
            )
            nodes.append(node)
        
        return nodes

class EnhancedVectorStore:
    """Enhanced vector store with advanced retrieval capabilities."""
    
    def __init__(self, chroma_client, collection_name="enhanced_rag_collection"):
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

class EnhancedRAGSystem:
    """Complete enhanced RAG system with Ollama and advanced retrieval."""
    
    def __init__(
        self, 
        gemini_api_key: str, 
        folder_id: str,
        model_settings: ModelSettings = ModelSettings(),
        retriever_settings: RetrieverSettings = RetrieverSettings()
    ):
        self.gemini_api_key = gemini_api_key
        self.folder_id = folder_id
        self.model_settings = model_settings
        self.retriever_settings = retriever_settings
        
        # Initialize model managers
        self.ollama_manager = OllamaModelManager(model_settings.ollama_host, model_settings.ollama_port)
        self.hf_embedding_manager = HuggingFaceEmbeddingManager(model_settings.embedding_model)
        
        # Initialize components
        self._setup_models()
        self._setup_vector_store()
        self._setup_retriever()
        
        # Document processing
        self.document_processor = MultiDocumentProcessor(self.folder_id)
        self.processed_nodes = []
        
        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def _setup_models(self):
        """Setup language models and embeddings."""
        try:
            # Setup embeddings
            if self.model_settings.use_hf_embeddings:
                Settings.embed_model = self.hf_embedding_manager.get_embedding_model()
                logger.info("Using HuggingFace embeddings")
            else:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=SecretStr(self.gemini_api_key)
                )
                logger.info("Using Google embeddings")
            
            # Setup LLM
            if self.model_settings.use_ollama:
                if not self.ollama_manager.check_ollama_connection():
                    raise Exception("Ollama server is not running. Please start Ollama.")
                
                if not self.ollama_manager.check_model_exists(self.model_settings.ollama_model):
                    logger.info(f"Pulling Ollama model: {self.model_settings.ollama_model}")
                    if not self.ollama_manager.pull_model(self.model_settings.ollama_model):
                        raise Exception(f"Failed to pull model: {self.model_settings.ollama_model}")
                
                Settings.llm = Ollama(
                    model=self.model_settings.ollama_model,
                    base_url=f"http://{self.model_settings.ollama_host}:{self.model_settings.ollama_port}",
                    temperature=self.model_settings.temperature,
                    context_window=self.model_settings.context_window,
                    request_timeout=300.0
                )
                
                # Also create LangChain-compatible LLM for query expansion
                self.langchain_llm = ChatOllama(
                    model=self.model_settings.ollama_model,
                    base_url=f"http://{self.model_settings.ollama_host}:{self.model_settings.ollama_port}",
                    temperature=self.model_settings.temperature,
                )
                logger.info(f"Using Ollama model: {self.model_settings.ollama_model}")
            else:
                self.langchain_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=SecretStr(self.gemini_api_key)
                )
                logger.info("Using Google Gemini model")
                
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
    
    def _setup_vector_store(self):
        """Setup vector store."""
        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path.cwd()
        
        db_path = base_dir / "chroma_db_enhanced"
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.vector_store = EnhancedVectorStore(self.chroma_client)
    
    def _setup_retriever(self):
        """Setup advanced retriever."""
        self.advanced_retriever = AdvancedRetriever(self.retriever_settings)
    
    def generate_alternative_queries(self, user_query: str, n: int = 3) -> List[str]:
        """Generate alternative queries using the configured LLM."""
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
                
                history_str = "\n".join(formatted_messages[-6:])

            prompt = f"""
            You are a query reformulation assistant.
            Given the user query: "{user_query}"
            
            Generate {n} alternative, well-structured queries with the same meaning to be used in context retrieval from a vector DB.
            Make sure you resolve any pronouns or ambiguous references using the chat history.
            
            Conversation History:
            {history_str}
            
            Return only the alternative queries, one per line.
            """
            
            response = self.langchain_llm.invoke([HumanMessage(content=prompt)])
            content = response.content if isinstance(response.content, str) else str(response.content)
            
            # Parse the response to extract queries
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            return queries[:n]
            
        except Exception as e:
            logger.warning(f"Could not generate alternative queries: {e}. Using original query only.")
            return [user_query]
    
    def process_documents(self) -> bool:
        """Process all documents and create vector index."""
        try:
            # Process documents to get LlamaIndex nodes
            self.processed_nodes = self.document_processor.process_extracted_text(
                self.document_processor.document_handler
            )
            
            if not self.processed_nodes:
                logger.warning("No nodes were created from documents")
                return False
            
            logger.info(f"Successfully processed {len(self.processed_nodes)} document nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def generate_answer_with_citations(self, query: str) -> Tuple[str, List[str]]:
        """Generate answer using advanced retrieval and citations."""
        try:
            if not self.processed_nodes:
                return "No documents have been processed yet. Please process documents first.", []
            
            # Create vector index from processed nodes
            vector_index = VectorStoreIndex(nodes=self.processed_nodes)
            
            # Choose retrieval strategy based on number of nodes
            if len(self.processed_nodes) > self.retriever_settings.top_k_rerank:
                retriever = self.advanced_retriever.create_router_retriever(vector_index, Settings.llm)
            elif self.retriever_settings.use_hybrid_retrieval:
                retriever = self.advanced_retriever.create_hybrid_retriever(vector_index, Settings.llm)
            else:
                retriever = VectorIndexRetriever(
                    index=vector_index,
                    similarity_top_k=self.retriever_settings.similarity_top_k,
                    embed_model=Settings.embed_model
                )
            
            # Generate alternative queries if enabled
            if self.retriever_settings.use_query_expansion:
                alt_queries = self.generate_alternative_queries(query, n=3)
                all_queries = [query] + alt_queries
            else:
                all_queries = [query]
            
            # Retrieve documents for all queries
            all_results = []
            for q in all_queries:
                query_bundle = QueryBundle(query_str=q)
                results = retriever.retrieve(query_bundle)
                all_results.extend(results)
            
            # Apply reranking if configured
            if self.advanced_retriever.rerank_model and all_results:
                query_bundle = QueryBundle(query_str=query)
                all_results = self.advanced_retriever.apply_reranking(all_results, query_bundle)
            
            # Deduplicate results
            unique_results = {}
            for result in all_results:
                node_id = result.node.node_id
                if node_id not in unique_results or result.score > unique_results[node_id].score:
                    unique_results[node_id] = result
            
            final_results = list(unique_results.values())[:12]
            
            if not final_results:
                return "I couldn't find relevant information in the document collection to answer your question.", []
            
            # Prepare context and source tracking
            source_map = {}
            context_parts = []
            
            for i, result in enumerate(final_results):
                source_key = f"source_{i+1}"
                source_doc = result.node.metadata.get('source_document', 'Unknown')
                source_map[source_key] = source_doc
                
                context_parts.append(
                    f"[{source_key} from {source_doc}]:\n{result.node.text}"
                )
            
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
                    
                    history_str = "\n".join(formatted_messages[-6:])
            except:
                history_str = ""
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            You are an expert AI assistant. Your task is to answer the user's question based *only* on the provided context from the document collection.

            **Strict Instructions:**
            1. Analyze the provided "Context from Documents" thoroughly.
            2. Use the following provided context to provide a detailed and accurate answer. Be sure to keep your responses relevant to the question.
            3. You can derive deductions from the provided context to answer questions but be sure to stay within the context of the document. Make the conversation flow naturally and try not to sound like a robot. For example, don't say "Based on the provided documents, ...". or "The text suggests that ...". Instead, just answer the question naturally.
            4. When you use information from a specific passage, cite it by placing its corresponding source tag (e.g., `[source_1]`) immediately after the statement.
            5. You can use multiple citations for a single sentence if the information is synthesized from multiple sources.
            6. If the context does not contain the information needed to answer the question, you **MUST** state: "I could not find sufficient information in the provided documents to answer this question." Do not use outside knowledge or make assumptions.

            **Conversation History:**
            {history}

            **Context from Documents:**
            {context}

            **Question:** {question}

            **Answer (with citations):**
            """)
            
            # Generate response using configured LLM
            prompt = prompt_template.format(
                context=context,
                question=query,
                history=history_str
            )
            
            response = self.langchain_llm.invoke([HumanMessage(content=prompt)])
            answer_text = response.content if isinstance(response.content, str) else str(response.content)
            
            # Process citations
            used_sources = set(re.findall(r'\[(source_\d+)\]', answer_text))
            
            if not used_sources:
                final_answer = answer_text
                sources_list = sorted(list(set(source_map.values())))
            else:
                # Map unique document filenames to final citation numbers
                unique_cited_filenames = sorted(list(set(source_map[marker] for marker in used_sources)))
                filename_to_citation_num = {filename: i + 1 for i, filename in enumerate(unique_cited_filenames)}

                # Replace each [source_X] marker with its corresponding final citation number
                final_answer = answer_text
                for marker in used_sources:
                    source_filename = source_map[marker]
                    citation_num = filename_to_citation_num[source_filename]
                    final_answer = final_answer.replace(f'[{marker}]', f' [{citation_num}]')

                # Build the de-duplicated footnote section
                footnote_section = ["\n\n---", "**Sources:**"]
                for filename, num in filename_to_citation_num.items():
                    footnote_section.append(f"[{num}] {filename}")
                
                final_answer += "\n" + "\n".join(footnote_section)
                sources_list = unique_cited_filenames

            # Save to memory
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
            "total_nodes": len(self.processed_nodes),
            "model_settings": {
                "llm_type": "Ollama" if self.model_settings.use_ollama else "Gemini",
                "llm_model": self.model_settings.ollama_model if self.model_settings.use_ollama else "gemini-2.0-flash",
                "embedding_type": "HuggingFace" if self.model_settings.use_hf_embeddings else "Google",
                "embedding_model": self.model_settings.embedding_model if self.model_settings.use_hf_embeddings else "models/embedding-001"
            },
            "retrieval_settings": {
                "use_hybrid": self.retriever_settings.use_hybrid_retrieval,
                "use_query_expansion": self.retriever_settings.use_query_expansion,
                "similarity_top_k": self.retriever_settings.similarity_top_k,
                "rerank_top_k": self.retriever_settings.top_k_rerank
            },
            "collection_status": "Ready" if self.processed_nodes else "Not Initialized"
        }
        return stats

# Streamlit Application with Enhanced Features

@st.cache_resource
def get_enhanced_rag_system(api_key, folder_id, use_ollama=True, use_hf_embeddings=True):
    """Cached function to initialize the enhanced RAG system."""
    model_settings = ModelSettings(
        use_ollama=use_ollama,
        use_hf_embeddings=use_hf_embeddings
    )
    retriever_settings = RetrieverSettings()
    return EnhancedRAGSystem(api_key, folder_id, model_settings, retriever_settings)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Multi-Document RAG System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS (same as before)
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
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .enhancement-badge {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
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
        <h1>🚀 Enhanced Multi-Document RAG System</h1>
        <p>Advanced document processing with Ollama, HuggingFace embeddings, and hybrid retrieval</p>
        <span class="enhancement-badge">NEW: Ollama Support</span>
        <span class="enhancement-badge">NEW: Advanced Retrieval</span>
        <span class="enhancement-badge">NEW: HF Embeddings</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ System Configuration")
        
        # Model configuration
        use_ollama = st.checkbox("Use Ollama for LLM", value=True, help="Use local Ollama models instead of Gemini")
        use_hf_embeddings = st.checkbox("Use HuggingFace Embeddings", value=True, help="Use local HF embeddings instead of Google")
        use_hybrid_retrieval = st.checkbox("Enable Hybrid Retrieval", value=True, help="Combine BM25 and vector search")
        use_query_expansion = st.checkbox("Enable Query Expansion", value=True, help="Generate alternative queries")
        
        st.markdown("---")
        st.markdown("## 📄 Document Management")
        
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            if folder_id:
                with st.spinner("Initializing enhanced RAG system..."):
                    try:
                        # Initialize system with current settings
                        rag_system = get_enhanced_rag_system(
                            gemini_api_key, 
                            folder_id, 
                            use_ollama=use_ollama,
                            use_hf_embeddings=use_hf_embeddings
                        )
                        st.session_state.rag_system = rag_system
                        
                        with st.spinner("Processing documents with advanced pipeline..."):
                            success = rag_system.process_documents()
                            if success:
                                st.session_state.documents_processed = True
                                st.success("✅ Documents processed with enhanced pipeline!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to process documents")
                    except Exception as e:
                        st.error(f"❌ Error initializing system: {str(e)}")
                        if "Ollama" in str(e):
                            st.info("💡 Make sure Ollama is running: `ollama serve`")
            else:
                st.error("Please provide a valid Google Drive folder ID")
    
    # Initialize session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    # System statistics in sidebar
    if st.session_state.documents_processed and st.session_state.rag_system:
        with st.sidebar:
            stats = st.session_state.rag_system.get_system_stats()
            
            st.markdown("## 📊 System Statistics")
            st.markdown(f"""
            <div class="stats-card">
                <strong>Documents Processed:</strong> {stats['total_documents']}<br>
                <strong>Total Nodes:</strong> {stats['total_nodes']:,}<br>
                <strong>Status:</strong> {stats['collection_status']}<br><br>
                
                <strong>🤖 Model Configuration:</strong><br>
                <strong>LLM:</strong> {stats['model_settings']['llm_type']} ({stats['model_settings']['llm_model']})<br>
                <strong>Embeddings:</strong> {stats['model_settings']['embedding_type']}<br><br>
                
                <strong>🔍 Retrieval Settings:</strong><br>
                <strong>Hybrid Retrieval:</strong> {'✅' if stats['retrieval_settings']['use_hybrid'] else '❌'}<br>
                <strong>Query Expansion:</strong> {'✅' if stats['retrieval_settings']['use_query_expansion'] else '❌'}<br>
                <strong>Top-K:</strong> {stats['retrieval_settings']['similarity_top_k']}<br>
                <strong>Rerank Top-K:</strong> {stats['retrieval_settings']['rerank_top_k']}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📁 Document Breakdown"):
                for doc, nodes in stats['processed_documents'].items():
                    st.write(f"• {doc}: {nodes} nodes")
        
        # Clear chat button in sidebar
        with st.sidebar:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.clear_memory()
                st.success("Chat history cleared!")
                st.rerun()
    
    # Main chat interface
    if not st.session_state.documents_processed:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #fff3cd; border-radius: 10px; margin: 2rem 0;">
            <h3 style="color: #856404;">🔧 System Not Ready</h3>
            <p style="color: #856404;">Please configure and process documents from the sidebar to start using the enhanced system.</p>
            <div style="margin-top: 1rem;">
                <p><strong>New Features Available:</strong></p>
                <ul style="text-align: left; display: inline-block;">
                    <li>🦙 Local Ollama model support</li>
                    <li>🤗 HuggingFace embedding models</li>
                    <li>🔄 Hybrid BM25 + Vector retrieval</li>
                    <li>🎯 Query expansion for better results</li>
                    <li>⚡ Advanced reranking with sentence transformers</li>
                    <li>🧠 Router-based retrieval strategy selection</li>
                </ul>
            </div>
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
        # Welcome message with enhanced features
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f8f9ff 0%, #e8f2ff 100%); border-radius: 20px; margin: 2rem 0;">
            <h3 style="color: #667eea;">🚀 Enhanced RAG System Ready</h3>
            <p style="color: #666; font-size: 1.1rem;">Ask me anything about the processed documents. I now use advanced retrieval strategies and local models for better performance.</p>
            
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 200px;">
                    <h4 style="margin: 0; color: #4CAF50;">🦙 Local Models</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Ollama integration for privacy</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 200px;">
                    <h4 style="margin: 0; color: #2196F3;">🔍 Smart Retrieval</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Hybrid BM25 + Vector search</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 200px;">
                    <h4 style="margin: 0; color: #FF9800;">⚡ Advanced Processing</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Query expansion & reranking</p>
                </div>
            </div>
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
        submit_button = st.button("🚀 Send", type="primary", use_container_width=True)
    
    # Handle query submission
    if submit_button and user_query.strip():
        with st.spinner("🧠 Generating enhanced response..."):
            try:
                answer, sources = st.session_state.rag_system.generate_answer_with_citations(user_query)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "user": user_query,
                    "assistant": answer,
                    "sources": sources
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()