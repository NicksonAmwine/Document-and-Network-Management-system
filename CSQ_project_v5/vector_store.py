import logging
from datetime import datetime
from typing import List, Dict

import chromadb
from chromadb.types import Metadata

# Local imports
from data_models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

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
