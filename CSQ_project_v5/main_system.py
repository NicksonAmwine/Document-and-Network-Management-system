import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, cast

# LangChain and other core imports
import chromadb
from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory

# Local imports
from data_models import QueryExpansion
from document_processor import MultiDocumentProcessor
from vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)

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

    def generate_alternative_queries(self, user_query: str, n: int = 3) -> List[str]:
        """Use the LLM to generate structured alternative queries."""

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

        prompt = f"""
        You are a query reformulation assistant.
        Given the user query:

        "{user_query}"

        Generate {n} alternative, well-structured queries with the same meaning to be used in context retrieval from a vector DB.
        Make sure you resolve any pronouns or ambiguous references using the chat history.
        Conversation History:
        {history_str}
        """
        try:
            model_structure = self.llm.with_structured_output(QueryExpansion)
            response = model_structure.invoke([HumanMessage(content=prompt)])
            parsed = cast(QueryExpansion, response)
            return parsed.queries
        except Exception as e:
            # fallback: return the original query if parsing fails
            logger.warning(f"Could not parse structured queries: {e}. Using original query only.")
            return [user_query]

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
            alt_queries = self.generate_alternative_queries(query, n=3)
            all_queries = [query] + alt_queries
            # Retrieve relevant documents
            all_results = []
            for q in all_queries:
                all_results.extend(self.vector_store.similarity_search(q, k=5))

            # Step 3: deduplicate and cap results (to avoid context bloat)
            unique_results = {}
            for r in all_results:
                if r.chunk_id not in unique_results:
                    unique_results[r.chunk_id] = r

            retrieval_results = list(unique_results.values())[:12]  
            
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
                final_answer = answer_text
                sources_list = sorted(list(set(res.source_document for res in retrieval_results)))
            else:
                # Map unique document filenames to a final citation number
                unique_cited_filenames = sorted(list(set(source_map[marker].source_document for marker in used_sources)))
                filename_to_citation_num = {filename: i + 1 for i, filename in enumerate(unique_cited_filenames)}

                # Replace each [source_X] marker with its corresponding final citation number
                final_answer = answer_text
                for marker in used_sources:
                    source_filename = source_map[marker].source_document
                    citation_num = filename_to_citation_num[source_filename]
                    final_answer = final_answer.replace(f'[{marker}]', f' [{citation_num}]')

                # Build the de-duplicated footnote section
                footnote_section = ["\n\n---", "**Sources:**"]
                for filename, num in filename_to_citation_num.items():
                    footnote_section.append(f"[{num}] {filename}")
                
                final_answer += "\n" + "\n".join(footnote_section)
                sources_list = unique_cited_filenames

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
