import os
from typing import List, cast
from pydantic import BaseModel, SecretStr
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import dotenv_values
import logging

# Configure logging to see potential warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Copied directly from your CSQ_project_v4.py ---
# Pydantic model for structured output
class QueryExpansion(BaseModel):
    queries: List[str]

class QueryGenerationTester:
    """A simplified class to test the multi-query generation logic."""

    def __init__(self, gemini_api_key: str):
        # Initialize only the components needed for query generation
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=SecretStr(gemini_api_key)
        )
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )

    def generate_alternative_queries(self, user_query: str, n: int = 3) -> List[str]:
        """
        This function is an exact copy of the one in your EnhancedRAGSystem
        to ensure the test is accurate.
        """
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
        except Exception:
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
            logger.error(f"Could not parse structured queries: {e}. Using original query only.")
            return [user_query]

    def add_to_history(self, user_query: str, ai_response: str):
        """A dummy function to simulate conversation history."""
        self.memory.save_context({"input": user_query}, {"output": ai_response})


def main():
    """Main function to run the test."""
    # Load configuration
    config = dotenv_values("csq_project.env")
    gemini_api_key = config.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not gemini_api_key:
        print("ERROR: GOOGLE_API_KEY not found! Please check your csq_project.env file.")
        return

    print("--- Multi-Query Generation Test ---")
    tester = QueryGenerationTester(gemini_api_key)

    # You can pre-populate history to test pronoun resolution
    # tester.add_to_history("What are the main challenges for Digital Twins?", "The main challenges are standardization and security.")

    while True:
        user_query = input("\nEnter your query (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        print("\nGenerating alternative queries...")
        
        # Generate the queries
        alternative_queries = tester.generate_alternative_queries(user_query)
        
        print("\n--- Generated Queries ---")
        print(f"Original: {user_query}")
        for i, q in enumerate(alternative_queries):
            print(f"Query {i+1}:  {q}")
        print("-------------------------\n")

        mock_ai_response = input("Enter a mock AI response to this query (for history): ")
        tester.add_to_history(user_query, mock_ai_response)
        print("✅ History updated for the next turn.")

if __name__ == "__main__":
    main()