import json
import os
from typing import Any, List, Optional, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

from compliance import ComplianceChecker
from ingest import fetch_chunks

URL = "https://www.ga4gh.org/framework/"
# Placeholder imports for future implementation
# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import PyPDFLoader

load_dotenv()

class RegBot:
    """
    Main class for the GA4GH Compliance Assistant.
    """
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    ):
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN not found")
        
        self.vector_db = None
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_llm()
        print("Initializing RegBot Core...")

    def load_llm(self):
        #Initialize hugging Face API client.

        self.client = InferenceClient(
            model=self.model_name,
            token=self.api_key
        )

    def ingest_policy_documents(self, url: str = None) -> List[Dict[str, str]]:
        print(f"Loading policy document from: {url}")
        return fetch_chunks(url) if url else fetch_chunks()
    
        

    def retrieve_relevant_clauses(self, user_query: str, clauses: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Phase 2: RAG Implementation.
        Search vector DB for clauses relevant to the user's consent form.
        """
        print("Retrieving relevant clauses...")
        
        if not clauses:
            print("Warning: No clauses provided for retrieval")
            return []
        
        user_embedding = self.embedding_model.encode(user_query, convert_to_tensor=True)

        results = []
        for chunk in clauses:
            # Skip chunks with empty content
            if not chunk.get("content") or not chunk["content"].strip():
                continue
            
            clause_embedding = self.embedding_model.encode(chunk['content'], convert_to_tensor=True)
            similarity = util.cos_sim(user_embedding, clause_embedding)[0][0].item()
            results.append({
                "clause_number": chunk["chunk_id"],
                "title": chunk["title"],
                "text": chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"],
                "similarity": round(similarity, 4),
                "source": chunk.get("source_url", URL)
            })
            
        # Sort by similarity and return top k
        top_results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
        return top_results

    def check_compliance(self, user_consent_form: str, clauses: List[Dict[str,str]], top_k: int = 3) -> dict:
        """
        Phase 3: LLM Analysis.
        Compares user input against retrieved GA4GH clauses.
        """
        print("Analyzing compliance gap...")
        checker = ComplianceChecker(self.client)
        return checker.check_compliance(user_consent_form, clauses, top_k)

if __name__ == "__main__":
    # Entry point for testing the pipeline
    bot = RegBot()
    clauses = bot.ingest_policy_documents(URL)
    
    # Test retrieval with sample query
    query = "We will obtain explicit consent from participants for data sharing and secondary research."
    retrieved_clauses = bot.retrieve_relevant_clauses(query, clauses, top_k=3)
    
    print(f"\nTop {len(retrieved_clauses)} relevant clauses for query:")
    print(f"'{query}'\n")
    
    for r in retrieved_clauses:
        print(f"[{r['clause_number']}] {r['title']} (similarity: {r['similarity']:.4f})")
        print(f"  {r['text']}")
        print()

    llm_output = bot.check_compliance(query, retrieved_clauses, top_k=3)
    print("\nCompliance Analysis Result:") 
    print(json.dumps(llm_output, indent=2))
