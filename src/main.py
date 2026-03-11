import json
import os
from pathlib import Path
from typing import Any, List, Optional, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from compliance import ComplianceChecker, _readable_citation, _load_display_names
from ingestion import fetch_chunks, fetch_pdf_chunks, fetch_all_pdfs, VectorStore
from ingestion.ingest_pdf import _load_source_config

URL = "https://www.ga4gh.org/framework/"

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
        
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.store = VectorStore(self.embedding_model)
        self.load_llm()
        print("Initializing RegBot Core...")

    def load_llm(self):
        #Initialize hugging Face API client.

        self.client = InferenceClient(
            model=self.model_name,
            token=self.api_key
        )

    def ingest_policy_documents(self, source: str = None) -> int:
        """ingest policy docs from url file or dir 
        
        """
        if source is None:
            source = URL

        print(f"Loading policy document from: {source}")

        path = Path(source)
        if path.is_dir():
            chunks = fetch_all_pdfs(source)
        elif path.is_file() and path.suffix.lower() == ".pdf":
            cfg = _load_source_config().get(path.name, {})
            chunks = fetch_pdf_chunks(source, doc_type=cfg.get("doc_type", "policy"))
        else:
            chunks = fetch_chunks(source)

        if not chunks:
            print("No chunks extracted.")
            return 0

        stored = self.store.store_chunks(chunks)
        del chunks
        print(f"Stored {stored} chunks in ChromaDB.")
        return stored

    def retrieve_relevant_clauses(self, user_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Phase 2: RAG Implementation.
        Query ChromaDB for clauses relevant to the user's consent form.
        """
        print("Retrieving relevant clauses...")
        results = self.store.query(user_query, top_k)
        if not results:
            print("Warning: ChromaDB collection is empty — run ingest first.")
        return results

    def check_compliance(self, user_consent_form: str, clauses: List[Dict[str,str]], top_k: int = 10) -> dict:
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
    bot.ingest_policy_documents("C:/Users/AliyaSabir/GA4GH-RegBot/src/docs")
    
    # Test retrieval with sample query
    query = "We will obtain explicit consent from participants for data sharing and secondary research."
    retrieved_clauses = bot.retrieve_relevant_clauses(query, top_k=10)
    
    print(f"\nTop {len(retrieved_clauses)} relevant clauses for query:")
    print(f"'{query}'\n")
    
    for r in retrieved_clauses:
        citation = _readable_citation(r['document_name'], r['clause_number'], _load_display_names(), r.get('title', ''))
        print(f"[{citation}] (similarity: {r['similarity']:.4f})")
        print(f"  {r['text']}")
        print()

    llm_output = bot.check_compliance(query, retrieved_clauses, top_k=10)
    print("\nCompliance Analysis Result:") 
    print(json.dumps(llm_output, indent=2, ensure_ascii=False))
