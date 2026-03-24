import os
from pathlib import Path
from typing import Any, List, Optional, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from compliance import ComplianceChecker
from ingestion import fetch_chunks, fetch_pdf_chunks, fetch_all_pdfs, VectorStore
from ingestion.ingest_pdf import _load_source_config

URL = "https://www.ga4gh.org/framework/"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

load_dotenv()

class RegBot:
    """
    Main class for the GA4GH Compliance Assistant.
    """
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    ):
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN not found")
        
        self.model_name = model_name
        embedding_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.store = VectorStore(self.embedding_model)
        self.load_llm()
        print("Initializing RegBot Core...")

    def load_llm(self):
        #initialize hugging Face API client.

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
            print("Warning: ChromaDB collection is empty â€” run ingest first.")
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
    # Delegate test runs to run_test.py to keep main uncluttered.
    from run_test import main as _run_test_main

    _run_test_main()
