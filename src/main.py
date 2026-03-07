import os
from typing import List, Optional, Dict
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests
from bs4 import BeautifulSoup
import re

from ingest import fetch_chunks
URL = "https://www.ga4gh.org/framework/"
# Placeholder imports for future implementation
# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import PyPDFLoader

class RegBot:
    """
    Main class for the GA4GH Compliance Assistant.
    """
    load_dotenv()
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    ):
        self.api_key = api_key or os.getenv("HF_API_TOKEN")
        if not self.api_key:
            raise ValueError("HF_API_TOKEN not found")
        
        self.vector_db = None
        self.model_name = model_name
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
    
        

    def retrieve_relevant_clauses(self, user_query: str) -> List[str]:
        """
        Phase 2: RAG Implementation.
        Search vector DB for clauses relevant to the user's consent form.
        """
        print("Retrieving regulatory context...")
        return ["Clause 4.1: Data Sharing", "Clause 2.3: Patient Consent"]

    def check_compliance(self, user_consent_form: str) -> dict:
        """
        Phase 3: LLM Analysis.
        Compares user input against retrieved GA4GH clauses.
        """
        print("Analyzing compliance gap...")
        # Placeholder for LLM Chain
        return {
            "status": "Non-Compliant",
            "missing_elements": ["Data Use Limitation", "Cloud Storage Provision"],
            "suggested_fix": "Add specific clause regarding secondary use of data."
        }

if __name__ == "__main__":
    # Entry point for testing the pipeline
    bot = RegBot()
    clauses = bot.ingest_policy_documents(URL)
    for c in clauses[:4]:
        print(f"[{c['chunk_id']}] {c['title']} (level={c['level']}, parent={c['parent_id']})")
        print(f"  {c['content'][:120]}...")
        print()

