import json
import re
from pathlib import Path
from typing import Any, List, Dict
from huggingface_hub import InferenceClient

SOURCES_CONFIG = Path(__file__).parent / "pdf_sources.json"


def _load_display_names() -> Dict[str, str]:
    #map document_name -> display_name
    if SOURCES_CONFIG.exists():
        with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
            sources = json.load(f)
        return {
            v["document_name"]: v.get("display_name", v["document_name"])
            for v in sources.values()
        }
    return {}


def _readable_citation(document_name: str, clause_number: str, display_map: Dict[str, str], title: str = "") -> str:
    #format a human-readable citation string
    display = display_map.get(document_name, document_name.replace("_", " ").title())
    # for consent toolkit docs, title is more meaningful than section number
    if title and any(x in document_name for x in
                     ["consent_clauses", "consent_toolkit"]):
        return f"{display} \u2013 {title.rstrip('.')}"

    clean = re.sub(r"_part(\d+)", r" (Part \1)", clause_number)
    clean = clean.replace("_", " ").rstrip(".")
    #en dash for readability 
    citation = f"{display} \u2013 Section {clean}"
    if title:
        citation += f": {title.rstrip('.')}"
    return citation

class ComplianceChecker:

    def __init__(self, client: InferenceClient):
        self.client = client
        self._display_map = _load_display_names()

    def _build_prompt(
        self,
        user_consent_form: str,
        clauses: List[Dict[str, str]]
    ) -> str:
        """construct the prompt sent to the LLM"""

        clause_context = "\n".join(
            f"[{_readable_citation(c['document_name'], c['clause_number'], self._display_map, c.get('title', ''))}] (source: {c.get('source', '')}): {c['text']}"
            for c in clauses
        )

        return (
            "You are a regulatory compliance assistant.\n\n"
            "Evaluate whether the following consent form complies with GA4GH regulatory clauses.\n\n"
            f"CONSENT FORM:\n{user_consent_form}\n\n"
            f"RELEVANT GA4GH CLAUSES:\n{clause_context}\n\n"
            "TASK:\n"
            "Compare the consent form against each GA4GH clause above.\n"
            "For each clause requirement that the consent form fails to address, describe WHAT is missing from the consent form in plain language.\n\n"
            "RULES FOR missing_elements:\n"
            "- Each element must describe a specific gap in the consent form (e.g. 'No disclosure of data sharing with third parties')\n"
            "- Do NOT use clause titles or citation labels as missing elements\n"
            "- Focus on what the consent form lacks, not which clause requires it\n\n"
            "RULES FOR citations:\n"
            "- Use the exact citation labels from the clauses above (text in square brackets)\n"
            "- Include the source URL from the clause (text after 'source:')\n"
            "- Include a short excerpt of the relevant clause text\n\n"
            "Return ONLY valid JSON in the format:\n\n"
            '{\n'
            '  "status": "Compliant | Partial | Non-Compliant",\n'
            '  "missing_elements": ["No description of what happens to samples after testing", "No withdrawal rights or data deletion procedure"],\n'
            '  "suggested_fix": "1. Add disclosure of sample storage and data retention policy. 2. Include withdrawal rights with procedure for data deletion. (numbered list)",\n'
            '  "citations": [\n'
            '    {"citation": "Document Name \u2013 Topic", "source_url": "https://...", "title": "clause title", "excerpt": "relevant clause text"}\n'
            '  ]\n'
            '}\n\n'
            "Respond only with a single JSON object. Do not include preamble, explanation, or markdown."
        )
    
    def _extract_json(self, text: str) -> Dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
        
    def check_compliance(self, user_consent_form:str, clauses: List[Dict[str,str]], top_k: int = 10) -> Dict[str,Any]:
        top_clauses = clauses[:top_k]

        fallback = {
            "status": "Unknown",
            "missing_elements": [],
            "suggested_fix": "",
            "citations": [
                {
                    "citation": _readable_citation(c.get("document_name", ""), c.get("clause_number", "unknown"), self._display_map, c.get("title", "")),
                    "title": c.get("title", ""),
                    "excerpt": c.get("text", ""),
                }
                for c in top_clauses
            ]
        }
        prompt = self._build_prompt(user_consent_form, top_clauses)   

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze regulatory compliance for genomic data consent forms."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2500,
                temperature=0.0
            )
            llm_output = response.choices[0].message.content
        except Exception as e:
            print("LLM request failed:", e)
            return fallback

        parsed = self._extract_json(llm_output) 
        if parsed is None:
            print("DEBUG: JSON parse failed. Raw output:", llm_output[:300])
            return fallback
        return parsed
    