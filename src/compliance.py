import json
import re
from typing import Any, List, Dict
from huggingface_hub import InferenceClient

class ComplianceChecker:
    """
    LLM-powered compliance checker for GA4GH clauses.
    """

    def __init__(self, client: InferenceClient):
        self.client = client

    def _build_prompt(
        self,
        user_consent_form: str,
        clauses: List[Dict[str, str]]
    ) -> str:
        """Construct the prompt sent to the LLM."""

        clause_context = "\n".join(
            f"[{c['clause_number']}] {c['title']}: {c['text']}"
            for c in clauses
        )

        return (
            "You are a regulatory compliance assistant.\n\n"
            "Evaluate whether the following consent form complies with GA4GH regulatory clauses.\n\n"
            f"CONSENT FORM:\n{user_consent_form}\n\n"
            f"RELEVANT GA4GH CLAUSES:\n{clause_context}\n\n"
            "TASK:\n"
            "Compare the consent form against the clauses.\n\n"
            "Identify:\n"
            "• whether the form is compliant\n"
            "• missing regulatory elements\n"
            "• suggested improvements\n\n"
            "Return ONLY valid JSON in the format:\n\n"
            '{\n'
            '  "status": "Compliant | Partial | Non-Compliant",\n'
            '  "missing_elements": ["list of missing requirements"],\n'
            '  "suggested_fix": "short recommendation",\n'
            '  "clause_citations": ["clause numbers used"]\n'
            '}\n\n'
            "Do not include any text outside the JSON."
        )
    
    def _extract_json(self, text: str) -> Dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
        
    def check_compliance(self, user_consent_form:str, clauses: List[Dict[str,str]], top_k: int = 3) -> Dict[str,Any]:
        top_clauses = clauses[:top_k]

        fallback = {
            "status": "Unknown",
            "missing_elements": [],
            "suggested_fix": "",
            "clause_citations": [c.get("clause_number", "unknown") for c in top_clauses]
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
                max_tokens=500,
                temperature=0.2
            )
            llm_output = response.choices[0].message.content
        except Exception as e:
            print("LLM request failed:", e)
            return fallback

        parsed = self._extract_json(llm_output)

        if parsed is None:
            return fallback

        return parsed
    