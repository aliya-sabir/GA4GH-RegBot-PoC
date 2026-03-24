import json
import re
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import InferenceClient

SOURCES_CONFIG = Path(__file__).parent / "pdf_sources.json"


def _load_display_names() -> Dict[str, str]:
    if SOURCES_CONFIG.exists():
        with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
            sources = json.load(f)
        return {
            value["document_name"]: value.get("display_name", value["document_name"])
            for value in sources.values()
        }
    return {}


def _readable_citation(
    document_name: str,
    clause_number: str,
    display_map: Dict[str, str],
    title: str = "",
) -> str:
    display = display_map.get(document_name, document_name.replace("_", " ").title())
    if title and any(token in document_name for token in ["consent_clauses", "consent_toolkit"]):
        return f"{display} \u2013 {title.rstrip('.')}"

    clean = re.sub(r"_part(\d+)", r" (Part \1)", clause_number)
    clean = clean.replace("_", " ").rstrip(".")
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
        clauses: List[Dict[str, str]],
    ) -> str:
        clause_context = "\n".join(
            (
                f"[{_readable_citation(c['document_name'], c['clause_number'], self._display_map, c.get('title', ''))}] "
                f"(source: {c.get('source', '')}): {c['text']}"
            )
            for c in clauses
        )

        return (
            "You are a regulatory compliance assistant for genomic data consent forms.\n\n"
            "CONSENT FORM TO EVALUATE:\n"
            f"{user_consent_form}\n\n"
            "RELEVANT GA4GH REGULATORY CLAUSES:\n"
            f"{clause_context}\n\n"
            "INSTRUCTIONS:\n"
            "Step 1 - Read the consent form carefully and identify what topics it DOES cover.\n"
            "Step 2 - For each GA4GH clause above, determine whether the consent form adequately addresses that requirement.\n"
            "Step 3 - List only requirements that the consent form FAILS to address or addresses inadequately.\n\n"
            "STRICT RULES:\n"
            "- A missing element must be something ABSENT from the consent form, not just mentioned in a clause.\n"
            "- If the consent form addresses a topic even partially, do NOT list it as missing.\n"
            "- Do NOT list items that are present in the consent form.\n"
            "- Each missing element must describe the gap in the consent form in plain language.\n"
            "- Do NOT use clause titles as missing elements.\n"
            "- Do NOT give unknown status.\n"
            "Return ONLY valid JSON:\n"
            '{\n'
            '  "status": "Compliant | Partial | Non-Compliant",\n'
            '  "missing_elements": ["specific gap in consent form language"],\n'
            '  "suggested_fix": "1. numbered specific action. 2. numbered specific action.",\n'
            '  "citations": [\n'
            '    {"citation": "exact label from the clause identifier", "source_url": "...", '
            '"title": "clause title", "excerpt": "exact clause text"}\n'
            "  ]\n"
            "}\n\n"
            "Respond only with the JSON object. No preamble, no markdown."
        )

    def _extract_json(self, text: str) -> Dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    def check_compliance(
        self,
        user_consent_form: str,
        clauses: List[Dict[str, str]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        top_clauses = clauses[:top_k]

        fallback = {
            "status": "Unknown",
            "missing_elements": [],
            "suggested_fix": "",
            "citations": [
                {
                    "citation": _readable_citation(
                        c.get("document_name", ""),
                        c.get("clause_number", "unknown"),
                        self._display_map,
                        c.get("title", ""),
                    ),
                    "title": c.get("title", ""),
                    "excerpt": c.get("text", ""),
                }
                for c in top_clauses
            ],
        }
        prompt = self._build_prompt(user_consent_form, top_clauses)

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze regulatory compliance for genomic data consent forms.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=2500,
                temperature=0.0,
            )
            llm_output = response.choices[0].message.content
        except Exception as exc:
            print("LLM request failed:", exc)
            return fallback

        parsed = self._extract_json(llm_output)
        if parsed is None:
            return fallback
        return parsed
