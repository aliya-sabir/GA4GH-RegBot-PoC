"""Run a single test scenario through the RegBot pipeline."""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from compliance import _readable_citation, _load_display_names
from main import RegBot

DEFAULT_DOCS_PATH = "C:/Users/AliyaSabir/GA4GH-RegBot/src/docs"
DEFAULT_SCENARIO_PATH = Path(__file__).parent / "evaluation" / "tests.jsonl"
DEFAULT_TOP_K = 7


def _clean_text(value: str) -> str:
    if not value:
        return ""
    return (
        value.replace("â€“", "-")
        .replace("â€”", "-")
        .replace("Ã¢â‚¬â€", "-")
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€˜", "'")
        .replace("?", "-")
    )


def _extract_scenario_title(scenario: Dict[str, Any]) -> str:
    consent_form = _clean_text(scenario.get("consent_form", ""))
    lines = [line.strip() for line in consent_form.splitlines() if line.strip()]

    for line in lines:
        if line.lower().startswith("study title:"):
            return line.split(":", 1)[1].strip()
        if line.lower().startswith("project:"):
            return line.split(":", 1)[1].strip()

    if lines:
        return lines[0]
    return scenario.get("id", "Unknown scenario")


def _shorten(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", _clean_text(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _format_result_score(result: Dict[str, Any]) -> str:
    similarity = result.get("similarity")
    if similarity is not None:
        return f"similarity: {similarity:.4f}"

    rerank_score = result.get("rerank_score")
    if rerank_score is not None:
        return f"rerank: {rerank_score:.4f}"

    bm25_score = result.get("bm25_score")
    if bm25_score is not None:
        return f"bm25: {bm25_score:.4f}"

    return "score: n/a"


def _load_test_scenario(
    scenario_path: Path = DEFAULT_SCENARIO_PATH,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    scenarios: List[Dict[str, Any]] = []
    with scenario_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            scenarios.append(json.loads(line))

    if not scenarios:
        raise ValueError(f"No scenarios found in {scenario_path}")

    if scenario_id:
        for scenario in scenarios:
            if scenario.get("id") == scenario_id:
                return scenario
        raise ValueError(f"Scenario '{scenario_id}' not found in {scenario_path}")

    return scenarios[0]


def run_test_scenario(
    scenario_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    docs_path: str = DEFAULT_DOCS_PATH,
) -> Dict[str, Any]:
    bot = RegBot()
    bot.ingest_policy_documents(docs_path)

    scenario = _load_test_scenario(scenario_id=scenario_id)
    consent_form = scenario["consent_form"]
    scenario_name = scenario.get("id", "unknown")
    scenario_title = _extract_scenario_title(scenario)
    expected_status = scenario.get("expected_output", {}).get("status", "Unknown")

    _print_section("Scenario")
    print(f"ID: {scenario_name}")
    print(f"Title: {scenario_title}")
    print(f"Expected status: {expected_status}")

    retrieved_clauses = bot.retrieve_relevant_clauses(consent_form, top_k=top_k)

    _print_section(f"Top {len(retrieved_clauses)} Relevant Clauses")

    for index, r in enumerate(retrieved_clauses, start=1):
        citation = _readable_citation(
            r["document_name"],
            r["clause_number"],
            _load_display_names(),
            r.get("title", ""),
        )
        print(f"{index}. {citation} ({_format_result_score(r)})")
        print(f"   {_shorten(r['text'])}")
        print()

    llm_output = bot.check_compliance(consent_form, retrieved_clauses, top_k=top_k)
    actual_status = llm_output.get("status", "Unknown")
    missing_elements = llm_output.get("missing_elements", [])
    suggested_fix = llm_output.get("suggested_fix", "")
    citations = llm_output.get("citations", [])

    _print_section("Compliance Summary")
    print(f"Scenario: {scenario_name} - {scenario_title}")
    print(f"Expected status: {expected_status}")
    print(f"Actual status: {actual_status}")
    print(f"Status match: {'Yes' if actual_status == expected_status else 'No'}")

    if missing_elements:
        print("\nKey gaps:")
        for gap in missing_elements[:5]:
            print(f"- {_clean_text(gap)}")
        if len(missing_elements) > 5:
            print(f"- Plus {len(missing_elements) - 5} additional gap(s)")
    else:
        print("\nKey gaps: None identified")

    if suggested_fix:
        print("\nRecommended fix:")
        print(_shorten(suggested_fix, limit=500))

    if citations:
        print("\nSupporting citations:")
        for citation in citations[:4]:
            label = _clean_text(citation.get("citation", ""))
            title = _clean_text(citation.get("title", ""))
            if title:
                print(f"- {label}: {title}")
            else:
                print(f"- {label}")

    _print_section("Raw JSON")
    print(json.dumps(llm_output, indent=2, ensure_ascii=False))
    return llm_output


def main() -> None:
    load_dotenv()
    scenario_id = os.getenv("TEST_SCENARIO_ID", "scenario_4")
    run_test_scenario(scenario_id=scenario_id)


if __name__ == "__main__":
    main()
