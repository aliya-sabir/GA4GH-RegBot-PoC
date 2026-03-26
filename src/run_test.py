import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from main import RegBot

DEFAULT_DOCS_PATH = "C:/Users/AliyaSabir/GA4GH-RegBot/src/docs"
DEFAULT_SCENARIO_PATH = Path(__file__).parent / "evaluation" / "tests.jsonl"
DEFAULT_TOP_K = 7


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
    retrieved_clauses = bot.retrieve_relevant_clauses(consent_form, top_k=top_k)
    llm_output = bot.check_compliance(consent_form, retrieved_clauses, top_k=top_k)

    print(json.dumps(llm_output, indent=2, ensure_ascii=False))
    return llm_output


def main() -> None:
    load_dotenv()
    scenario_id = os.getenv("TEST_SCENARIO_ID", "scenario_4")
    run_test_scenario(scenario_id=scenario_id)


if __name__ == "__main__":
    main()
