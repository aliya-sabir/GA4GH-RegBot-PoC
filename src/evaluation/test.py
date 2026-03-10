import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class Citation(BaseModel):
    citation: str = Field(description="Document name and section number, e.g. 'GA4GH Framework - Section 2.1'")
    title: str = Field(description="Title of the cited clause")
    excerpt: str = Field(description="Relevant text excerpt from the clause")


class ExpectedOutput(BaseModel):
    status: str = Field(description="Compliant | Partial | Non-Compliant")
    missing_elements: list[str] = Field(description="List of missing regulatory requirements")
    suggested_fix: str = Field(description="Recommendation to make the form compliant")
    citations: list[Citation] = Field(description="GA4GH clause references")


class TestScenario(BaseModel):
    id: str = Field(description="Scenario identifier")
    consent_form: str = Field(description="Synthetic consent form text")
    expected_output: ExpectedOutput = Field(description="Expected compliance analysis output")
    questions: list[str] = Field(description="Evaluation questions for RAG retrieval testing")


def load_tests() -> list[TestScenario]:
    #load test scenarios from JSONL file
    tests = []
    with open (TEST_FILE, "r", encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestScenario(**data))
    return tests