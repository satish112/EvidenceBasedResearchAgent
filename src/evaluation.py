from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .controller import ResearchAssistantController


def evaluate(controller: ResearchAssistantController, cases_path: str | Path) -> pd.DataFrame:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    rows: List[Dict] = []
    for case in cases:
        query = case["query"]
        expected_terms = case.get("expected_terms", [])
        result = controller.run(query)
        answer_lower = result.get("answer", "").lower()
        matched_terms = [term for term in expected_terms if term.lower() in answer_lower]
        term_coverage = len(matched_terms) / max(1, len(expected_terms))
        evidence_count = len(result.get("evidence", []))
        confidence = result.get("analysis", {}).get("confidence", 0.0) if result.get("analysis") else 0.0
        output_passed = result.get("output_check", {}).get("passed", False)
        rows.append(
            {
                "query": query,
                "expected_terms": len(expected_terms),
                "matched_terms": len(matched_terms),
                "term_coverage": round(term_coverage, 2),
                "evidence_count": evidence_count,
                "confidence": confidence,
                "output_guardrail_passed": output_passed,
                "status": result.get("status"),
            }
        )
    return pd.DataFrame(rows)
