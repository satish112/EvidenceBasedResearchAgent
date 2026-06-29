from __future__ import annotations

import argparse
from pathlib import Path

from src.controller import ResearchAssistantController
from src.evaluation import evaluate

ROOT = Path(__file__).parent
DOCS = ROOT / "data" / "sample_docs"
CASES = ROOT / "evaluation" / "eval_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Evidence-Based Research Assistant Agent")
    parser.add_argument("--query", type=str, default="What are the effects of climate change on crop yields?")
    parser.add_argument("--evaluate", action="store_true", help="Run sample evaluation cases")
    args = parser.parse_args()

    controller = ResearchAssistantController(DOCS)

    if args.evaluate:
        df = evaluate(controller, CASES)
        print(df.to_string(index=False))
        print("\nAverage term coverage:", round(df["term_coverage"].mean(), 2))
        print("Average confidence:", round(df["confidence"].mean(), 2))
        return

    result = controller.run(args.query)
    print(result["answer"])
    print("\n--- Plan ---")
    for subq in result.get("plan", {}).get("sub_questions", []):
        print("-", subq)
    print("\n--- Output guardrail ---")
    print(result.get("output_check"))


if __name__ == "__main__":
    main()
