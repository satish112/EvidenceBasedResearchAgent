from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .agents import AnalystAgent, GuardrailAgent, PlannerAgent, ResearchAgent, WriterAgent
from .retrieval import LocalRetriever


class ResearchAssistantController:
    def __init__(self, docs_path: str | Path):
        self.guardrail = GuardrailAgent()
        self.planner = PlannerAgent()
        self.retriever = LocalRetriever(docs_path)
        self.researcher = ResearchAgent(self.retriever)
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

    def run(self, query: str, top_k: int = 4) -> Dict:
        input_check = self.guardrail.check_input(query)
        if not input_check["allowed"]:
            return {
                "status": "blocked",
                "input_check": input_check,
                "answer": f"Request blocked: {input_check['reason']}",
            }

        plan = self.planner.plan(query)
        evidence = self.researcher.retrieve(plan, top_k=top_k)
        analysis = self.analyst.analyze(query, evidence)
        answer = self.writer.write(query, plan, evidence, analysis)
        output_check = self.guardrail.check_output(answer, len(evidence), analysis.confidence)

        return {
            "status": "completed",
            "input_check": input_check,
            "plan": asdict(plan),
            "evidence": [e.__dict__ for e in evidence],
            "analysis": asdict(analysis),
            "output_check": output_check,
            "answer": answer,
        }
