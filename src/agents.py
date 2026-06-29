from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .retrieval import LocalRetriever, RetrievedChunk


@dataclass
class ResearchPlan:
    original_query: str
    sub_questions: List[str]
    success_criteria: List[str]


@dataclass
class AnalysisResult:
    candidate_branches: List[Dict]
    selected_branch: Dict
    confidence: float
    needs_human_review: bool
    review_reason: str | None = None


class GuardrailAgent:
    blocked_terms = {
        "private email",
        "confidential",
        "proprietary",
        "ssn",
        "social security",
        "password",
        "medical record",
        "client data",
    }

    def check_input(self, query: str) -> Dict:
        lower = query.lower()
        matched = [term for term in self.blocked_terms if term in lower]
        if matched:
            return {
                "allowed": False,
                "reason": "The request appears to involve private, confidential, or sensitive data.",
                "matched_terms": matched,
            }
        if len(query.strip()) < 10:
            return {"allowed": False, "reason": "The query is too short or ambiguous.", "matched_terms": []}
        return {"allowed": True, "reason": None, "matched_terms": []}

    def check_output(self, answer: str, evidence_count: int, confidence: float) -> Dict:
        issues = []
        if evidence_count == 0:
            issues.append("No retrieved evidence was used.")
        if "Evidence references" not in answer:
            issues.append("Final answer does not include evidence references.")
        if confidence < 0.35:
            issues.append("Low confidence answer requires human review.")
        return {"passed": len(issues) == 0, "issues": issues}


class PlannerAgent:
    def plan(self, query: str) -> ResearchPlan:
        base = query.strip().rstrip("?")
        sub_questions = [
            f"What are the most relevant facts for: {base}?",
            f"What evidence supports or limits conclusions about: {base}?",
            f"Are there contradictions, uncertainty, or boundary conditions for: {base}?",
        ]
        success_criteria = [
            "answer is grounded in retrieved evidence",
            "key uncertainty or limitations are stated",
            "final response includes evidence references",
        ]
        return ResearchPlan(original_query=query, sub_questions=sub_questions, success_criteria=success_criteria)


class ResearchAgent:
    def __init__(self, retriever: LocalRetriever):
        self.retriever = retriever

    def retrieve(self, plan: ResearchPlan, top_k: int = 4) -> List[RetrievedChunk]:
        combined_query = plan.original_query + " " + " ".join(plan.sub_questions)
        return self.retriever.search(combined_query, top_k=top_k)


class AnalystAgent:
    def analyze(self, query: str, evidence: List[RetrievedChunk]) -> AnalysisResult:
        if not evidence:
            return AnalysisResult(
                candidate_branches=[],
                selected_branch={"claim": "Insufficient evidence", "support": [], "score": 0.0},
                confidence=0.0,
                needs_human_review=True,
                review_reason="No relevant evidence was retrieved.",
            )

        branches = []
        for idx, item in enumerate(evidence[:3], start=1):
            text = item.text
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 35]
            claim = sentences[0] if sentences else text[:220]
            score = min(1.0, item.score + 0.15 * len(sentences))
            branches.append(
                {
                    "branch_id": idx,
                    "claim": claim,
                    "source": item.source,
                    "chunk_id": item.chunk_id,
                    "retrieval_score": round(item.score, 3),
                    "score": round(score, 3),
                }
            )

        selected = max(branches, key=lambda b: b["score"])
        confidence = round(sum(b["score"] for b in branches) / len(branches), 3)
        needs_review = confidence < 0.35 or len(evidence) < 2
        reason = None
        if needs_review:
            reason = "Evidence is limited or confidence is below the review threshold."

        return AnalysisResult(
            candidate_branches=branches,
            selected_branch=selected,
            confidence=confidence,
            needs_human_review=needs_review,
            review_reason=reason,
        )


class WriterAgent:
    def write(self, query: str, plan: ResearchPlan, evidence: List[RetrievedChunk], analysis: AnalysisResult) -> str:
        if not evidence:
            return (
                "I could not produce a fully grounded answer because no relevant evidence was retrieved. "
                "Human review or additional source material is required.\n\nEvidence references: none"
            )

        evidence_points = []
        for item in evidence[:4]:
            snippet = item.text.replace("\n", " ")[:280]
            evidence_points.append(f"- [{item.source} | chunk {item.chunk_id} | score {item.score:.2f}] {snippet}...")

        uncertainty = ""
        if analysis.needs_human_review:
            uncertainty = f"\n\nHuman review flag: {analysis.review_reason}"

        answer = f"""Research question: {query}

Summary answer:
Based on the retrieved evidence, the strongest supported conclusion is: {analysis.selected_branch.get('claim', 'No selected claim')}.

Reasoning summary:
The system decomposed the query into sub-questions, retrieved the most relevant document chunks, evaluated candidate reasoning branches, and selected the branch with the strongest evidence score. The answer should be interpreted within the scope of the local evidence corpus.

Evidence references:
{chr(10).join(evidence_points)}

Confidence score: {analysis.confidence:.2f}{uncertainty}
"""
        return answer
