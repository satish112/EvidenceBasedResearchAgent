"""Streamlit UI for the Evidence-Based Research Assistant Agent.

A capstone prototype that visualizes the multi-agent workflow:
Guardrail -> Planner -> Research (TF-IDF retrieval) -> Analyst -> Writer -> Output Guardrail.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.controller import ResearchAssistantController
from src.evaluation import evaluate
from src.retrieval import LocalRetriever

ROOT = Path(__file__).parent
DOCS = ROOT / "data" / "sample_docs"
CASES = ROOT / "evaluation" / "eval_cases.json"

EXAMPLE_QUERIES = [
    "What are the effects of climate change on crop yields?",
    "Why is retrieval important for a research assistant agent?",
    "What safety guardrails should a research agent include?",
]

PIPELINE_STAGES = [
    ("Guardrail", "Screen the request for unsafe or private-data content."),
    ("Planner", "Decompose the question into sub-questions and success criteria."),
    ("Research", "Retrieve the most relevant evidence chunks via TF-IDF similarity."),
    ("Analyst", "Score candidate reasoning branches and estimate confidence."),
    ("Writer", "Synthesize a final, evidence-referenced answer."),
    ("Output Check", "Verify the answer is grounded and references its evidence."),
]


# --------------------------------------------------------------------------- #
# Caching: build the controller / retriever once per session.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_controller() -> ResearchAssistantController:
    return ResearchAssistantController(DOCS)


@st.cache_resource(show_spinner=False)
def get_retriever() -> LocalRetriever:
    return LocalRetriever(DOCS)


@st.cache_data(show_spinner=False)
def run_evaluation() -> pd.DataFrame:
    return evaluate(get_controller(), CASES)


def confidence_label(confidence: float) -> tuple[str, str]:
    """Return a (label, emoji) pair describing a confidence score."""
    if confidence >= 0.7:
        return "High", "🟢"
    if confidence >= 0.35:
        return "Moderate", "🟡"
    return "Low", "🔴"


# --------------------------------------------------------------------------- #
# Page configuration & header.
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Evidence-Based Research Assistant Agent")
st.caption(
    "A capstone prototype using multi-agent coordination, retrieval-augmented "
    "generation, branch-based reasoning, and safety guardrails."
)

# --------------------------------------------------------------------------- #
# Sidebar: controls, architecture, corpus stats.
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Controls")
    top_k = st.slider(
        "Retrieved evidence chunks",
        min_value=1,
        max_value=6,
        value=4,
        help="How many document chunks the Research agent retrieves.",
    )

    st.divider()
    st.subheader("🧩 Agent pipeline")
    for name, desc in PIPELINE_STAGES:
        st.markdown(f"**{name}** — {desc}")

    st.divider()
    st.subheader("📚 Local corpus")
    try:
        retriever = get_retriever()
        sources = sorted({c.source for c in retriever.chunks})
        st.metric("Documents", len(sources))
        st.metric("Indexed chunks", len(retriever.chunks))
        for source in sources:
            st.markdown(f"- `{source}`")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Could not load corpus: {exc}")

    st.divider()
    st.caption(
        "Offline prototype — TF-IDF retrieval, no external API required. "
        "Citations refer to local synthetic documents."
    )


# --------------------------------------------------------------------------- #
# Tabs.
# --------------------------------------------------------------------------- #
tab_research, tab_eval, tab_corpus, tab_about = st.tabs(
    ["🔍 Research", "📊 Evaluation", "📄 Corpus", "ℹ️ About"]
)


# ----------------------------- Research tab -------------------------------- #
with tab_research:
    st.subheader("Ask a research question")

    example = st.selectbox(
        "Start from an example (optional)",
        ["— Type your own —", *EXAMPLE_QUERIES],
    )
    default_query = "" if example == "— Type your own —" else example
    query = st.text_area(
        "Research question",
        value=default_query or EXAMPLE_QUERIES[0],
        height=90,
        label_visibility="collapsed",
        placeholder="e.g. What are the effects of climate change on crop yields?",
    )

    run = st.button("🚀 Run Agent", type="primary", use_container_width=True)

    if run:
        controller = get_controller()
        with st.status("Running multi-agent workflow…", expanded=True) as status:
            st.write("🛡️ Input guardrail check…")
            st.write("🗂️ Planning sub-questions…")
            st.write("🔎 Retrieving evidence (TF-IDF)…")
            st.write("🧠 Analyzing reasoning branches…")
            st.write("✍️ Writing grounded answer…")
            result = controller.run(query, top_k=top_k)
            if result.get("status") == "blocked":
                status.update(label="Request blocked by guardrail", state="error")
            else:
                status.update(label="Workflow complete", state="complete")

        # ---- Blocked by the input guardrail. -------------------------------
        if result.get("status") == "blocked":
            st.error("🚫 Request blocked by the input guardrail.")
            check = result.get("input_check", {})
            st.markdown(f"**Reason:** {check.get('reason', 'Unknown')}")
            if check.get("matched_terms"):
                st.markdown(
                    "**Matched sensitive terms:** "
                    + ", ".join(f"`{t}`" for t in check["matched_terms"])
                )
            st.info(result.get("answer", ""))
            st.stop()

        analysis = result["analysis"]
        output_check = result["output_check"]
        evidence = result["evidence"]
        confidence = analysis.get("confidence", 0.0)
        label, emoji = confidence_label(confidence)

        # ---- Top-line metrics. ---------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{confidence:.2f}", help="Mean reasoning-branch score.")
        m2.metric("Confidence level", f"{emoji} {label}")
        m3.metric("Evidence chunks", len(evidence))
        m4.metric(
            "Output guardrail",
            "✅ Passed" if output_check.get("passed") else "⚠️ Flagged",
        )

        st.progress(min(1.0, max(0.0, confidence)), text=f"Confidence: {confidence:.0%}")

        if analysis.get("needs_human_review"):
            st.warning(
                f"⚠️ **Human review recommended:** {analysis.get('review_reason')}"
            )

        # ---- Final answer. -------------------------------------------------
        st.subheader("📝 Final Answer")
        st.markdown(result["answer"].replace("\n", "  \n"))

        st.divider()

        # ---- Plan + Analysis side by side. ---------------------------------
        col_plan, col_analysis = st.columns(2)

        with col_plan:
            st.subheader("🗂️ Research Plan")
            plan = result["plan"]
            st.markdown("**Sub-questions**")
            for i, sub_q in enumerate(plan.get("sub_questions", []), start=1):
                st.markdown(f"{i}. {sub_q}")
            st.markdown("**Success criteria**")
            for crit in plan.get("success_criteria", []):
                st.markdown(f"- {crit}")

        with col_analysis:
            st.subheader("🧠 Analysis")
            branches = analysis.get("candidate_branches", [])
            if branches:
                branch_df = pd.DataFrame(branches)
                chart_df = branch_df.set_index("branch_id")[["retrieval_score", "score"]]
                st.markdown("**Candidate reasoning branch scores**")
                st.bar_chart(chart_df)
                selected = analysis.get("selected_branch", {})
                st.success(
                    f"**Selected branch #{selected.get('branch_id', '?')}** "
                    f"(score {selected.get('score', 0):.3f}) — "
                    f"source `{selected.get('source', 'n/a')}`"
                )
            else:
                st.info("No reasoning branches were produced (no evidence).")

        st.divider()

        # ---- Retrieved evidence. -------------------------------------------
        st.subheader("📎 Retrieved Evidence")
        if not evidence:
            st.info("No evidence cleared the relevance threshold for this query.")
        for i, item in enumerate(evidence, start=1):
            with st.expander(
                f"Evidence {i} · `{item['source']}` · chunk {item['chunk_id']} "
                f"· score {item['score']:.3f}",
                expanded=(i == 1),
            ):
                st.progress(
                    min(1.0, item["score"]),
                    text=f"Relevance score: {item['score']:.3f}",
                )
                st.write(item["text"])

        st.divider()

        # ---- Guardrail detail + raw trace. ---------------------------------
        col_guard, col_raw = st.columns(2)
        with col_guard:
            st.subheader("🛡️ Guardrail Checks")
            st.markdown("**Input check**")
            st.json(result.get("input_check", {}))
            st.markdown("**Output check**")
            if output_check.get("passed"):
                st.success("All output guardrails passed.")
            else:
                for issue in output_check.get("issues", []):
                    st.warning(f"- {issue}")
        with col_raw:
            st.subheader("🔧 Raw trace")
            st.caption("Full structured result returned by the controller.")
            st.json(result)


# ---------------------------- Evaluation tab ------------------------------- #
with tab_eval:
    st.subheader("📊 Evaluation on sample cases")
    st.caption(
        f"Runs the agent over the test cases in `{CASES.name}` and reports "
        "term coverage, evidence usage, and confidence."
    )
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating sample cases…"):
            df = run_evaluation()

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg term coverage", f"{df['term_coverage'].mean():.2f}")
        c2.metric("Avg confidence", f"{df['confidence'].mean():.2f}")
        c3.metric(
            "Guardrail pass rate",
            f"{df['output_guardrail_passed'].mean():.0%}",
        )

        st.dataframe(df, use_container_width=True)

        st.markdown("**Term coverage by query**")
        st.bar_chart(df.set_index("query")[["term_coverage", "confidence"]])
    else:
        st.info("Click **Run evaluation** to score the sample test cases.")


# ------------------------------ Corpus tab --------------------------------- #
with tab_corpus:
    st.subheader("📄 Local document corpus")
    st.caption("Synthetic, public-style documents used as the evidence base.")
    doc_files = sorted(DOCS.glob("*.txt"))
    if not doc_files:
        st.warning(f"No `.txt` documents found in {DOCS}")
    for doc in doc_files:
        with st.expander(f"`{doc.name}`"):
            st.text(doc.read_text(encoding="utf-8"))


# ------------------------------- About tab --------------------------------- #
with tab_about:
    st.subheader("ℹ️ About this project")
    st.markdown(
        """
This prototype demonstrates an **autonomous, evidence-based research assistant**
built from cooperating agents rather than a single LLM prompt.

**Why multi-agent?** A prompt-only model can hallucinate, miss contradictions, or
make unsupported claims. Splitting the work into specialized agents — with
retrieval grounding and explicit guardrails — makes the reasoning *transparent*
and *checkable*.

**Workflow**

```
User Query
   → 🛡️ Input Guardrail (block unsafe / private-data requests)
   → 🗂️ Planner        (decompose into sub-questions)
   → 🔎 Research        (TF-IDF retrieval over local corpus)
   → 🧠 Analyst         (score reasoning branches, estimate confidence)
   → ✍️ Writer          (synthesize evidence-referenced answer)
   → 🛡️ Output Guardrail (require grounding + evidence references)
   → ✅ Final Answer
```

**Safety guardrails**
- Rejects private / confidential / sensitive-data requests.
- Retrieves only from the local allowed corpus.
- Requires evidence references in every answer.
- Flags low-confidence answers for human review.

**Stack:** Python · scikit-learn (TF-IDF) · pandas · Streamlit — fully offline,
no paid API required. Embeddings + FAISS, live web search, and an LLM critic are
documented as future improvements in the README.
        """
    )
