# Evidence-Based Research Assistant Agent

A lightweight capstone prototype for an autonomous Research Assistant Agent. The system uses a multi-agent workflow, retrieval-augmented generation concepts, Tree-of-Thought-inspired branch evaluation, guardrails, and simple evaluation metrics to produce evidence-backed research answers from public, synthetic, or anonymized documents.

## Problem

Research users often need more than a single LLM response. They need the system to gather evidence, compare sources, handle uncertainty, and explain conclusions transparently. A prompt-only LLM can hallucinate, miss contradictions, or produce unsupported claims.

## Intended Users

- Students preparing literature reviews or reports
- Analysts comparing public reports and documents
- Researchers who need evidence-backed summaries
- Knowledge workers reviewing public information

## Architecture

The workflow contains four primary agents plus a guardrail layer:

1. **Planner Agent**: decomposes the user query into sub-questions and research goals.
2. **Research Agent**: retrieves relevant document chunks using TF-IDF semantic-style similarity over a local corpus.
3. **Analyst Agent**: evaluates retrieved evidence, detects support/limitations, and creates candidate reasoning branches.
4. **Writer Agent**: synthesizes a final answer with evidence references.
5. **Guardrail Agent**: blocks unsafe/private-data requests and checks that the answer contains evidence references.

The controller coordinates agents using a hybrid workflow:

`User Query -> Guardrail Check -> Planner -> Retriever -> Analyst -> Writer -> Output Guardrail -> Final Answer`

## Current Implementation

This prototype is intentionally simple and reproducible. It does not require a paid LLM API. It uses:

- Python 3.10+
- scikit-learn for TF-IDF retrieval
- pandas for evaluation output
- Streamlit for optional UI

An OpenAI, Anthropic, or local Ollama model can be added later, but the included version runs offline using local sample documents.

## Repository Structure

```text
research_assistant_agent/
├── app.py                       # Optional Streamlit UI
├── run_demo.py                  # CLI demo
├── requirements.txt             # Python dependencies
├── src/
│   ├── agents.py                # Planner, Research, Analyst, Writer, Guardrail agents
│   ├── controller.py            # End-to-end orchestration
│   ├── retrieval.py             # Local document loading and retrieval
│   └── evaluation.py            # Simple evaluation metrics
├── data/sample_docs/            # Synthetic/public-style sample research documents
├── evaluation/eval_cases.json   # Sample test cases
└── notebooks/demo.ipynb         # Lightweight notebook demo
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run CLI Demo

```bash
python run_demo.py --query "What are the effects of climate change on crop yields?"
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## Static Browser Version (no install, deploy anywhere)

A fully self-contained, client-side version lives in [`docs/index.html`](docs/index.html).
It re-implements the entire agent pipeline — TF-IDF retrieval, planner, analyst,
writer, and guardrails — in plain HTML/CSS/JavaScript with **no server, no build
step, and no API key**. The retrieval logic mirrors the Python `scikit-learn`
implementation (smooth IDF, L2 normalization, identical tokenizer and chunking),
so its scores match the Python prototype exactly.

Try it locally by simply opening the file in a browser:

```bash
open docs/index.html        # macOS  (or just double-click it)
```

### Deploy free on GitHub Pages

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Select branch `main` and folder **`/docs`**, then **Save**.
5. Your app will be live at `https://<your-username>.github.io/<repo-name>/`.

The same `docs/index.html` also works on Netlify, Vercel, Cloudflare Pages, or any
static file host — just point it at the `docs/` folder.

## Run Evaluation

```bash
python run_demo.py --evaluate
```

## Sample Output

The system returns:

- decomposed research plan
- retrieved evidence chunks
- analysis notes
- final evidence-backed answer
- safety and groundedness checks

## Safety and Guardrails

The system includes:

- rejection of private/confidential data requests
- retrieval from local allowed corpus only
- output requirement for evidence references
- escalation flag when confidence is low or evidence is insufficient
- logging of intermediate decisions for review

## Limitations

- Retrieval is local and small-scale in this prototype.
- The reasoning branch scoring is heuristic, not a full LLM-based critic.
- The system does not yet crawl live web sources.
- Citations refer to local sample documents rather than external URLs.
- Evaluation is based on small synthetic test cases.

## Future Improvements

- Add live public web search with source filtering.
- Add embeddings with FAISS or Chroma.
- Add an LLM-powered critic and answer generator.
- Add citation validation and quote-level attribution.
- Expand evaluation with larger benchmark questions.
- Deploy as a hosted Streamlit or FastAPI application.

## GitHub Repository

`https://github.com/satish112/EvidenceBasedResearchAgent`
