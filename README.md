# Arxiv.org Agent v2

An LLM powered research assistant that finds, ranks, and explains recent AI papers on arxiv.org.

This version adds a **free hosted LLM option** in addition to the **OpenAI** option.

## ▶️ Demo (v2)

v2 will live at a new Streamlit URL once deployed.

---

## ✨ What is new in v2

- **Two LLM providers:**
  - **Free hosted model**  
    Uses an open weight LLM and embedding model served by a separate backend API.  
    No OpenAI account or API key needed.
  - **OpenAI**  
    Uses your own OpenAI API key for classification, citation prediction, and summaries.

- **In both modes:**
  - Fetches recent `cs.AI` and `cs.LG` papers from arxiv.org
  - Uses embeddings to pick up to 150 candidate papers for your brief
  - Builds a prediction set and estimates 1 year citation impact
  - Ranks and highlights top N papers

- **In OpenAI mode:**
  - Full LLM based relevance classification (primary / secondary / off topic)
  - LLM based citation predictions with natural language explanations
  - Plain English summaries of each top paper

- **In Free hosted mode:**
  - Embedding based candidate selection
  - Simple heuristic relevance classification
  - Citation predictions that fall back to a score based heuristic when needed
  - Plain English summaries are skipped to keep the backend light

---

## 🧠 How it works (pipeline)

1. **You provide**
   - A short research brief
   - Optional "not looking for" text
   - Date range (3 days, 7 days, 30 days)

2. **The agent**
   - Fetches recent `cs.AI` + `cs.LG` papers
   - Uses embeddings to select up to 150 candidates that match your brief
   - Labels them as primary, secondary, or off topic
   - Builds a prediction set with at least ~20 papers when possible
   - Predicts 1 year citation counts with an LLM or a score based fallback
   - Ranks papers by predicted citations

3. **You get**
   - Ranked table of all selected papers
   - Highlighted top N papers
   - Links to arXiv and PDFs
   - In OpenAI mode, plain English summaries and explanations
   - A downloadable ZIP with all JSON artifacts and a markdown report

---

## 🔧 Running locally (UI only)

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/YOUR_USER/arxiv-ai-agent-v2.git
cd arxiv-ai-agent-v2

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Then start the Streamlit app:

```bash
streamlit run app.py
```

The terminal will show a local URL like:

```
http://localhost:8501
```

---

## Free hosted model vs OpenAI

### Free hosted model

- No API key required.
- The app calls a backend API, configured by:
  - `FREE_LLM_API_BASE` environment variable (default `http://localhost:8000`)
- The backend is implemented in a separate repo (see `arxiv-agent-backend`).

### OpenAI

- Select OpenAI in the sidebar.
- Enter your OpenAI API key in the sidebar.
- The key is kept in memory only for the session and never written to disk.

---

## Backend (free hosted model)

The backend server that powers the free hosted option lives in a separate project, for example:

- https://github.com/YOUR_USER/arxiv-agent-backend

It exposes two endpoints:

- `POST /chat` for LLM calls
- `POST /embeddings` for embeddings

See that repo for details on running it with FastAPI and Uvicorn.

---

## ⚠️ Limitations and ethics

- Citation predictions are approximate signals, not ground truth.
- The model may reflect existing academic biases and trends.
- Do not use these scores alone for hiring, promotion, funding, or formal evaluation.
- Always read the actual papers before relying on any ranking.

