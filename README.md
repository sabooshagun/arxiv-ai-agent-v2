# 📘 Research Agent v2

A lightweight research assistant that finds, ranks, and analyzes recent AI papers from arXiv.

Research Agent v2 introduces two modes:

- **OpenAI mode** — full classification, prediction, and summaries
- **Free local mode** — fully free and self-contained using local embeddings + heuristics

There is no backend server in v2.  
Everything runs inside the Streamlit app.

## 🚀 What's New in v2

### 1. OpenAI Mode (Full Features)

If you have an OpenAI API key, you get:

- LLM-based relevance classification
- Natural-language citation prediction
- Plain-English summaries for top papers
- More accurate signal quality
- Consistent explanations

This mode mirrors the original v1 workflow but with improved pipeline logic.

### 2. Free Local Model Mode (Zero Cost)

Runs entirely inside the Streamlit app using:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Heuristic rules for:
  - Primary vs secondary classification
  - Citation scoring
- No generative LLM needed
- No API key needed
- Skips plain-English summaries (to keep it fast and lightweight)

This mode still:

- Fetches up to 150 recent arXiv papers
- Selects and classifies candidates
- Builds a citation scoring set
- Produces ranked results

Designed for users who want a completely free and private pipeline.

## 🧠 How the Pipeline Works

Regardless of mode:

### 1. You provide

- A research brief
- Optional exclusions
- A date range (3, 7, or 30 days)

### 2. The agent

- Fetches recent cs.AI + cs.LG papers
- Computes embeddings (local or OpenAI)
- Selects up to 150 matching candidates
- Assigns relevance labels
- Builds a prediction set
- Produces citation predictions
  - LLM-based in OpenAI mode
  - Heuristic-based in free mode
- Ranks papers
- Highlights the top N

### 3. You receive

- A ranked table with metadata and links
- Optional plain-English summaries (OpenAI mode only)
- Optional citation explanation text (OpenAI mode only)
- A downloadable ZIP of all reports and JSON artifacts

## 💻 Run Locally (Recommended)

**Clone the repo:**

```bash
git clone https://github.com/YOUR_USER/arxiv-ai-agent-v2.git
cd arxiv-ai-agent-v2
```

**Create a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the UI:**

```bash
streamlit run app.py
```

You will see:

```
http://localhost:8501
```

## 🔌 Model Provider Options

### Option A — OpenAI (Full experience)

- Select **OpenAI** in the sidebar
- Enter your API key (never saved locally)
- Adds full LLM reasoning:
  - Primary/secondary/off-topic classification
  - Citation prediction paragraphs
  - Plain English summaries

### Option B — Free Local Mode (Default)

- Select **Free Local Model**
- No API key required
- Everything runs inside Streamlit using:
  - MiniLM embeddings
  - Heuristic relevance scores
  - Heuristic citation scoring
- Summaries and explanations are disabled in this mode (OpenAI-only features).

## 🧩 Architecture Overview

### Free Local Mode (self-contained)

- Loads MiniLM embeddings via `sentence-transformers`
- Computes similarity-based relevance
- Computes heuristic citation scores
- No HTTP requests, no backend, no external APIs

### OpenAI Mode

- Uses OpenAI embeddings
- Uses OpenAI for classification, prediction, summaries
- Requires API key
- Everything else (candidate selection, ranking, exporting) is shared.

## ⚠️ Limitations

- Citation predictions are approximate signals
- Free local mode offers heuristic predictions, not LLM reasoning
- Academic bias may appear in model-based predictions
- Do not use these scores as the sole basis for evaluation of researchers or grants
- Always read the actual papers

## 📄 License

MIT License — see LICENSE.
