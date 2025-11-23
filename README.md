# 📘 Research Agent v2

A lightweight research assistant that finds, ranks, and explains recent AI papers from arXiv.

Now supports both OpenAI models and a free local LLM backend.

This is the v2 Streamlit UI application.  
The backend for the free model option lives in a separate repo.

## 🚀 What's New in v2

Research Agent v2 introduces a dual-mode architecture:

### 1. OpenAI Mode

For users with an OpenAI API key.  
Provides the full experience:

- LLM-based relevance classification
- Natural-language citation impact scoring
- Plain English paper summaries
- More informative and nuanced ranking signals

### 2. Free Local Model Mode

For users who want a zero-cost option.

Powered by a separate backend server using:

- Phi-2 as the hosted LLM
- MiniLM-L6-v2 as the embedding model

This mode:

- Does not require any API key
- Uses heuristic relevance scoring
- Uses scaled citation impact scores instead of LLM reasoning about impact
- Skips plain English summaries to keep the backend fast
- Still fetches and ranks up to 150 arXiv papers

Both modes run through the same pipeline, differing only in how relevance and citation impact scores are computed.

## 🧠 How Research Agent Works

Whether you use OpenAI or the free backend, the pipeline is:

### 1. You provide

- A short research brief
- Optional "not looking for" text
- A date range (3, 7, or 30 days)

### 2. The agent

- Fetches recent cs.AI + cs.LG papers from arXiv
- Computes embeddings
- Selects up to 150 candidate papers
- Classifies papers as primary, secondary, or off topic
- Builds a citation scoring set (targeting ~20+ papers)
- Assigns 1 year citation impact scores
- Ranks papers by citation impact score
- Highlights the top N papers

### 3. You receive

- A full ranked table
- A clean UI with metadata and links
- A summary for top N papers (OpenAI mode only)
- A ZIP archive with all artifacts (JSON + markdown report)

## 💻 Running Locally (UI Only)

**Clone:**

```bash
git clone https://github.com/nurtekinsavasai/arxiv-ai-agent-v2.git
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

Your browser will open:

```
http://localhost:8501
```

## 🔌 Choosing a Model Provider

### Option A: OpenAI

In the sidebar:

1. Select **OpenAI**
2. Enter your API key
3. Run the pipeline

OpenAI mode enables:

- Full LLM-based relevance classification
- LLM-based citation impact scores with natural language explanations
- Plain English summaries of the top N papers

Your API key is never written to disk.

### Option B: Free Local Model (Default)

No key required.

To use the free local LLM server:

**Clone the backend repo:**

```bash
git clone https://github.com/nurtekinsavasai/arxiv-agent-backend.git
```

**Install backend dependencies:**

```bash
cd arxiv-agent-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Start the backend:**

```bash
python server.py
```

You should see:

```
Uvicorn running on http://0.0.0.0:8000
```

Now run the UI and choose **Free local model (no API key)** in the sidebar.

The UI will call:

- `POST /chat`
- `POST /embeddings`

using:

```python
FREE_LLM_API_BASE = "http://localhost:8000"
```

(You can override this with an environment variable if deploying the backend elsewhere.)

## 🧩 Backend Architecture (Free Model)

The backend exposes three endpoints:

### `POST /chat`

- Uses Phi-2 via Transformers
- Performs greedy decoding for speed
- Max 200 new tokens

### `POST /embeddings`

- Uses MiniLM-L6-v2 (SentenceTransformers)
- Returns batched embedding vectors

### `GET /health`

- Health check endpoint for the UI.

The backend is CPU-only by default for stability across devices.

## ⚠️ Limitations

- Citation impact scores are heuristic ranking signals, not ground truth.
- Free backend mode uses heuristics; OpenAI mode uses LLM judgment, which can still reflect academic and social biases.
- Never use these scores alone for hiring, funding, promotion, or other high stakes decisions.
- Always read the actual papers before drawing conclusions or making decisions.
