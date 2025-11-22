# 📘 Research Agent v2

A lightweight research assistant that finds, ranks, and explains recent AI papers from arXiv.

Now supports both OpenAI models and a free local LLM backend.

This is the v2 Streamlit UI application.  
Backend for the free model option lives in a separate repo.

## 🚀 What's New in v2

Research Agent v2 introduces a dual-mode architecture:

### 1. OpenAI Mode

For users with an OpenAI API key.  
Provides the full experience:

- LLM-based relevance classification
- Natural-language citation prediction
- Plain English paper summaries
- More accurate and more detailed results

### 2. Free Local Model Mode

For users who want a zero-cost option.

Powered by a separate backend server using:

- Phi-2 as the hosted LLM
- MiniLM-L6-v2 as the embedding model

This mode:

- Does not require any API key
- Uses heuristic relevance scoring
- Uses scaled citation scores instead of LLM reasoning
- Skips plain English summaries to keep the backend fast
- Still fetches and ranks up to 150 arXiv papers

Both modes run through the same pipeline, differing only in how classification and prediction are computed.

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
- Classifies papers
- Builds a prediction set (~20 minimum)
- Predicts 1-year citation impact
- Ranks papers
- Highlights the top N papers

### 3. You receive

- A full ranked table
- A clean UI with metadata and links
- A summary for top N papers (OpenAI mode only)
- A ZIP archive with all artifacts (JSON + markdown report)

## 💻 Running Locally (UI Only)

**Clone:**

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

OpenAI mode enables full classification, full citation prediction, and plain English summaries.

Your API key is never written to disk.

### Option B: Free Local Model (Default)

No key required.

To use the free local LLM server:

**Clone the backend repo:**
```
https://github.com/YOUR_USER/arxiv-agent-backend
```

**Install backend dependencies**

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

Run the UI and choose **Free Hosted Model** in the sidebar.

The UI will call:

- `POST /chat`
- `POST /embeddings`

using:

```python
FREE_LLM_API_BASE = http://localhost:8000
```

(You can override this with an environment variable if deploying the backend elsewhere.)

## 🧩 Backend Architecture (Free Model)

The backend exposes two endpoints:

**POST /chat**

- Uses Phi-2 via Transformers
- Performs greedy decoding for speed
- Max 200 new tokens

**POST /embeddings**

- Uses MiniLM-L6-v2 (SentenceTransformers)
- Returns batched embedding vectors

**GET /health**

- Health check endpoint for Streamlit.

Backend is CPU-only by default for stability across devices.

## ⚠️ Limitations

- Citation predictions are signals, not ground truth
- Free backend mode uses heuristics, not LLM reasoning
- Academic bias may appear in LLM predictions
- Never use these scores alone for hiring or funding decisions
- Always read the papers before drawing conclusions

## 📄 License

MIT License (see LICENSE file)
