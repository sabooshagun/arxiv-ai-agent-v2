import requests
import feedparser
import time
import json
import os
import math
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_LLM = "gpt-5.2"
OUTPUT_WEIGHTS = "moneyball_weights.json"
OUTPUT_LOGS = "training_logs.csv"

# --- DYNAMIC DATE GENERATION ---
def get_month_bounds(months_ago):
    """Calculates start and end datetime for a specific month in the past."""
    today = datetime.now()
    
    # Calculate target year and month handling rollovers
    target_month = today.month - months_ago
    target_year = today.year
    
    while target_month <= 0:
        target_month += 12
        target_year -= 1
        
    start_date = datetime(target_year, target_month, 1)
    
    # Calculate end of that month (start of next month - 1 second)
    if target_month == 12:
        next_month = datetime(target_year + 1, 1, 1)
    else:
        next_month = datetime(target_year, target_month + 1, 1)
        
    end_date = next_month - timedelta(seconds=1)
    
    return start_date, end_date

def generate_date_configs():
    """Generates ArXiv query strings for the sliding window."""
    configs = {"train": [], "test": []}
    
    # Training Window: 14 months ago and 13 months ago
    for months_back in [14, 13]:
        start, end = get_month_bounds(months_back)
        label = start.strftime("%Y_%b").upper() # e.g. 2024_NOV
        
        # ArXiv format: YYYYMMDDHHMM
        q_start = start.strftime("%Y%m%d0000")
        q_end = end.strftime("%Y%m%d2359")
        
        configs["train"].append({
            "label": f"TRAIN_{label}",
            "query": f"submittedDate:[{q_start} TO {q_end}]"
        })

    # Testing Window: 12 months ago and 11 months ago
    for months_back in [12, 11]:
        start, end = get_month_bounds(months_back)
        label = start.strftime("%Y_%b").upper()
        
        q_start = start.strftime("%Y%m%d0000")
        q_end = end.strftime("%Y%m%d2359")
        
        configs["test"].append({
            "label": f"TEST_{label}",
            "query": f"submittedDate:[{q_start} TO {q_end}]"
        })
        
    return configs

# --- PART 1: DATA ACQUISITION ---
def fetch_enriched_papers(configs, max_per_month=75):
    """Pulls ArXiv papers and enriches with S2 citation data."""
    api_key = os.getenv("S2_API_KEY")
    base_url = "http://export.arxiv.org/api/query"
    s2_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    all_papers = []
    
    # Combine lists for processing
    batch_list = configs["train"] + configs["test"]
    
    print(f"📡  [1/4] Polling ArXiv & Semantic Scholar ({len(batch_list)} batches)...")
    
    for conf in batch_list:
        print(f"    Batch: {conf['label']}...", end="", flush=True)
        papers_in_batch = 0
        start = 0
        
        full_query = f"(cat:cs.AI OR cat:cs.LG) AND {conf['query']}"
        
        while papers_in_batch < max_per_month:
            try:
                resp = requests.get(base_url, params={
                    "search_query": full_query, "start": start, "max_results": 50, 
                    "sortBy": "submittedDate", "sortOrder": "descending"
                }, timeout=10)
                feed = feedparser.parse(resp.content)
                if not feed.entries: break
                
                for entry in feed.entries:
                    if papers_in_batch >= max_per_month: break
                    
                    title = entry.title.replace("\n", " ").strip()
                    
                    # S2 Enrichment
                    meta = None
                    try:
                        s2_params = {"query": title, "limit": 1, "fields": "title,citationCount,authors.citationCount"}
                        headers = {"x-api-key": api_key} if api_key else {}
                        r = requests.get(s2_url, headers=headers, params=s2_params, timeout=3)
                        if r.status_code == 200:
                            data = r.json()
                            if data.get('data'): meta = data['data'][0]
                    except: pass
                    
                    # Only keep if we have ground truth citations
                    if meta and "citationCount" in meta:
                        auth_cites = [a.get('citationCount', 0) for a in meta.get('authors', []) if a.get('citationCount')]
                        
                        all_papers.append({
                            "dataset": conf['label'],
                            "title": title,
                            "abstract": entry.summary.replace("\n", " ").strip(),
                            "actual_citations": meta.get('citationCount', 0),
                            "max_author_citations": max(auth_cites) if auth_cites else 0
                        })
                        papers_in_batch += 1
                        
                    if not api_key: time.sleep(0.2)
                
                start += 50
            except Exception as e:
                print(f"Err: {e}")
                break
        print(f" Done (+{papers_in_batch})")
        
    return pd.DataFrame(all_papers)

# --- PART 2: FEATURE EXTRACTION ---
def extract_features(df):
    """Calculates H1 (Fame), H2 (Hype), H3 (Sniper), H4 (Utility)."""
    print(f"🧠  [2/4] Extracting Features for {len(df)} papers...")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Pre-calculate simple heuristics
    df['h1_fame'] = df['max_author_citations'].apply(lambda x: min(math.log(x + 1) * 8, 95))
    
    def calc_hype(t):
        t = t.lower()
        s = 0
        if "benchmark" in t or "dataset" in t: s += 50
        if "survey" in t: s += 40
        if "llm" in t: s += 10
        return s
    df['h2_hype'] = df['title'].apply(calc_hype)
    
    def calc_sniper(t):
        t = t.lower()
        s = 0
        if "benchmark" in t: s += 50
        niche = ["lidar", "3d", "audio", "wireless", "agriculture", "traffic", "physics"]
        if any(n in t for n in niche): s -= 20
        return s
    df['h3_sniper'] = df['title'].apply(calc_sniper)
    
    # LLM Call for H4 (Blind Utility)
    print("    Running LLM analysis (Blind Utility)...")
    utilities = []
    for i, row in df.iterrows():
        if i % 10 == 0: print(f"    Processing {i}/{len(df)}...", end="\r")
        
        prompt = (
            "Analyze this abstract BLINDLY (ignore authors).\n"
            "Rate 'Citation Potential' from 0-10 based on:\n"
            "1. **Broad Utility:** (Benchmarks/GenAI) = High.\n"
            "2. **Niche:** (Agri/Physics) = Low.\n"
            "3. **Saturation:** (New Architectures) = Mid-Low.\n\n"
            f"Title: {row['title']}\n"
            f"Abstract: {row['abstract'][:500]}...\n\n"
            "Return JSON: { \"score\": <int> }"
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL_LLM, messages=[{"role": "user", "content": prompt}], temperature=0.0
            )
            content = resp.choices[0].message.content
            if "```" in content:
                content = content.split("```json")[-1].split("```")[0]
            elif "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            
            data = json.loads(content)
            utilities.append(data.get("score", 5))
        except:
            utilities.append(5)
    
    df['h4_blind_llm'] = utilities
    print(f"\n    ✅ Features extracted.")
    return df

# --- PART 3: TRAINING ---
def train_moneyball_model(train_df):
    """Trains the MoE model and returns weights."""
    print(f"🌳  [3/4] Training Moneyball Model on {len(train_df)} papers...")
    
    X_cols = ['h1_fame', 'h2_hype', 'h3_sniper', 'h4_blind_llm']
    y_col = 'actual_citations'
    
    # Ensure numeric
    for col in X_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
    
    X_train = train_df[X_cols]
    y_train = np.log1p(train_df[y_col])  # Log transform
    
    # Train
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract weights
    weights = {
        "h1_fame": float(model.feature_importances_[0]),
        "h2_hype": float(model.feature_importances_[1]),
        "h3_sniper": float(model.feature_importances_[2]),
        "h4_blind_llm": float(model.feature_importances_[3])
    }
    
    print(f"    ✅ Model trained. Weights: {weights}")
    return model, weights

# --- PART 4: EVALUATION ---
def evaluate_model(model, test_df):
    """Evaluates on test set and returns Precision@10."""
    print(f"📊  [4/4] Evaluating on {len(test_df)} test papers...")
    
    X_cols = ['h1_fame', 'h2_hype', 'h3_sniper', 'h4_blind_llm']
    y_col = 'actual_citations'
    
    for col in X_cols:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    X_test = test_df[X_cols]
    test_df['predicted'] = model.predict(X_test)
    
    # Precision@10
    threshold = test_df[y_col].quantile(0.8)
    top_10 = test_df.sort_values(by='predicted', ascending=False).head(10)
    hits = top_10[top_10[y_col] >= threshold]
    
    precision = len(hits) / 10
    
    print(f"    Threshold: >={threshold:.1f} citations")
    print(f"    Top 10 Predictions:")
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        icon = "✅" if row[y_col] >= threshold else "❌"
        print(f"    {i}. {icon} Pred:{row['predicted']:.2f} | Act:{row[y_col]} | {row['title'][:40]}...")
    
    print(f"    ✅ Precision@10: {precision:.0%}")
    return precision

# --- MAIN PIPELINE ---
def main():
    print("=" * 70)
    print("🚀 MONEYBALL TRAINING & DEPLOYMENT PIPELINE")
    print("=" * 70)
    
    # Generate dynamic date configs
    date_configs = generate_date_configs()
    print(f"📅 Using sliding window: Training on {len(date_configs['train'])} months, Testing on {len(date_configs['test'])} months")
    
    # 1. Fetch Data
    all_df = fetch_enriched_papers(date_configs)
    
    if len(all_df) == 0:
        print("❌ Insufficient data. Check API keys and date ranges.")
        return
    
    # Split into train and test based on dataset label
    train_df = all_df[all_df['dataset'].str.startswith('TRAIN_')].copy()
    test_df = all_df[all_df['dataset'].str.startswith('TEST_')].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("❌ Insufficient data split. Check date ranges.")
        return
    
    # 2. Extract Features
    train_df = extract_features(train_df)
    test_df = extract_features(test_df)
    
    # 3. Train Model
    model, weights = train_moneyball_model(train_df)
    
    # 4. Evaluate
    precision = evaluate_model(model, test_df)
    
    # 5. Save Artifacts
    print(f"\n💾 Saving artifacts...")
    with open(OUTPUT_WEIGHTS, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"    ✅ Weights saved to {OUTPUT_WEIGHTS}")
    
    # Save training log
    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "precision_at_10": precision,
        "weights": weights
    }
    
    if os.path.exists(OUTPUT_LOGS):
        logs_df = pd.read_csv(OUTPUT_LOGS)
        logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        logs_df = pd.DataFrame([log_entry])
    
    logs_df.to_csv(OUTPUT_LOGS, index=False)
    print(f"    ✅ Training log saved to {OUTPUT_LOGS}")
    
    print(f"\n✅ Pipeline complete! Precision@10: {precision:.0%}")

if __name__ == "__main__":
    main()

