# 🧠 Moneyball: The One-Click Retraining Protocol

This documentation explains how to operate the Moneyball Citation Predictor.

The system uses a **Mixture of Experts (MoE)** approach, combining Author Fame, Topic Hype, and Content Utility. It uses Machine Learning to dynamically learn the optimal weights for these factors by analyzing historical data using a **Sliding Window** approach.

## 🚀 Quick Start

### Set Environment Variables:
```bash
export OPENAI_API_KEY="sk-..."
export S2_API_KEY="..." # Optional but recommended for Semantic Scholar
```

### Run the Factory:
```bash
python3 train_and_deploy.py
```

### Check Results:
- **Console Output:** Shows the Precision@10 on unseen test data.
- **`moneyball_weights.json`:** The new "Brain" containing the optimal weights (e.g., Fame: 0.84, Utility: 0.16).
- **`training_logs.csv`:** The full dataset of papers and scores used for training.

## ⚙️ How It Works (Internal Pipeline)

The script `train_and_deploy.py` performs 4 sequential steps automatically. It calculates dates relative to the current date, ensuring the model never goes stale.

### 1. Fetch Data (Sliding Window)

It pulls distinct datasets from ArXiv based on relative months:

- **Training Set:** Papers from 14 months ago and 13 months ago.
  - **Purpose:** To teach the model what a "winner" looks like after ~1 year of maturity.
- **Testing Set:** Papers from 12 months ago and 11 months ago.
  - **Purpose:** To validate the model on a slightly fresher "Unseen" dataset (Out-of-Time validation).

**Note:** It automatically enriches these papers with actual citation counts from Semantic Scholar.

### 2. Extract Features (The 4 Experts)

It calculates 4 scores for every paper:

- **H1 (Fame):** `log(Author Citations)`. Measures distribution power.
- **H2 (Hype):** Keywords (Benchmarks > Models). Measures search volume.
- **H3 (Sniper):** Penalties (Niche Topics). Measures market ceiling.
- **H4 (Blind Utility):** GPT-5.2 rates the abstract 0-100 ignoring author names. Measures content quality.

### 3. Train Brain (Gradient Boosting)

- It trains a **Gradient Boosting Regressor** on the Training Set to predict citation counts.
- It analyzes which features actually contributed to accurate predictions.
- It distills the complex model into a simple **Linear Formula (JSON)**.

### 4. Validate (The Exam)

- It takes the weights learned in Step 3 and applies them to Step 1's Test Set.
- If **Precision@10 >= 50%**, the model is declared **Production Ready**.

## 📦 Deployment

To use the new weights in your live application:

1. Open `moneyball_weights.json`.
2. Copy the values.
3. Update your production variables:

```python
# In your live app:
WEIGHT_FAME = 0.84   # Derived from weight_fame
WEIGHT_UTIL = 0.16   # Derived from weight_utility
# etc...
```

## 🧹 Troubleshooting

- **"Not enough training data":** The script needs at least 50 papers with valid citation counts. Ensure your `S2_API_KEY` is valid.
- **Precision is low (<40%):** The market may have shifted. Delete `training_logs.csv` and run the script again to fetch fresh data.


