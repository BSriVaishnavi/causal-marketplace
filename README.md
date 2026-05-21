# Causal Marketplace Experimentation Platform

> **"Built a causal inference experimentation platform on a fashion-electronics marketplace — demonstrated where standard A/B fails under seasonal confounding, and applied DiD and Synthetic Control to recover true causal effects"**

---

## Project Overview

A full-stack experimentation platform built on a simulated fashion + electronics marketplace with 10,000 users, 800 products, and 200,000 interactions. The project demonstrates a critical failure mode in standard A/B testing caused by seasonal confounding — and fixes it using causal inference methods.

**The core finding:** Naive A/B testing concluded *"kill both variants, control wins"* with p=0.000000. Difference-in-Differences and Synthetic Control revealed the opposite — Variant A was the true winner with a +1.37pp causal lift in conversion rate, representing an estimated **$15,657 in missed revenue**.

link: https://causal-marketplace.streamlit.app/
---

## Architecture

causal_marketplace/
├── market.ipynb                  # Main Jupyter notebook (all phases)
├── app/
│   ├── api/
│   │   └── main.py               # FastAPI backend
│   └── dashboard/
│       └── streamlit_app.py      # Streamlit dashboard
├── users.csv                     # 10K simulated users
├── products.csv                  # 800 products (fashion + electronics)
├── interactions_exp_final.csv    # 200K interactions with experiment groups
└── synthetic_control.csv         # Synthetic control results

---

## What Was Tested

| Variant | Recommendation Method | Discount |
|---|---|---|
| Control | Popularity-based | 0% |
| Variant A | Collaborative Filtering | 5% |
| Variant B | Session-aware | 10% |

---

## The Flip Moment

| Method | Winner | Conversion Lift | Decision |
|---|---|---|---|
| Naive A/B (t-test) | ❌ Control | — | Kill both variants |
| Difference-in-Differences | ✅ Variant A | +1.37pp | Ship Variant A |
| Synthetic Control | ✅ Variant A | +1.12pp avg, +2.96pp peak | Ship Variant A |

### Why A/B Was Wrong
- Control group had more interactions during peak holiday months (Nov/Dec)
- Electronics buyers (high converters in Nov/Dec) were unevenly distributed across groups
- Naive t-test picked up seasonal signal as treatment effect
- DiD and Synthetic Control removed the seasonal component and revealed the truth

---

## Methods

### Difference-in-Differences (DiD)
Compares the *change* in conversion rate from pre-period (Jan–Jun) to post-period (Jul–Dec) across treatment and control groups. Controls for pre-existing differences between groups.

DiD Estimate = (Treated_post - Treated_pre) - (Control_post - Control_pre)

- Variant A DiD estimate: **+0.0137** (p=0.000000) ✅
- Variant B DiD estimate: **-0.0014** (p=0.5527) — no effect

### Synthetic Control
Constructs a weighted counterfactual control group from donor units (control + variant B) that best matches variant A in the pre-treatment period. The post-treatment gap between actual and synthetic is the causal effect.

- Optimal weights: 22.3% control + 77.7% variant B
- Pre-period gap: -0.0037 (near zero — good fit) ✅
- Post-period avg gap: **+0.0112** (true causal lift)
- Peak gap (December): **+0.0296**

---

## Stack

| Layer | Technology |
|---|---|
| Data simulation | Python, NumPy, Pandas, Faker |
| Statistical testing | SciPy, Statsmodels |
| Machine learning | Scikit-learn |
| Causal inference | Statsmodels OLS, SciPy optimize |
| Visualization | Matplotlib, Seaborn |
| Backend API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Environment | Anaconda, Python 3.12 |

---

## Setup & Run

### 1. Clone and set up environment
```bash
git clone https://github.com/yourusername/causal-marketplace.git
cd causal-marketplace
conda create -n causal_marketplace python=3.12 -y
conda activate causal_marketplace
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter notebook fastapi uvicorn streamlit plotly faker tqdm statsmodels
```

### 2. Run the notebook
```bash
jupyter notebook
# Open market.ipynb and Run All
```

### 3. Start FastAPI backend
```bash
uvicorn app.api.main:app --reload
# API runs at http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

### 4. Start Streamlit dashboard
```bash
streamlit run app/dashboard/streamlit_app.py
# Dashboard runs at http://127.0.0.1:8501
```

---

## Key Results

- **10,000** simulated users across 3 behavioral profiles
- **800 products** across fashion and electronics categories
- **200,000 interactions** with seasonal confounding baked in
- **Naive A/B p-value:** 0.000000 — statistically significant but causally wrong
- **DiD causal lift:** +1.37 percentage points (p=0.000000)
- **Synthetic Control lift:** +1.12pp average, +2.96pp peak (December)
- **Estimated missed revenue:** $15,657 if naive A/B decision was followed

---

## Resume Bullets

- Built end-to-end causal inference experimentation platform on a simulated fashion-electronics marketplace with 10K users and 200K interactions
- Demonstrated that naive A/B testing overestimated control group performance due to seasonal confounding, producing statistically significant but causally incorrect results (p=0.000000)
- Applied Difference-in-Differences to recover a true causal lift of +1.37pp for Variant A (collaborative filtering + 5% discount), reversing the naive A/B conclusion
- Constructed Synthetic Control counterfactual with 22% control / 78% variant B donor weights, confirming +1.12pp average causal lift growing to +2.96pp in peak season
- Quantified $15,657 estimated revenue impact of using causal vs naive methods
- Deployed FastAPI recommendation and experiment API with Streamlit dashboard
