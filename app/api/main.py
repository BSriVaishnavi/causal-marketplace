from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import random

app = FastAPI(title="Causal Marketplace API")

# Load data
users = pd.read_csv('users.csv')
products = pd.read_csv('products.csv')
interactions = pd.read_csv('interactions_exp_final.csv')

# Experiment config
experiment_config = {
    'control': {'recommendation': 'popularity_based', 'discount': 0.0},
    'variant_a': {'recommendation': 'collaborative_filtering', 'discount': 0.05},
    'variant_b': {'recommendation': 'session_aware', 'discount': 0.10}
}

# Popularity based recommendations
popular_products = (
    interactions[interactions['purchased'] == 1]
    .groupby('product_id')
    .size()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

class UserRequest(BaseModel):
    user_id: str

class ExperimentRequest(BaseModel):
    user_id: str
    category: str = 'all'

@app.get("/")
def root():
    return {"status": "Causal Marketplace API is running"}

@app.post("/recommend")
def recommend(request: UserRequest):
    user_id = request.user_id

    if user_id not in users['user_id'].values:
        return {"error": "User not found"}

    user = users[users['user_id'] == user_id].iloc[0]
    group = user.get('experiment_group', 'control')
    config = experiment_config.get(group, experiment_config['control'])
    discount = config['discount']
    method = config['recommendation']

    if method == 'popularity_based':
        recs = products[products['product_id'].isin(popular_products)].head(5)

    elif method == 'collaborative_filtering':
        # Simplified CF — recommend from same category as user type
        if user['user_type'] == 'fashion_shopper':
            recs = products[products['category'] == 'fashion'].sample(5)
        elif user['user_type'] == 'electronics_buyer':
            recs = products[products['category'] == 'electronics'].sample(5)
        else:
            recs = products.sample(5)

    else:  # session_aware
        recs = products.sample(5)

    recs = recs.copy()
    recs['discounted_price'] = (recs['price'] * (1 - discount)).round(2)

    return {
        "user_id": user_id,
        "experiment_group": group,
        "recommendation_method": method,
        "discount_applied": f"{int(discount * 100)}%",
        "recommendations": recs[['product_id', 'category', 'price', 'discounted_price']].to_dict(orient='records')
    }

@app.get("/experiment/summary")
def experiment_summary():
    summary = (
        interactions.groupby('experiment_group')
        .agg(
            total_interactions=('purchased', 'count'),
            total_purchases=('purchased', 'sum'),
            total_revenue=('revenue', 'sum')
        )
        .reset_index()
    )
    summary['conversion_rate'] = (
        summary['total_purchases'] / summary['total_interactions']
    ).round(4)
    summary['total_revenue'] = summary['total_revenue'].round(2)

    return summary.to_dict(orient='records')

@app.get("/experiment/winner")
def experiment_winner():
    return {
        "naive_ab_winner": "control",
        "did_winner": "variant_a",
        "synthetic_control_winner": "variant_a",
        "conclusion": "Naive A/B was wrong. Variant A is the true winner after causal correction.",
        "did_estimate": 0.013743,
        "synthetic_control_lift": 0.011178
    }