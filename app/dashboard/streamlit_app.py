import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Causal Marketplace Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    users         = pd.read_csv('users.csv')
    products      = pd.read_csv('products.csv')
    interactions  = pd.read_csv('interactions_exp_final.csv', parse_dates=['date'])
    sc_pivot      = pd.read_csv('synthetic_control.csv')
    return users, products, interactions, sc_pivot

users, products, interactions, sc_pivot = load_data()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🛍️ Causal Marketplace")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "A/B Testing", "Causal Inference", "The Flip Moment", "Recommendations"]
)

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🛍️ Causal Marketplace Experimentation Platform")
    st.markdown("### Fashion + Electronics | A/B Testing vs Causal Inference")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users",        f"{len(users):,}")
    col2.metric("Total Products",     f"{len(products):,}")
    col3.metric("Total Interactions", f"{len(interactions):,}")
    col4.metric("Overall CVR",        f"{interactions['purchased'].mean():.2%}")

    st.markdown("---")
    st.subheader("User Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = users['user_type'].value_counts()
        ax.bar(vc.index, vc.values, color=['steelblue','coral','green'])
        ax.set_title('Users by Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        vc2 = users['experiment_group'].value_counts()
        ax.bar(vc2.index, vc2.values, color=['steelblue','coral','green'])
        ax.set_title('Users by Experiment Group')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Monthly Purchase Volume by Category")
    monthly = interactions[interactions['purchased'] == 1]\
        .groupby(['month','category']).size().reset_index(name='purchases')

    fig, ax = plt.subplots(figsize=(12, 4))
    for cat in ['fashion','electronics']:
        d = monthly[monthly['category'] == cat]
        ax.plot(d['month'], d['purchases'], marker='o', label=cat)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('Purchases')
    ax.legend()
    ax.set_title('Seasonal Confounding — Both Categories Spike at Different Times')
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 2 — A/B TESTING
# ══════════════════════════════════════════════════════════
elif page == "A/B Testing":
    st.title("📊 Naive A/B Testing Results")
    st.warning("⚠️ These results are confounded by seasonality — see Causal Inference page for corrected results.")
    st.markdown("---")

    metrics = interactions.groupby('experiment_group').agg(
        total_interactions=('purchased','count'),
        purchases=('purchased','sum'),
        total_revenue=('revenue','sum')
    ).reset_index()
    metrics['ctr']              = (interactions.groupby('experiment_group')['clicked'].mean().values)
    metrics['conversion_rate']  = (metrics['purchases'] / metrics['total_interactions']).round(4)
    metrics['avg_order_value']  = (interactions[interactions['purchased']==1]
                                   .groupby('experiment_group')['revenue'].mean().values)
    metrics['total_revenue']    = metrics['total_revenue'].round(2)

    st.subheader("Experiment Metrics")
    st.dataframe(metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("Statistical Significance (t-test)")

    control   = interactions[interactions['experiment_group']=='control']['purchased'].values
    variant_a = interactions[interactions['experiment_group']=='variant_a']['purchased'].values
    variant_b = interactions[interactions['experiment_group']=='variant_b']['purchased'].values

    t_a, p_a = stats.ttest_ind(control, variant_a)
    t_b, p_b = stats.ttest_ind(control, variant_b)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Control vs Variant A p-value", f"{p_a:.6f}")
        st.error("Naive A/B says: Control wins ❌")
    with col2:
        st.metric("Control vs Variant B p-value", f"{p_b:.6f}")
        st.error("Naive A/B says: Control wins ❌")

    st.markdown("---")
    st.subheader("Conversion Rate by Month per Group")
    monthly_conv = interactions.groupby(['month','experiment_group'])\
        .apply(lambda x: x['purchased'].sum()/len(x)).reset_index(name='cvr')

    fig, ax = plt.subplots(figsize=(12, 4))
    for grp in ['control','variant_a','variant_b']:
        d = monthly_conv[monthly_conv['experiment_group']==grp]
        ax.plot(d['month'], d['cvr'], marker='o', label=grp)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('Conversion Rate')
    ax.legend()
    ax.set_title('Conversion Rate by Month — Seasonal Spikes Visible')
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 3 — CAUSAL INFERENCE
# ══════════════════════════════════════════════════════════
elif page == "Causal Inference":
    st.title("🔬 Causal Inference Results")
    st.success("✅ These results control for seasonal confounding")
    st.markdown("---")

    st.subheader("Difference-in-Differences")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("DiD Estimate — Variant A", "+0.0137")
        st.metric("p-value", "0.000000")
        st.success("DiD says: Variant A wins ✅")
    with col2:
        st.metric("DiD Estimate — Variant B", "-0.0014")
        st.metric("p-value", "0.5527")
        st.warning("DiD says: No significant effect for Variant B")

    st.markdown("---")
    st.subheader("Synthetic Control — Variant A vs Counterfactual")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(sc_pivot['month'], sc_pivot['variant_a'],
                 marker='o', label='Variant A (Actual)', color='steelblue')
    axes[0].plot(sc_pivot['month'], sc_pivot['synthetic_a'],
                 marker='o', linestyle='--', label='Synthetic Control', color='coral')
    axes[0].axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, label='Treatment Start')
    axes[0].fill_between(
        sc_pivot['month'], sc_pivot['synthetic_a'], sc_pivot['variant_a'],
        where=sc_pivot['month'] > 6, alpha=0.2, color='green', label='Causal Effect'
    )
    axes[0].set_title('Synthetic Control: Actual vs Counterfactual')
    axes[0].set_xticks(range(1,13))
    axes[0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                              'Jul','Aug','Sep','Oct','Nov','Dec'])
    axes[0].set_ylabel('Conversion Rate')
    axes[0].legend()

    axes[1].bar(sc_pivot['month'], sc_pivot['gap'],
                color=np.where(sc_pivot['gap'] > 0, 'green', 'red'))
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5)
    axes[1].set_title('Causal Gap by Month')
    axes[1].set_xticks(range(1,13))
    axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                              'Jul','Aug','Sep','Oct','Nov','Dec'])
    axes[1].set_ylabel('Gap in Conversion Rate')

    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 4 — THE FLIP MOMENT
# ══════════════════════════════════════════════════════════
elif page == "The Flip Moment":
    st.title("🚨 The Flip Moment")
    st.markdown("### Where Naive A/B Testing Gets It Wrong")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Naive A/B Test")
        st.error("Winner: **Control**")
        st.markdown("- Control CVR: 2.86%")
        st.markdown("- Variant A CVR: 2.26%")
        st.markdown("- Variant B CVR: 1.74%")
        st.markdown("- p-value: 0.000000")
        st.markdown("**Decision: Kill both variants**")

    with col2:
        st.subheader("Difference-in-Differences")
        st.success("Winner: **Variant A**")
        st.markdown("- DiD Estimate: +0.0137")
        st.markdown("- p-value: 0.000000")
        st.markdown("- Variant B: No effect")
        st.markdown("**Decision: Ship Variant A**")

    with col3:
        st.subheader("Synthetic Control")
        st.success("Winner: **Variant A**")
        st.markdown("- Avg causal lift: +1.12pp")
        st.markdown("- Peak lift (Dec): +2.96pp")
        st.markdown("- Pre-period gap: -0.37pp ✅")
        st.markdown("**Decision: Ship Variant A**")

    st.markdown("---")
    st.subheader("Why A/B Was Wrong")
    st.markdown("""
    - Control group had **more interactions during peak holiday months**
    - Electronics buyers (high converters in Nov/Dec) were unevenly distributed
    - Naive t-test picked up **seasonal signal as treatment effect**
    - DiD and Synthetic Control **removed the seasonal component** and revealed the truth
    """)

    st.subheader("Business Impact")
    avg_order = interactions[interactions['purchased']==1]['revenue'].mean()
    total_users = len(users[users['experiment_group']=='variant_a'])
    estimated_lift = 0.0137 * total_users * avg_order
    st.metric(
        "Estimated Revenue Lift from Shipping Variant A",
        f"${estimated_lift:,.2f}",
        delta="Would have been missed by naive A/B"
    )

# ══════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.title("🎯 Recommendation Engine")
    st.markdown("---")

    user_id = st.selectbox("Select a User ID", users['user_id'].sample(50).sort_values())

    if user_id:
        user = users[users['user_id'] == user_id].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("User Type",         user['user_type'])
        col2.metric("Experiment Group",  user['experiment_group'])
        col3.metric("Price Sensitivity", f"{user['price_sensitivity']:.2f}")

        st.markdown("---")
        group   = user['experiment_group']
        discount_map = {'control': 0.0, 'variant_a': 0.05, 'variant_b': 0.10}
        discount = discount_map[group]

        if group == 'control':
            popular = (interactions[interactions['purchased']==1]
                       .groupby('product_id').size()
                       .sort_values(ascending=False).head(5).index.tolist())
            recs = products[products['product_id'].isin(popular)].copy()
        elif group == 'variant_a':
            cat = 'fashion' if user['user_type']=='fashion_shopper' else 'electronics'
            recs = products[products['category']==cat].sample(5).copy()
        else:
            recs = products.sample(5).copy()

        recs['discounted_price'] = (recs['price'] * (1 - discount)).round(2)
        recs['discount_applied'] = f"{int(discount*100)}%"

        st.subheader(f"Recommendations for {user_id} — {group.upper()}")
        st.dataframe(
            recs[['product_id','category','subcategory','price','discounted_price','discount_applied']],
            use_container_width=True
        )