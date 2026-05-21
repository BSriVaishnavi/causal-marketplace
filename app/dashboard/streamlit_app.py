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
    page_title="Causal Marketplace",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0f1117; }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    
    .hero-sub {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover { transform: translateY(-2px); }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #667eea;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .flip-wrong {
        background: linear-gradient(135deg, #2d1515 0%, #3d1a1a 100%);
        border: 2px solid #fc4e4e;
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .flip-right {
        background: linear-gradient(135deg, #0d2d1a 0%, #0d3d22 100%);
        border: 2px solid #48bb78;
        border-radius: 16px;
        padding: 1.5rem;
    }

    .flip-neutral {
        background: linear-gradient(135deg, #1a2035 0%, #1a2a45 100%);
        border: 2px solid #667eea;
        border-radius: 16px;
        padding: 1.5rem;
    }

    .winner-badge-wrong {
        background: #fc4e4e;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }

    .winner-badge-right {
        background: #48bb78;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }

    .revenue-impact {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }

    .revenue-number {
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        line-height: 1;
    }

    .revenue-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }

    .tag {
        background: #2d3748;
        color: #a0aec0;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
        display: inline-block;
    }

    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #667eea;
    }

    .stRadio > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    users        = pd.read_csv('users.csv')
    products     = pd.read_csv('products.csv')
    interactions = pd.read_csv('interactions_exp_final.csv', parse_dates=['date'])
    sc_pivot     = pd.read_csv('synthetic_control.csv')
    return users, products, interactions, sc_pivot

users, products, interactions, sc_pivot = load_data()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛍️ Causal Marketplace</div>', unsafe_allow_html=True)
    st.markdown("*Fashion + Electronics | Causal Inference Platform*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 A/B Testing", "🔬 Causal Inference", "🚨 The Flip Moment", "🎯 Recommendations"]
    )
    st.markdown("---")
    st.markdown("**Stack**")
    for tag in ["Python", "FastAPI", "Streamlit", "DiD", "Synthetic Control", "Scikit-learn"]:
        st.markdown(f'<span class="tag">{tag}</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Key Finding**")
    st.error("Naive A/B: Control wins ❌")
    st.success("Causal methods: Variant A wins ✅")

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="hero-title">Causal Marketplace<br>Experimentation Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Fashion + Electronics | Where Naive A/B Testing Fails — and Causal Inference Fixes It</div>', unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">10K</div><div class="metric-label">Simulated Users</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">800</div><div class="metric-label">Products</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">200K</div><div class="metric-label">Interactions</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">2.26%</div><div class="metric-label">Overall CVR</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Revenue impact teaser
    st.markdown("""
    <div class="revenue-impact">
        <div class="revenue-number">$15,657</div>
        <div class="revenue-label">Estimated revenue missed if naive A/B decision was followed</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Seasonal Confounding</div>', unsafe_allow_html=True)

    monthly = interactions[interactions['purchased'] == 1]\
        .groupby(['month','category']).size().reset_index(name='purchases')

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1f2e')
    ax.set_facecolor('#1a1f2e')
    for cat, color in [('fashion','#667eea'), ('electronics','#f6ad55')]:
        d = monthly[monthly['category'] == cat]
        ax.plot(d['month'], d['purchases'], marker='o', label=cat, color=color, linewidth=2.5)
        ax.fill_between(d['month'], d['purchases'], alpha=0.1, color=color)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], color='#a0aec0')
    ax.set_ylabel('Purchases', color='#a0aec0')
    ax.tick_params(colors='#a0aec0')
    ax.spines['bottom'].set_color('#2d3748')
    ax.spines['left'].set_color('#2d3748')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1f2e', labelcolor='white')
    ax.set_title('Fashion peaks in summer — Electronics spikes in Nov/Dec holiday season', color='#e2e8f0', pad=15)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">User Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    for col, groupby, title, colors in [
        (col1, 'user_type', 'Users by Type', ['#667eea','#f6ad55','#48bb78']),
        (col2, 'experiment_group', 'Users by Experiment Group', ['#667eea','#f6ad55','#48bb78'])
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            vc = users[groupby].value_counts()
            bars = ax.bar(vc.index, vc.values, color=colors, edgecolor='none', width=0.5)
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                       f'{val:,}', ha='center', color='#a0aec0', fontsize=9)
            ax.set_title(title, color='#e2e8f0')
            ax.tick_params(colors='#a0aec0')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_yticks([])
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 2 — A/B TESTING
# ══════════════════════════════════════════════════════════
elif "A/B" in page:
    st.markdown('<div class="hero-title">Naive A/B Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Statistically significant — but causally wrong</div>', unsafe_allow_html=True)

    st.warning("⚠️ These results are confounded by seasonality. Navigate to Causal Inference for corrected results.")

    metrics = interactions.groupby('experiment_group').agg(
        total_interactions=('purchased','count'),
        purchases=('purchased','sum'),
        total_revenue=('revenue','sum')
    ).reset_index()
    metrics['conversion_rate'] = (metrics['purchases'] / metrics['total_interactions']).round(4)
    metrics['total_revenue']   = metrics['total_revenue'].round(2)

    st.markdown('<div class="section-header">Experiment Metrics</div>', unsafe_allow_html=True)
    st.dataframe(metrics, use_container_width=True)

    st.markdown('<div class="section-header">Statistical Significance</div>', unsafe_allow_html=True)

    control   = interactions[interactions['experiment_group']=='control']['purchased'].values
    variant_a = interactions[interactions['experiment_group']=='variant_a']['purchased'].values
    variant_b = interactions[interactions['experiment_group']=='variant_b']['purchased'].values

    t_a, p_a = stats.ttest_ind(control, variant_a)
    t_b, p_b = stats.ttest_ind(control, variant_b)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="flip-wrong">
            <div class="winner-badge-wrong">Control vs Variant A</div><br>
            <b style="color:#fc4e4e;font-size:1.8rem;">p = {p_a:.6f}</b><br>
            <span style="color:#fc4e4e;">❌ Naive A/B says: Control wins</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="flip-wrong">
            <div class="winner-badge-wrong">Control vs Variant B</div><br>
            <b style="color:#fc4e4e;font-size:1.8rem;">p = {p_b:.6f}</b><br>
            <span style="color:#fc4e4e;">❌ Naive A/B says: Control wins</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Conversion Rate by Month</div>', unsafe_allow_html=True)

    monthly_conv = interactions.groupby(['month','experiment_group'])\
        .apply(lambda x: x['purchased'].sum()/len(x)).reset_index(name='cvr')

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1f2e')
    ax.set_facecolor('#1a1f2e')
    colors = {'control':'#667eea','variant_a':'#f6ad55','variant_b':'#48bb78'}
    for grp in ['control','variant_a','variant_b']:
        d = monthly_conv[monthly_conv['experiment_group']==grp]
        ax.plot(d['month'], d['cvr'], marker='o', label=grp,
                color=colors[grp], linewidth=2.5)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], color='#a0aec0')
    ax.set_ylabel('Conversion Rate', color='#a0aec0')
    ax.tick_params(colors='#a0aec0')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#2d3748')
    ax.legend(facecolor='#1a1f2e', labelcolor='white')
    ax.set_title('All groups spike together — seasonal signal, not treatment effect', color='#e2e8f0', pad=15)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 3 — CAUSAL INFERENCE
# ══════════════════════════════════════════════════════════
elif "Causal" in page:
    st.markdown('<div class="hero-title">Causal Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Difference-in-Differences + Synthetic Control — the corrected answer</div>', unsafe_allow_html=True)

    st.success("✅ These results control for seasonal confounding")

    st.markdown('<div class="section-header">Difference-in-Differences</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="flip-right">
            <div class="winner-badge-right">Control vs Variant A</div><br>
            <b style="color:#48bb78;font-size:1.8rem;">DiD = +0.0137</b><br>
            <span style="color:#a0aec0;">p-value: 0.000000</span><br><br>
            <span style="color:#48bb78;">✅ DiD says: Variant A wins</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="flip-neutral">
            <div class="winner-badge-wrong">Control vs Variant B</div><br>
            <b style="color:#a0aec0;font-size:1.8rem;">DiD = -0.0014</b><br>
            <span style="color:#a0aec0;">p-value: 0.5527</span><br><br>
            <span style="color:#a0aec0;">— No significant effect</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Synthetic Control — Variant A vs Counterfactual</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor='#1a1f2e')
    for ax in axes:
        ax.set_facecolor('#1a1f2e')

    axes[0].plot(sc_pivot['month'], sc_pivot['variant_a'],
                 marker='o', label='Variant A (Actual)', color='#667eea', linewidth=2.5)
    axes[0].plot(sc_pivot['month'], sc_pivot['synthetic_a'],
                 marker='o', linestyle='--', label='Synthetic Control', color='#f6ad55', linewidth=2.5)
    axes[0].axvline(x=6.5, color='#718096', linestyle='--', linewidth=1.5, label='Treatment Start')
    axes[0].fill_between(
        sc_pivot['month'], sc_pivot['synthetic_a'], sc_pivot['variant_a'],
        where=sc_pivot['month'] > 6, alpha=0.25, color='#48bb78', label='Causal Effect'
    )
    axes[0].set_title('Actual vs Counterfactual\n(Green = True Causal Lift)', color='#e2e8f0')
    axes[0].set_xticks(range(1,13))
    axes[0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], color='#a0aec0')
    axes[0].tick_params(colors='#a0aec0')
    axes[0].set_ylabel('Conversion Rate', color='#a0aec0')
    axes[0].legend(facecolor='#1a1f2e', labelcolor='white')
    for spine in ['top','right']:
        axes[0].spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        axes[0].spines[spine].set_color('#2d3748')

    bar_colors = ['#48bb78' if g > 0 else '#fc4e4e' for g in sc_pivot['gap']]
    axes[1].bar(sc_pivot['month'], sc_pivot['gap'], color=bar_colors, edgecolor='none')
    axes[1].axhline(y=0, color='#718096', linewidth=1)
    axes[1].axvline(x=6.5, color='#718096', linestyle='--', linewidth=1.5, label='Treatment Start')
    axes[1].set_title('Causal Gap by Month\n(Variant A − Synthetic Control)', color='#e2e8f0')
    axes[1].set_xticks(range(1,13))
    axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], color='#a0aec0')
    axes[1].tick_params(colors='#a0aec0')
    axes[1].set_ylabel('Gap in Conversion Rate', color='#a0aec0')
    axes[1].legend(facecolor='#1a1f2e', labelcolor='white')
    for spine in ['top','right']:
        axes[1].spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        axes[1].spines[spine].set_color('#2d3748')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 4 — THE FLIP MOMENT
# ══════════════════════════════════════════════════════════
elif "Flip" in page:
    st.markdown('<div class="hero-title">🚨 The Flip Moment</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Same data. Three methods. Three different business decisions.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="flip-wrong">
            <div class="winner-badge-wrong">❌ Naive A/B Test</div>
            <b style="color:#fc4e4e;font-size:1.4rem;">Winner: Control</b>
            <hr style="border-color:#3d1a1a;">
            <p style="color:#a0aec0;font-size:0.9rem;">
            • Control CVR: 2.86%<br>
            • Variant A CVR: 2.26%<br>
            • Variant B CVR: 1.74%<br>
            • p-value: 0.000000
            </p>
            <b style="color:#fc4e4e;">Decision: Kill both variants</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="flip-right">
            <div class="winner-badge-right">✅ Difference-in-Differences</div>
            <b style="color:#48bb78;font-size:1.4rem;">Winner: Variant A</b>
            <hr style="border-color:#0d3d22;">
            <p style="color:#a0aec0;font-size:0.9rem;">
            • DiD Estimate: +0.0137<br>
            • p-value: 0.000000<br>
            • Variant B: No effect<br>
            • Controls for pre-trends
            </p>
            <b style="color:#48bb78;">Decision: Ship Variant A</b>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="flip-right">
            <div class="winner-badge-right">✅ Synthetic Control</div>
            <b style="color:#48bb78;font-size:1.4rem;">Winner: Variant A</b>
            <hr style="border-color:#0d3d22;">
            <p style="color:#a0aec0;font-size:0.9rem;">
            • Avg causal lift: +1.12pp<br>
            • Peak lift (Dec): +2.96pp<br>
            • Pre-period gap: -0.37pp ✅<br>
            • Weights: 22% ctrl / 78% var_b
            </p>
            <b style="color:#48bb78;">Decision: Ship Variant A</b>
        </div>
        """, unsafe_allow_html=True)

    avg_order = interactions[interactions['purchased']==1]['revenue'].mean()
    total_users = len(users[users['experiment_group']=='variant_a'])
    estimated_lift = 0.0137 * total_users * avg_order

    st.markdown(f"""
    <div class="revenue-impact">
        <div style="color:rgba(255,255,255,0.7);font-size:1rem;margin-bottom:0.5rem;">
        ESTIMATED REVENUE MISSED BY NAIVE A/B TESTING
        </div>
        <div class="revenue-number">${estimated_lift:,.2f}</div>
        <div class="revenue-label">
        Based on {total_users:,} Variant A users × {avg_order:.2f} avg order value × 1.37pp causal lift
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Why A/B Was Wrong</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **The Confounding Problem**
        - Control group had more interactions during peak holiday months
        - Electronics buyers (high converters in Nov/Dec) were unevenly distributed
        - Naive t-test picked up seasonal signal as treatment effect
        - Standard A/B assumes stable unit treatment value — violated here
        """)
    with col2:
        st.markdown("""
        **How Causal Methods Fixed It**
        - **DiD** compared the *change* pre→post, not the *level*
        - This cancels out time-invariant group differences
        - **Synthetic Control** built a counterfactual from donor units
        - Isolated what Variant A *caused* vs what seasonality caused
        """)

# ══════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
elif "Recommendations" in page:
    st.markdown('<div class="hero-title">Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Personalized recommendations per experiment group</div>', unsafe_allow_html=True)

    user_id = st.selectbox("Select a User ID", users['user_id'].sample(50).sort_values())

    if user_id:
        user = users[users['user_id'] == user_id].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("User Type", user['user_type'])
        col2.metric("Experiment Group", user['experiment_group'])
        col3.metric("Price Sensitivity", f"{user['price_sensitivity']:.2f}")
        col4.metric("Session Frequency", f"{user['session_frequency']} visits/mo")

        st.markdown("---")

        group = user['experiment_group']
        discount_map = {'control': 0.0, 'variant_a': 0.05, 'variant_b': 0.10}
        method_map = {
            'control': 'Popularity-based',
            'variant_a': 'Collaborative Filtering',
            'variant_b': 'Session-aware'
        }
        discount = discount_map[group]

        col1, col2 = st.columns(2)
        col1.info(f"**Method:** {method_map[group]}")
        col2.info(f"**Discount Applied:** {int(discount*100)}%")

        if group == 'control':
            popular = (interactions[interactions['purchased']==1]
                      .groupby('product_id').size()
                      .sort_values(ascending=False).head(5).index.tolist())
            recs = products[products['product_id'].isin(popular)].copy()
        elif group == 'variant_a':
            cat = 'fashion' if user['user_type'] == 'fashion_shopper' else 'electronics'
            recs = products[products['category'] == cat].sample(5).copy()
        else:
            recs = products.sample(5).copy()

        recs['discounted_price'] = (recs['price'] * (1 - discount)).round(2)
        recs['discount_applied'] = f"{int(discount*100)}%"
        recs['savings'] = (recs['price'] - recs['discounted_price']).round(2)

        st.markdown(f'<div class="section-header">Recommendations for {user_id} — {group.upper()}</div>', unsafe_allow_html=True)
        st.dataframe(
            recs[['product_id','category','subcategory','price','discounted_price','savings','discount_applied']],
            use_container_width=True
        )