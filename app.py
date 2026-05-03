# ═══════════════════════════════════════════════════════════════════════════════
#  Customer Persona Intelligence Platform  —  app.py
#  Run:  streamlit run app.py
#  Requirements: pip install streamlit plotly wordcloud pandas numpy scikit-learn
#                          matplotlib vaderSentiment openpyxl
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import io
import base64
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Persona Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f0f2f5;
}
.main .block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { background: #131921 !important; }
section[data-testid="stSidebar"] * { color: #d1d5db !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 5px solid;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 140px;
    display: flex; flex-direction: column; justify-content: space-between;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.13); }
.kpi-icon  { font-size: 26px; margin-bottom: 4px; }
.kpi-label { font-size: 12px; color: #6b7280; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.6px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #111827; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #9ca3af; margin-top: 3px; }

/* ── Section titles ── */
.section-title {
    font-size: 18px; font-weight: 700; color: #111827;
    margin: 28px 0 16px 0; padding-bottom: 10px;
    border-bottom: 2px solid #e5e7eb;
    display: flex; align-items: center; gap: 8px;
}

/* ── Chart cards ── */
.chart-card {
    background: #ffffff; border-radius: 16px; padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

/* ── Persona card ── */
.persona-card {
    background: #ffffff; border-radius: 16px; padding: 22px 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.09);
    border-top: 5px solid;
    margin-bottom: 16px;
}
.persona-name  { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.persona-desc  { font-size: 13px; color: #6b7280; line-height: 1.6; margin-bottom: 14px; }
.persona-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 700; margin: 2px;
}
.strategy-box {
    background: #f8faff; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 14px 16px;
    font-size: 13px; color: #374151; line-height: 1.6;
    margin-top: 12px;
}
.strategy-box b { color: #1d4ed8; }

/* ── Chat messages ── */
.chat-user {
    background: #131921; color: #ffffff; border-radius: 16px 16px 4px 16px;
    padding: 12px 18px; margin: 8px 0 8px 60px;
    font-size: 13px; line-height: 1.5; max-width: 85%;
    margin-left: auto;
}
.chat-bot {
    background: #ffffff; color: #111827; border-radius: 16px 16px 16px 4px;
    padding: 12px 18px; margin: 8px 60px 8px 0;
    font-size: 13px; line-height: 1.6; max-width: 85%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.chat-bot b { color: #131921; }

/* ── Keyword tags ── */
.kw-tag {
    display: inline-block; background: #f3f4f6; color: #374151;
    border-radius: 20px; padding: 4px 12px; font-size: 12px;
    font-weight: 500; margin: 3px 2px; border: 1px solid #e5e7eb;
}

/* ── Sidebar navigation ── */
.nav-btn {
    display: block; width: 100%; text-align: left;
    background: transparent; border: none; cursor: pointer;
    padding: 10px 14px; border-radius: 8px; font-size: 14px;
    color: #d1d5db; font-family: 'DM Sans', sans-serif;
    transition: background 0.15s;
}
.nav-btn:hover { background: #232f3e; }
.nav-btn.active { background: #ff9900; color: #131921 !important;
                   font-weight: 700; }

/* ── Carousel ── */
.carousel-wrapper {
    overflow: hidden; width: 100%;
    background: #ffffff; border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    padding: 20px 0 20px 20px;
    margin-bottom: 24px;
}
.carousel-track {
    display: flex; gap: 16px;
    animation: scroll-left 28s linear infinite;
    width: max-content;
}
.carousel-track:hover { animation-play-state: paused; }
@keyframes scroll-left {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.product-card {
    background: #f8f9fa; border-radius: 12px; padding: 14px;
    width: 180px; flex-shrink: 0; cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid #e5e7eb;
    text-align: center;
}
.product-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 12px 28px rgba(0,0,0,0.15);
    background: #ffffff;
}
.product-img {
    width: 100%; height: 120px; object-fit: cover;
    border-radius: 8px; margin-bottom: 10px;
    background: #e5e7eb;
}
.product-title {
    font-size: 12px; font-weight: 600; color: #111827;
    margin-bottom: 4px; line-height: 1.3;
}
.product-price { font-size: 14px; font-weight: 700; color: #B12704; }
.product-stars { font-size: 11px; color: #f59e0b; margin: 2px 0; }
.product-badge {
    display: inline-block; background: #ff9900; color: #131921;
    font-size: 9px; font-weight: 800; padding: 2px 6px;
    border-radius: 3px; text-transform: uppercase; margin-top: 3px;
}

/* ── Amazon-style top navbar ── */
.top-navbar {
    background: #131921; padding: 10px 28px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; margin-bottom: 0;
    position: sticky; top: 0; z-index: 999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.navbar-brand {
    font-size: 22px; font-weight: 800; color: #ffffff;
    letter-spacing: -0.5px; white-space: nowrap;
}
.navbar-brand span { color: #ff9900; }
.navbar-search {
    flex: 1; max-width: 540px;
    background: #ffffff; border-radius: 6px;
    display: flex; align-items: center; overflow: hidden;
}
.navbar-search input {
    flex: 1; border: none; outline: none;
    padding: 9px 14px; font-size: 13px;
    font-family: 'DM Sans', sans-serif; background: transparent;
}
.navbar-search-btn {
    background: #ff9900; border: none; cursor: pointer;
    padding: 8px 14px; font-size: 16px;
}
.navbar-profile {
    display: flex; align-items: center; gap: 10px;
    color: #d1d5db; font-size: 13px;
}
.navbar-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: #ff9900; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 14px;
    color: #131921;
}
.navbar-time { font-size: 11px; color: #9ca3af; }
.navbar-loc  { font-size: 11px; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PIPELINE  —  runs once, cached
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳ Loading & processing data…")
def load_and_process():
    df_raw = pd.read_excel("Amazon_Reviews.xlsx")

    custom_stops = (
        list(ENGLISH_STOP_WORDS)
        + ["product","amazon","buy","bought","order","ordered","get","got",
           "one","would","could","also","even","really","item","price",
           "shipping","delivery","star","food","like","good","great",
           "love","taste","very","just","don","its"]
    )

    def clean(text):
        text = str(text).lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df_raw["clean_text"]   = df_raw["Text"].apply(clean)
    df_raw["review_text"]  = df_raw["Text"]
    df_raw["rating"]       = df_raw["Score"]
    df_raw["word_count"]   = df_raw["Text"].apply(lambda x: len(str(x).split()))

    # VADER sentiment
    analyzer = SentimentIntensityAnalyzer()
    df_raw["vader"] = df_raw["Text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )
    df_raw["sentiment"] = df_raw["vader"].apply(
        lambda v: "Positive" if v >= 0.05 else ("Negative" if v <= -0.05 else "Neutral")
    )

    # TF-IDF + KMeans (4 clusters as specified)
    tfidf = TfidfVectorizer(
        max_features=500, min_df=5, max_df=0.85,
        ngram_range=(1, 2), stop_words=custom_stops
    )
    tfidf_matrix = tfidf.fit_transform(df_raw["clean_text"])

    km = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    df_raw["cluster"] = km.fit_predict(tfidf_matrix)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    df_raw["pca_x"] = reduced[:, 0]
    df_raw["pca_y"] = reduced[:, 1]

    # Top keywords per cluster from centroids
    terms = tfidf.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    top_kw = {
        i: [terms[j] for j in order_centroids[i, :12]]
        for i in range(4)
    }

    return df_raw, tfidf, km, top_kw


df, tfidf_model, kmeans_model, top_keywords = load_and_process()

# ═══════════════════════════════════════════════════════════════════════════════
#  PERSONA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
PERSONAS = {
    0: {
        "name":  "Budget Buyers",
        "icon":  "🛒",
        "color": "#3b82f6",
        "bg":    "#eff6ff",
        "desc":  (
            "The largest segment (72.3% of reviews). Price-conscious shoppers who buy "
            "everyday snacks, chips, and candy in bulk. Broadly satisfied (Avg ★4.17) "
            "but switch brands easily for deals. Short, transactional reviews averaging "
            "73 words. Strong response to bundles and Subscribe & Save."
        ),
        "strategy": (
            "<b>Priority: Medium.</b> Launch bulk-buy discounts and Subscribe & Save "
            "offers. Use high review volume from this segment as social proof on listings. "
            "A/B test packaging — this group is the most responsive to visual changes. "
            "Retarget with seasonal promotions (holiday snack bundles, back-to-school deals)."
        ),
        "sat": 77,
    },
    1: {
        "name":  "Quality Seekers",
        "icon":  "☕",
        "color": "#f59e0b",
        "bg":    "#fffbeb",
        "desc":  (
            "Coffee and specialty beverage connoisseurs (6.4% of reviews). Highly satisfied "
            "(Avg ★4.38) with strong brand loyalty to specific roasts, pods, and flavors. "
            "Reviews mention Senseo, French Roast, vanilla — precise and educated. "
            "Moderate word count (82 words). Very low complaint rate."
        ),
        "strategy": (
            "<b>Priority: Low — Nurture & expand.</b> Highlight roast intensity, origin, "
            "and brewing method specs on all coffee listings. Create a 'Specialty Coffee' "
            "storefront section. Offer sampler packs for new SKUs to this segment first. "
            "Invite top reviewers to brand ambassador or Vine programme."
        ),
        "sat": 85,
    },
    2: {
        "name":  "Dissatisfied Users",
        "icon":  "⚠️",
        "color": "#ef4444",
        "bg":    "#fef2f2",
        "desc":  (
            "Ginger tea and herbal beverage buyers who expected stronger, more authentic "
            "flavor (15.8% of reviews). Lowest satisfaction (Avg ★3.54) and highest "
            "percentage of Negative reviews. Write detailed critical reviews (avg 128 words) "
            "— these are Amazon's most at-risk and vocal customers."
        ),
        "strategy": (
            "<b>Priority: HIGH — Immediate action needed.</b> Rewrite product descriptions "
            "to set accurate flavor intensity expectations. Add a Mild/Medium/Bold scale "
            "to herbal tea listings. Trigger a follow-up email with replacement offer for "
            "all 1–2 star reviews in this cluster. Audit ginger tea SKUs for quality "
            "consistency issues."
        ),
        "sat": 57,
    },
    3: {
        "name":  "Loyal Customers",
        "icon":  "💛",
        "color": "#10b981",
        "bg":    "#f0fdf4",
        "desc":  (
            "Health-focused peanut butter and protein powder buyers (5.5% of reviews). "
            "Highest satisfaction of all segments (Avg ★4.48, VADER 0.66). Passionate, "
            "detailed reviewers who write about low-fat, high-protein benefits. Brand-loyal "
            "to PB2 and similar products. Natural brand ambassadors."
        ),
        "strategy": (
            "<b>Priority: Low — Leverage as advocates.</b> Recruit top reviewers into "
            "an ambassador programme with early product access. Feature their reviews "
            "in A+ content and marketing materials. Build a loyalty tier with exclusive "
            "health-food bundles. Cross-sell protein bars, supplements, and sports nutrition."
        ),
        "sat": 88,
    },
}

CLUSTER_COLORS = {
    0: "#3b82f6",
    1: "#f59e0b",
    2: "#ef4444",
    3: "#10b981",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PRODUCT CAROUSEL DATA
# ═══════════════════════════════════════════════════════════════════════════════
PRODUCTS = [
    {"title": "Emerald Cocoa Roast Almonds 11oz",       "price": "$8.49",  "stars": "★★★★★", "badge": "Best Seller",  "img": "https://placehold.co/180x120/e8f5e9/2e7d32?text=🍫+Almonds"},
    {"title": "PB2 Powdered Peanut Butter 16oz",         "price": "$9.99",  "stars": "★★★★★", "badge": "Top Pick",     "img": "https://placehold.co/180x120/fff9e6/e65100?text=🥜+PB2"},
    {"title": "Senseo Coffee Pods Dark Roast ×36",       "price": "$14.99", "stars": "★★★★☆", "badge": "Popular",      "img": "https://placehold.co/180x120/3e2723/fff?text=☕+Coffee"},
    {"title": "Gold Kili Ginger Lemon Beverage ×20",     "price": "$6.75",  "stars": "★★★☆☆", "badge": "Review It",    "img": "https://placehold.co/180x120/e0f7fa/006064?text=🍋+Ginger"},
    {"title": "Nature Valley Almond Peanut Bars ×12",    "price": "$7.25",  "stars": "★★★★★", "badge": "Fan Favourite","img": "https://placehold.co/180x120/f3e5f5/6a1b9a?text=🌾+Bars"},
    {"title": "Stash Premium Green Tea ×100",            "price": "$11.49", "stars": "★★★★☆", "badge": "Health Pick",  "img": "https://placehold.co/180x120/e8f5e9/1b5e20?text=🍵+Tea"},
    {"title": "Lindt Excellence Dark Chocolate 70%",     "price": "$5.99",  "stars": "★★★★★", "badge": "Best Seller",  "img": "https://placehold.co/180x120/3e2723/ffcc80?text=🍫+Dark"},
    {"title": "PG Tips Pyramid Tea Bags ×240",           "price": "$18.99", "stars": "★★★★★", "badge": "UK Favourite", "img": "https://placehold.co/180x120/fff3e0/bf360c?text=🫖+PG+Tips"},
    {"title": "Lay's Classic Chips Variety Pack ×28",    "price": "$19.49", "stars": "★★★★☆", "badge": "Value Pack",   "img": "https://placehold.co/180x120/fffde7/f57f17?text=🥔+Chips"},
    {"title": "Pedigree Good Bites Skin & Coat ×40",     "price": "$12.29", "stars": "★★★★★", "badge": "Pet Fave",     "img": "https://placehold.co/180x120/e1f5fe/01579b?text=🐾+Treats"},
    {"title": "Emerald Cocoa Roast Almonds 11oz",        "price": "$8.49",  "stars": "★★★★★", "badge": "Best Seller",  "img": "https://placehold.co/180x120/e8f5e9/2e7d32?text=🍫+Almonds"},
    {"title": "PB2 Powdered Peanut Butter 16oz",         "price": "$9.99",  "stars": "★★★★★", "badge": "Top Pick",     "img": "https://placehold.co/180x120/fff9e6/e65100?text=🥜+PB2"},
    {"title": "Senseo Coffee Pods Dark Roast ×36",       "price": "$14.99", "stars": "★★★★☆", "badge": "Popular",      "img": "https://placehold.co/180x120/3e2723/fff?text=☕+Coffee"},
    {"title": "Gold Kili Ginger Lemon Beverage ×20",     "price": "$6.75",  "stars": "★★★☆☆", "badge": "Review It",    "img": "https://placehold.co/180x120/e0f7fa/006064?text=🍋+Ginger"},
    {"title": "Nature Valley Almond Peanut Bars ×12",    "price": "$7.25",  "stars": "★★★★★", "badge": "Fan Favourite","img": "https://placehold.co/180x120/f3e5f5/6a1b9a?text=🌾+Bars"},
]

# ═══════════════════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def chart_cluster_donut():
    counts = df["cluster"].value_counts().sort_index()
    labels = [f"{PERSONAS[i]['icon']} {PERSONAS[i]['name']}" for i in counts.index]
    colors = [CLUSTER_COLORS[i] for i in counts.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=counts.values,
        hole=0.58, marker_colors=colors,
        textinfo="percent", textfont_size=12,
        hovertemplate="<b>%{label}</b><br>Reviews: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", font_size=12, x=1.0),
        showlegend=True,
        annotations=[dict(text=f"<b>{len(df):,}</b><br>reviews", x=0.5, y=0.5,
                          font_size=14, showarrow=False, font_color="#111827")],
    )
    return fig


def chart_sentiment_stacked():
    order = ["Positive", "Neutral", "Negative"]
    colors_s = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}
    data = df.groupby(["cluster", "sentiment"]).size().unstack(fill_value=0)
    for c in order:
        if c not in data: data[c] = 0
    data = data[order]
    pct = data.div(data.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    for s in order:
        fig.add_trace(go.Bar(
            name=s,
            x=[f"{PERSONAS[i]['icon']} {PERSONAS[i]['name']}" for i in pct.index],
            y=pct[s].values,
            marker_color=colors_s[s],
            text=[f"{v:.0f}%" for v in pct[s].values],
            textposition="inside", textfont_size=11,
            hovertemplate=f"<b>{s}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18, font_size=12),
        xaxis=dict(tickfont_size=11, gridcolor="#f3f4f6"),
        yaxis=dict(title="% of Reviews", gridcolor="#f3f4f6",
                   ticksuffix="%", tickfont_size=11),
    )
    return fig


def chart_pca_scatter():
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="pca_x", y="pca_y",
        color=sample["cluster"].map(lambda c: f"{PERSONAS[c]['icon']} {PERSONAS[c]['name']}"),
        color_discrete_map={
            f"{PERSONAS[c]['icon']} {PERSONAS[c]['name']}": CLUSTER_COLORS[c]
            for c in CLUSTER_COLORS
        },
        opacity=0.55, size_max=6,
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2",
                "color": "Persona"},
        hover_data={"pca_x": False, "pca_y": False},
        custom_data=["rating", "sentiment"],
    )
    fig.update_traces(
        marker=dict(size=5),
        hovertemplate="<b>Rating:</b> %{customdata[0]}★<br>"
                      "<b>Sentiment:</b> %{customdata[1]}<extra></extra>",
    )
    fig.update_layout(
        height=440, margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
        legend=dict(orientation="v", font_size=12, x=1.01),
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(gridcolor="#e5e7eb", zeroline=False),
    )
    return fig


def chart_rating_dist():
    fig = go.Figure()
    for cid, p in PERSONAS.items():
        sub = df[df["cluster"] == cid]
        counts = sub["rating"].value_counts().sort_index()
        fig.add_trace(go.Bar(
            name=f"{p['icon']} {p['name']}",
            x=[f"★{i}" for i in counts.index],
            y=counts.values,
            marker_color=p["color"],
            opacity=0.85,
            hovertemplate="<b>" + p["name"] + "</b><br>%{x}: %{y:,}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=320,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22, font_size=11),
        xaxis=dict(gridcolor="#f3f4f6"),
        yaxis=dict(title="Reviews", gridcolor="#f3f4f6"),
    )
    return fig


def chart_wordcloud(cluster_id):
    wc_stops = {
        "product","amazon","one","would","could","really","also","even",
        "this","that","they","them","have","with","just","very","from",
        "will","get","got","not","but","was","are","the","and","for","you",
        "good","great","like","food","taste","flavor","don","much","when",
        "more","about","what","than","been","some","other","only","were",
        "those","your","all","its","can","has","there","which","how",
    }
    text = " ".join(df[df["cluster"] == cluster_id]["clean_text"].tolist())
    cmaps = {0: "Blues", 1: "Oranges", 2: "Reds", 3: "Greens"}
    wc = WordCloud(
        width=700, height=300,
        background_color="#ffffff",
        stopwords=wc_stops,
        colormap=cmaps[cluster_id],
        max_words=70, min_font_size=10,
        collocations=False,
        prefer_horizontal=0.8,
    ).generate(text)
    fig_wc, ax = plt.subplots(figsize=(9, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig_wc.patch.set_alpha(0)
    buf = io.BytesIO()
    fig_wc.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                   transparent=True)
    buf.seek(0)
    plt.close(fig_wc)
    return buf


def chart_sentiment_pie(cluster_id):
    sub = df[df["cluster"] == cluster_id]
    counts = sub["sentiment"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker_colors=["#10b981", "#f59e0b", "#ef4444"],
        hole=0.5, textinfo="percent+label", textfont_size=12,
    ))
    fig.update_layout(
        height=240, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  CHATBOT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def chatbot_response(user_msg: str) -> str:
    msg = user_msg.lower().strip()

    if any(w in msg for w in ["best persona", "best cluster", "happiest", "top segment"]):
        return (
            "**💛 Loyal Customers** (Cluster 3) is your best-performing persona.\n\n"
            "- **Satisfaction rate:** 88%\n"
            "- **Avg Rating:** ★4.48\n"
            "- **VADER Score:** 0.66 (most positive language)\n"
            "- **Focus:** Health-oriented PB2 and protein products\n\n"
            "These customers are your brand ambassadors — recruit them into a "
            "loyalty programme immediately."
        )

    if any(w in msg for w in ["worst", "dissatisfied", "problem", "low rating", "unhappy", "negative"]):
        return (
            "**⚠️ Dissatisfied Users** (Cluster 2) needs urgent attention.\n\n"
            "- **Satisfaction rate:** 57% — lowest of all 4 segments\n"
            "- **Avg Rating:** ★3.54\n"
            "- **Focus:** Ginger & herbal teas with unmet flavor expectations\n"
            "- **Avg review length:** 128 words (unhappy customers write more)\n\n"
            "**Immediate actions:** Rewrite product descriptions with flavor "
            "intensity scales and trigger follow-up offers for 1–2 star reviewers."
        )

    if any(w in msg for w in ["strategy", "recommend", "suggest", "improve", "action"]):
        return (
            "Here are the **top 4 strategic priorities** based on NLP cluster analysis:\n\n"
            "1. 🔴 **Dissatisfied Users** → Fix ginger tea product descriptions, "
            "add flavor intensity ratings, issue replacement offers\n"
            "2. 🟡 **Budget Buyers** → Launch Subscribe & Save bundles, seasonal promos\n"
            "3. 🟡 **Quality Seekers** → Highlight coffee certifications, create "
            "specialty storefront section\n"
            "4. 🟢 **Loyal Customers** → Recruit as brand ambassadors, cross-sell "
            "protein & health products"
        )

    if any(w in msg for w in ["how many", "count", "size", "total", "number"]):
        counts = df["cluster"].value_counts().sort_index()
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** {counts.get(i,0):,} reviews "
            f"({counts.get(i,0)/len(df)*100:.1f}%)"
            for i in range(4)
        ])
        return f"**Dataset breakdown — 4,000 total reviews:**\n\n{lines}"

    if any(w in msg for w in ["sentiment", "positive", "vader"]):
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** "
            f"{df[df['cluster']==i]['vader'].mean():.3f} VADER · "
            f"{df[df['cluster']==i]['is_satisfied'].mean()*100 if 'is_satisfied' in df.columns else PERSONAS[i]['sat']:.0f}% satisfied"
            for i in range(4)
        ])
        return f"**Sentiment scores per persona:**\n\n{lines}"

    if any(w in msg for w in ["keyword", "topic", "words", "talk about"]):
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** "
            f"{', '.join(top_keywords[i][:6])}"
            for i in range(4)
        ])
        return f"**Top keywords per cluster (from TF-IDF centroids):**\n\n{lines}"

    if any(w in msg for w in ["hello", "hi", "hey", "help"]):
        return (
            "👋 Hello! I'm your **Customer Persona AI Assistant**.\n\n"
            "I can answer questions like:\n"
            "- *Who is the best/worst persona?*\n"
            "- *What strategy should I use?*\n"
            "- *How many customers per cluster?*\n"
            "- *What are the top keywords?*\n"
            "- *What is the sentiment breakdown?*\n\n"
            "Ask me anything about the 4 customer personas!"
        )

    return (
        "I can help with questions about **personas, sentiment, strategy, "
        "keywords, or cluster sizes**. Try asking:\n"
        "- *'What is the best persona?'*\n"
        "- *'Which cluster needs urgent attention?'*\n"
        "- *'Give me strategy suggestions'*"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CAROUSEL HTML
# ═══════════════════════════════════════════════════════════════════════════════
def render_carousel():
    cards_html = ""
    for p in PRODUCTS:
        cards_html += f"""
        <div class="product-card">
          <img src="{p['img']}" class="product-img" alt="{p['title']}" onerror="this.style.background='#e5e7eb';this.src=''"/>
          <div class="product-title">{p['title']}</div>
          <div class="product-stars">{p['stars']}</div>
          <div class="product-price">{p['price']}</div>
          <div><span class="product-badge">{p['badge']}</span></div>
        </div>"""

    return f"""
    <div class="carousel-wrapper">
      <div style="font-size:14px;font-weight:700;color:#111827;
                  padding: 0 0 12px 4px;letter-spacing:-0.2px">
        🛒 Frequently Reviewed Products
      </div>
      <div class="carousel-track">
        {cards_html}
      </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  TOP NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_navbar():
    now  = datetime.datetime.now().strftime("%I:%M %p")
    date = datetime.datetime.now().strftime("%b %d, %Y")
    st.markdown(f"""
    <div class="top-navbar">
      <div class="navbar-brand">🛒 Customer <span>Persona</span> Intelligence</div>
      <div class="navbar-search">
        <input type="text" placeholder="Search products, personas, keywords…" />
        <button class="navbar-search-btn">🔍</button>
      </div>
      <div class="navbar-profile">
        <div>
          <div class="navbar-time">🕐 {now} &nbsp;|&nbsp; {date}</div>
          <div class="navbar-loc">📍 Hyderabad, IN</div>
        </div>
        <div class="navbar-avatar">A</div>
        <div>
          <div style="font-weight:600;color:#fff;font-size:13px">Analyst</div>
          <div style="font-size:10px;color:#9ca3af">MBA · BA III</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 8px 10px 8px;text-align:center;">
          <div style="font-size:28px">🧠</div>
          <div style="font-size:15px;font-weight:700;color:#fff;
                      letter-spacing:-0.3px;margin-top:4px">
            NLP Persona Platform</div>
          <div style="font-size:11px;color:#9ca3af;margin-top:2px">
            MBA Business Analytics</div>
        </div>
        <hr style="border-color:#2d3748;margin:10px 0 16px 0"/>
        """, unsafe_allow_html=True)

        nav_items = [
            ("🏠", "Home"),
            ("📊", "Analytics"),
            ("👤", "Personas"),
            ("🤖", "AI Assistant"),
        ]
        if "page" not in st.session_state:
            st.session_state.page = "Home"

        for icon, label in nav_items:
            active_style = (
                "background:#ff9900;color:#131921 !important;font-weight:700;"
                if st.session_state.page == label
                else "color:#d1d5db;"
            )
            clicked = st.button(
                f"  {icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
            )
            if clicked:
                st.session_state.page = label
                st.rerun()

        st.markdown("""
        <hr style="border-color:#2d3748;margin:20px 0 12px 0"/>
        <div style="padding:0 8px;">
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:0.6px;margin-bottom:10px;">Dataset Stats</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Total Reviews",   f"{len(df):,}")
        st.metric("Unique Products", f"{df['ProductId'].nunique():,}")
        st.metric("Personas Found",  "4")
        st.metric("Avg ★ Rating",    f"{df['rating'].mean():.2f}")

        st.markdown("""
        <hr style="border-color:#2d3748;margin:16px 0 12px 0"/>
        <div style="font-size:10px;color:#6b7280;text-align:center;padding-bottom:10px">
          NLP Final Project · Phase 4<br>K-Means + LDA + VADER
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home():
    pad = "padding: 24px 32px 0 32px"

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Key Performance Indicators</div>',
                unsafe_allow_html=True)

    total       = len(df)
    pct_pos     = (df["sentiment"] == "Positive").mean() * 100
    n_clusters  = df["cluster"].nunique()
    avg_rating  = df["rating"].mean()
    pct_neg     = (df["sentiment"] == "Negative").mean() * 100

    kpi_data = [
        ("#3b82f6", "👥", "Total Reviews",     f"{total:,}",        "across 4 personas"),
        ("#10b981", "😊", "Positive Sentiment",f"{pct_pos:.1f}%",   f"Neg: {pct_neg:.1f}%"),
        ("#f59e0b", "⭐", "Avg Star Rating",   f"{avg_rating:.2f}", "out of 5.00"),
        ("#8b5cf6", "🗂️",  "Clusters Found",    f"{n_clusters}",     "via K-Means NLP"),
        ("#ef4444", "⚠️", "At-Risk Reviews",   f"{(df['cluster']==2).sum():,}", "Dissatisfied Users"),
    ]
    cols = st.columns(5)
    for col, (color, icon, label, value, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color}">
              <div>
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
              </div>
              <div>
                <div class="kpi-value" style="color:{color}">{value}</div>
                <div class="kpi-sub">{sub}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Carousel ─────────────────────────────────────────────────────────────
    st.markdown(f'<div style="{pad};padding-top:24px">', unsafe_allow_html=True)
    st.markdown(render_carousel(), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────────────────
    st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Cluster & Sentiment Overview</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card"><b style="font-size:14px;color:#374151">Cluster Distribution</b>', unsafe_allow_html=True)
        st.plotly_chart(chart_cluster_donut(), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card"><b style="font-size:14px;color:#374151">Sentiment per Persona (%)</b>', unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_stacked(), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Rating distribution ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">⭐ Star Rating Distribution per Persona</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(chart_rating_dist(), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Quick summary table ───────────────────────────────────────────────────
    st.markdown(f'<div style="{pad};padding-bottom:32px">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂️ Persona Summary Table</div>',
                unsafe_allow_html=True)
    rows = []
    for cid, p in PERSONAS.items():
        sub = df[df["cluster"] == cid]
        rows.append({
            "Persona": f"{p['icon']}  {p['name']}",
            "Reviews": f"{len(sub):,}",
            "Share":   f"{len(sub)/len(df)*100:.1f}%",
            "Avg ★":   f"{sub['rating'].mean():.2f}",
            "% Positive": f"{(sub['sentiment']=='Positive').mean()*100:.1f}%",
            "% Negative": f"{(sub['sentiment']=='Negative').mean()*100:.1f}%",
            "Avg Words":  f"{sub['word_count'].mean():.0f}",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    pad = "padding: 24px 32px"
    st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 PCA Cluster Scatter Plot</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:12px;color:#6b7280;margin-bottom:8px">
    TF-IDF vectors (500 dimensions) compressed to 2D via PCA.
    Each dot = one review. Distance represents textual similarity.
    Sample of 1,500 reviews shown for performance.
    </p>""", unsafe_allow_html=True)
    st.plotly_chart(chart_pca_scatter(), use_container_width=True,
                    config={"displayModeBar": True})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Top keywords grid ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏷️ Top Keywords per Cluster</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for cid, p in PERSONAS.items():
        with cols[cid]:
            kws = top_keywords[cid]
            tags = "".join([f'<span class="kw-tag">{w}</span>' for w in kws])
            st.markdown(f"""
            <div style="background:{p['bg']};border-radius:14px;padding:18px;
                        border-top:4px solid {p['color']};min-height:200px;">
              <div style="font-size:22px;margin-bottom:4px">{p['icon']}</div>
              <div style="font-size:13px;font-weight:700;color:{p['color']};
                          margin-bottom:12px">{p['name']}</div>
              <div>{tags}</div>
            </div>""", unsafe_allow_html=True)

    # ── Vader distribution ───────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:28px">📉 VADER Compound Score Distribution</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_v = go.Figure()
    for cid, p in PERSONAS.items():
        sub = df[df["cluster"] == cid]
        fig_v.add_trace(go.Violin(
            y=sub["vader"], name=f"{p['icon']} {p['name']}",
            box_visible=True, meanline_visible=True,
            fillcolor=p["color"], opacity=0.6,
            line_color=p["color"],
        ))
    fig_v.update_layout(
        height=340, margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="VADER Score", gridcolor="#e5e7eb",
                   zeroline=True, zerolinecolor="#d1d5db"),
        showlegend=False,
    )
    st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PERSONAS
# ═══════════════════════════════════════════════════════════════════════════════
def page_personas():
    pad = "padding: 24px 32px"
    st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Customer Persona Explorer</div>',
                unsafe_allow_html=True)

    selected_name = st.selectbox(
        "Select a customer persona to explore:",
        [f"{PERSONAS[i]['icon']}  {PERSONAS[i]['name']}" for i in range(4)],
        key="persona_select",
    )
    cid = next(i for i in range(4)
               if f"{PERSONAS[i]['icon']}  {PERSONAS[i]['name']}" == selected_name)
    p   = PERSONAS[cid]
    sub = df[df["cluster"] == cid]

    # ── Persona header card ──────────────────────────────────────────────────
    sat_pct  = (sub["sentiment"] == "Positive").mean() * 100
    neg_pct  = (sub["sentiment"] == "Negative").mean() * 100
    neu_pct  = (sub["sentiment"] == "Neutral").mean() * 100

    st.markdown(f"""
    <div class="persona-card" style="border-top-color:{p['color']}">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
        <span style="font-size:40px">{p['icon']}</span>
        <div>
          <div class="persona-name" style="color:{p['color']}">{p['name']}</div>
          <div style="font-size:12px;color:#6b7280">
            Cluster {cid} &nbsp;·&nbsp; {len(sub):,} reviews
            ({len(sub)/len(df)*100:.1f}% of dataset)
          </div>
        </div>
      </div>
      <p class="persona-desc">{p['desc']}</p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
        <span class="persona-badge"
              style="background:{p['color']}20;color:{p['color']}">
          ⭐ Avg Rating: {sub['rating'].mean():.2f}</span>
        <span class="persona-badge" style="background:#dcfce7;color:#166534">
          😊 Positive: {sat_pct:.1f}%</span>
        <span class="persona-badge" style="background:#fee2e2;color:#991b1b">
          😞 Negative: {neg_pct:.1f}%</span>
        <span class="persona-badge" style="background:#fef9c3;color:#854d0e">
          😐 Neutral: {neu_pct:.1f}%</span>
        <span class="persona-badge" style="background:#f3f4f6;color:#374151">
          📝 Avg Words: {sub['word_count'].mean():.0f}</span>
      </div>
      <div class="strategy-box">
        <div style="font-size:12px;font-weight:700;color:#1d4ed8;
                    text-transform:uppercase;letter-spacing:0.4px;
                    margin-bottom:6px">💡 Strategy Recommendation</div>
        {p['strategy']}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="chart-card"><b style="font-size:13px;color:#374151">Sentiment Breakdown</b>', unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_pie(cid), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card"><b style="font-size:13px;color:#374151">Rating Distribution</b>', unsafe_allow_html=True)
        sub_counts = sub["rating"].value_counts().sort_index()
        fig_r = go.Figure(go.Bar(
            x=[f"★{i}" for i in sub_counts.index],
            y=sub_counts.values,
            marker_color=p["color"], marker_opacity=0.85,
            text=sub_counts.values, textposition="outside",
        ))
        fig_r.update_layout(
            height=230, margin=dict(t=10, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6"),
            showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Word cloud ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">☁️ Word Cloud</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    wc_buf = chart_wordcloud(cid)
    st.image(wc_buf, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sample reviews ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📝 Sample Reviews from this Persona</div>',
                unsafe_allow_html=True)
    score_filter = st.slider("Filter by star rating:", 1, 5, (1, 5), key="score_slider")
    filtered = sub[
        (sub["rating"] >= score_filter[0]) & (sub["rating"] <= score_filter[1])
    ].sample(min(5, len(sub[
        (sub["rating"] >= score_filter[0]) & (sub["rating"] <= score_filter[1])
    ])), random_state=42)

    for _, row in filtered.iterrows():
        stars = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
        summary = str(row.get("Summary", "")).strip()[:70]
        text    = str(row["review_text"])[:280].strip()
        vader_v = row["vader"]
        sent_color = "#10b981" if vader_v > 0.05 else ("#ef4444" if vader_v < -0.05 else "#f59e0b")
        st.markdown(f"""
        <div style="background:#ffffff;border-radius:10px;padding:14px 18px;
                    margin-bottom:10px;box-shadow:0 1px 6px rgba(0,0,0,0.07);
                    border-left:4px solid {p['color']}">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <span style="color:{p['color']};font-weight:700;font-size:14px">{stars}</span>
            <span style="font-size:13px;font-weight:600;color:#111827">{summary}</span>
            <span style="margin-left:auto;font-size:11px;font-weight:600;
                         color:{sent_color}">VADER: {vader_v:.3f}</span>
          </div>
          <div style="font-size:12px;color:#6b7280;line-height:1.6">{text}…</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
def page_ai_assistant():
    pad = "padding: 24px 32px"
    st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Persona Assistant</div>',
                unsafe_allow_html=True)

    # Info banner
    st.markdown("""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                padding:14px 18px;margin-bottom:20px;font-size:13px;color:#1e40af">
      <b>💡 Powered by NLP cluster analysis.</b> Ask me about personas, sentiment,
      business strategy, keyword themes, or cluster statistics.
      Try: <i>"Who is the best persona?"</i> · <i>"Which cluster needs urgent attention?"</i>
      · <i>"Give me strategy suggestions"</i>
    </div>""", unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "👋 Hello! I'm your **Customer Persona AI Assistant**.\n\n"
             "I've analysed **4,000 Amazon food reviews** and identified "
             "**4 customer personas** using K-Means clustering, LDA topic modeling, "
             "and VADER sentiment analysis.\n\n"
             "Ask me anything — try *'What is the best persona?'* or "
             "*'Give me strategy suggestions'*.")
        ]

    # Render chat
    chat_container = st.container()
    with chat_container:
        for role, msg in st.session_state.chat_history:
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg)

    # Quick-action buttons
    st.markdown("**Quick Questions:**")
    qcols = st.columns(4)
    quick_qs = [
        "Best persona?",
        "Worst cluster?",
        "Strategy suggestions",
        "Sentiment breakdown",
    ]
    for col, q in zip(qcols, quick_qs):
        with col:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                st.session_state.chat_history.append(("user", q))
                st.session_state.chat_history.append(
                    ("bot", chatbot_response(q))
                )
                st.rerun()

    # Chat input
    user_input = st.chat_input("Ask about personas, strategy, keywords, sentiment…")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(
            ("bot", chatbot_response(user_input))
        )
        st.rerun()

    # Clear button
    if len(st.session_state.chat_history) > 1:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = st.session_state.chat_history[:1]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════════
render_navbar()
render_sidebar()

page = st.session_state.get("page", "Home")

if page == "Home":
    page_home()
elif page == "Analytics":
    page_analytics()
elif page == "Personas":
    page_personas()
elif page == "AI Assistant":
    page_ai_assistant()
