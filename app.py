# ═══════════════════════════════════════════════════════════════════════════════
#  Customer Persona Intelligence Platform  (NLP-Based)
#  Author  : MBA Business Analytics — NLP Final Project
#  Deploy  : streamlit run app.py
#  Cloud   : share.streamlit.io  (upload app.py + Amazon_Reviews.xlsx)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import warnings
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# ── Must be first Streamlit call ──────────────────────────────────────────────
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f5f5f5 !important;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background: #f5f5f5;
}
section[data-testid="stSidebar"] {
    background: #131921 !important;
    border-right: 1px solid #232f3e;
}
section[data-testid="stSidebar"] .stMetric label,
section[data-testid="stSidebar"] .stMetric div {
    color: #d1d5db !important;
}
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

/* ── Navbar ── */
.top-navbar {
    background: #131921;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    position: sticky;
    top: 0;
    z-index: 999;
}
.navbar-brand {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    white-space: nowrap;
    line-height: 1.2;
}
.navbar-brand span { color: #ff9900; }
.navbar-tagline { font-size: 10px; color: #9ca3af; letter-spacing: 0.3px; }
.navbar-search {
    flex: 1;
    max-width: 520px;
    background: #ffffff;
    border-radius: 6px;
    display: flex;
    align-items: center;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.navbar-search input {
    flex: 1; border: none; outline: none;
    padding: 9px 14px; font-size: 13px;
    font-family: 'Inter', sans-serif;
    background: transparent; color: #111;
}
.navbar-search-btn {
    background: #ff9900; border: none;
    cursor: pointer; padding: 9px 16px;
    font-size: 15px; color: #131921;
    font-weight: 700;
}
.navbar-profile {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #d1d5db;
}
.navbar-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ff9900, #e67e00);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px; color: #131921;
    box-shadow: 0 2px 6px rgba(255,153,0,0.4);
}
.navbar-info { text-align: right; }
.navbar-name { font-size: 13px; font-weight: 600; color: #fff; }
.navbar-meta { font-size: 10px; color: #9ca3af; }

/* ── Page content wrapper ── */
.page-wrap { padding: 24px 32px 40px 32px; }

/* ── Section titles ── */
.section-title {
    font-size: 17px; font-weight: 700; color: #111827;
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #e5e7eb;
    display: flex; align-items: center; gap: 8px;
    letter-spacing: -0.2px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 80px; height: 80px;
    border-radius: 50%;
    opacity: 0.06;
    transform: translate(20px, -20px);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.12);
}
.kpi-icon { font-size: 28px; margin-bottom: 8px; display: block; }
.kpi-label {
    font-size: 11px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.7px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 36px; font-weight: 800;
    color: #111827; line-height: 1;
    letter-spacing: -1px;
}
.kpi-sub { font-size: 11px; color: #9ca3af; margin-top: 6px; }

/* ── Chart Cards ── */
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}
.chart-card-title {
    font-size: 14px; font-weight: 700; color: #374151;
    margin-bottom: 14px; display: flex; align-items: center; gap: 6px;
}

/* ── Carousel ── */
.carousel-outer {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 0 20px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    overflow: hidden;
    margin-bottom: 24px;
}
.carousel-label {
    font-size: 14px; font-weight: 700; color: #111827;
    padding-bottom: 14px; padding-left: 4px;
    display: flex; align-items: center; gap: 6px;
}
.carousel-track {
    display: flex;
    gap: 14px;
    width: max-content;
    animation: scrollLeft 30s linear infinite;
}
.carousel-track:hover { animation-play-state: paused; }
@keyframes scrollLeft {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.product-card {
    background: #f9fafb;
    border-radius: 12px;
    padding: 12px;
    width: 175px;
    flex-shrink: 0;
    border: 1px solid #e5e7eb;
    transition: transform 0.22s ease, box-shadow 0.22s ease, background 0.2s;
    cursor: pointer;
    text-align: center;
}
.product-card:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0 14px 32px rgba(0,0,0,0.14);
    background: #ffffff;
}
.product-img {
    width: 100%; height: 110px;
    border-radius: 8px; margin-bottom: 10px;
    object-fit: cover; background: #e5e7eb;
    display: block;
}
.product-title {
    font-size: 11.5px; font-weight: 600;
    color: #111827; line-height: 1.35;
    margin-bottom: 4px; min-height: 32px;
}
.product-price { font-size: 15px; font-weight: 800; color: #B12704; }
.product-stars { font-size: 12px; color: #f59e0b; margin: 3px 0; }
.product-badge {
    display: inline-block;
    background: #ff9900; color: #131921;
    font-size: 9px; font-weight: 800;
    padding: 2px 7px; border-radius: 3px;
    text-transform: uppercase; margin-top: 4px;
    letter-spacing: 0.3px;
}

/* ── Persona Cards ── */
.persona-hero {
    background: #ffffff;
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.09);
    border-top: 6px solid;
    margin-bottom: 20px;
}
.persona-name {
    font-size: 22px; font-weight: 800;
    letter-spacing: -0.5px; margin-bottom: 3px;
}
.persona-meta { font-size: 12px; color: #6b7280; margin-bottom: 14px; }
.persona-desc {
    font-size: 13px; color: #4b5563;
    line-height: 1.7; margin-bottom: 16px;
}
.badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 11px;
    font-weight: 700; margin: 3px 2px;
}
.strategy-box {
    background: #f0f7ff;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px; color: #1e3a5f;
    line-height: 1.65; margin-top: 14px;
}
.strategy-box .strat-title {
    font-size: 11px; font-weight: 700;
    color: #1d4ed8; text-transform: uppercase;
    letter-spacing: 0.6px; margin-bottom: 6px;
}

/* ── Review cards ── */
.review-item {
    background: #ffffff;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    border-left: 4px solid;
}
.review-stars { font-weight: 700; font-size: 14px; }
.review-summary { font-size: 13px; font-weight: 600; color: #111827; }
.review-text { font-size: 12px; color: #6b7280; line-height: 1.6; margin-top: 4px; }

/* ── Chat ── */
.chat-container {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    min-height: 400px;
    margin-bottom: 16px;
}
.chat-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px; color: #1e40af;
    margin-bottom: 18px;
    line-height: 1.6;
}

/* ── Keyword tags ── */
.kw-tag {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 2px;
    border: 1px solid #e5e7eb;
    transition: background 0.15s;
}

/* ── Analytics info box ── */
.info-box {
    background: #f8faff;
    border: 1px solid #dbeafe;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 12px; color: #374151;
    line-height: 1.6;
    margin-bottom: 14px;
}

/* ── Sidebar nav buttons ── */
.stButton > button {
    background: transparent !important;
    color: #d1d5db !important;
    border: none !important;
    text-align: left !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: background 0.15s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background: #232f3e !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PIPELINE  —  cached, runs once
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳  Processing 4,000 reviews — please wait…")
def load_and_process():
    raw = pd.read_excel("Amazon_Reviews.xlsx")

    stops = (list(ENGLISH_STOP_WORDS) +
             ["product","amazon","buy","bought","order","ordered",
              "get","got","one","would","could","also","even","really",
              "item","price","shipping","delivery","star","food","like",
              "good","great","love","taste","don","its","just","ve","re"])

    def clean(t):
        t = str(t).lower()
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    raw["clean_text"]  = raw["Text"].apply(clean)
    raw["review_text"] = raw["Text"]
    raw["rating"]      = raw["Score"]
    raw["word_count"]  = raw["Text"].apply(lambda x: len(str(x).split()))

    # VADER
    sia = SentimentIntensityAnalyzer()
    raw["vader"]     = raw["Text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    raw["sentiment"] = raw["vader"].apply(
        lambda v: "Positive" if v >= 0.05 else ("Negative" if v <= -0.05 else "Neutral"))

    # TF-IDF
    vec = TfidfVectorizer(max_features=500, min_df=5, max_df=0.85,
                          ngram_range=(1, 2), stop_words=stops)
    M = vec.fit_transform(raw["clean_text"])

    # KMeans  (4 clusters as specified)
    km = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    raw["cluster"] = km.fit_predict(M)

    # PCA
    reduced   = PCA(n_components=2, random_state=42).fit_transform(M.toarray())
    raw["pca_x"] = reduced[:, 0]
    raw["pca_y"] = reduced[:, 1]

    # Top keywords per cluster from TF-IDF centroids
    terms = vec.get_feature_names_out()
    oc    = km.cluster_centers_.argsort()[:, ::-1]
    kw    = {i: [terms[j] for j in oc[i, :12]] for i in range(4)}

    return raw, kw


df, TOP_KW = load_and_process()


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSONA DICTIONARY   (Cluster 0→Budget Buyers … 3→Loyal Customers)
# ═══════════════════════════════════════════════════════════════════════════════
PERSONAS = {
    0: {
        "name":     "Budget Buyers",
        "icon":     "🛒",
        "color":    "#3b82f6",
        "bg":       "#eff6ff",
        "desc": (
            "Snack and confectionery enthusiasts who shop for value — chips, almonds, "
            "chocolate, kettle corn and mixed nuts. The second-largest segment (13.2% of "
            "reviews) with a solid average rating of ★4.25. Reviews are concise (~90 words) "
            "and transactional. Price-sensitive and highly responsive to bundle deals."
        ),
        "strategy": (
            "<b>Priority: Medium.</b> Drive repeat purchase with Subscribe & Save discounts "
            "on snack variety packs. Seasonal limited-edition flavours (holiday chocolate, "
            "summer kettle corn) convert this segment strongly. Use their high review volume "
            "as social proof. A/B test packaging — this group is the most visually swayable."
        ),
    },
    1: {
        "name":     "Quality Seekers",
        "icon":     "💛",
        "color":    "#10b981",
        "bg":       "#f0fdf4",
        "desc": (
            "Health-focused peanut butter and protein powder buyers — the highest-rated "
            "persona (★4.48 avg, VADER 0.66). Centred on PB2 powdered peanut butter: "
            "calorie-conscious, high-protein, low-fat messaging resonates deeply. Smallest "
            "segment (5.5% of reviews) but the most enthusiastic and loyal. Natural brand "
            "ambassadors who voluntarily promote products."
        ),
        "strategy": (
            "<b>Priority: Low — Leverage as brand advocates.</b> Recruit top reviewers "
            "into an ambassador or Amazon Vine programme. Give early access to new health "
            "and protein product launches. Feature their reviews in A+ content. Cross-sell "
            "protein bars, supplements, and sports nutrition to this segment."
        ),
    },
    2: {
        "name":     "Dissatisfied Users",
        "icon":     "⚠️",
        "color":    "#ef4444",
        "bg":       "#fef2f2",
        "desc": (
            "Ginger tea and herbal beverage buyers who expected stronger, more authentic "
            "flavour — the lowest-rated persona (★3.53 avg). Write the longest reviews "
            "in the dataset (~119 words) because dissatisfied customers explain in detail. "
            "Key complaint: products described as 'strong ginger' or 'bold lemon' tasted "
            "weak and watered-down. Amazon's most at-risk and vocal customer group (15.4%)."
        ),
        "strategy": (
            "<b>Priority: HIGH — Immediate action required.</b> Rewrite herbal tea product "
            "descriptions with a Mild / Medium / Bold flavour-intensity scale. Trigger an "
            "automated follow-up with a replacement offer for all 1–2 star reviews in this "
            "cluster. Audit ginger tea SKUs for batch-level quality inconsistency. Add "
            "customer photos showing brew strength to set accurate visual expectations."
        ),
    },
    3: {
        "name":     "Loyal Customers",
        "icon":     "☕",
        "color":    "#f59e0b",
        "bg":       "#fffbeb",
        "desc": (
            "The largest group (65.9%) — broadly satisfied everyday shoppers covering "
            "coffee, general snacks, pet treats, and home staples. Average rating ★4.17 "
            "with short, positive reviews (~72 words). No single strong product theme — "
            "these are Amazon's general food & grocery buyers who shop across categories. "
            "Broadly loyal but not deeply emotionally invested in any single brand."
        ),
        "strategy": (
            "<b>Priority: Medium — Retain and upsell.</b> Subscribe & Save is the primary "
            "retention lever for this segment. Cross-category recommendations ('Customers "
            "also bought') perform well here. Highlight fast delivery and fresh stock on "
            "listings. Seasonal grocery bundle promotions convert this group reliably."
        ),
    },
}

COLORS = {0: "#3b82f6", 1: "#10b981", 2: "#ef4444", 3: "#f59e0b"}


# ═══════════════════════════════════════════════════════════════════════════════
#  PRODUCT CAROUSEL DATA
# ═══════════════════════════════════════════════════════════════════════════════
PRODUCTS = [
    {"title": "Emerald Cocoa Roast Almonds 11oz",    "price": "$8.49",  "stars": "★★★★★", "badge": "Best Seller",
     "img": "https://placehold.co/175x110/e8f5e9/2e7d32?text=🍫+Almonds"},
    {"title": "PB2 Powdered Peanut Butter 16oz",      "price": "$9.99",  "stars": "★★★★★", "badge": "Top Pick",
     "img": "https://placehold.co/175x110/fff9e6/e65100?text=🥜+PB2"},
    {"title": "Senseo Coffee Pods Dark Roast ×36",    "price": "$14.99", "stars": "★★★★☆", "badge": "Popular",
     "img": "https://placehold.co/175x110/3e2723/fff8e1?text=☕+Coffee"},
    {"title": "Gold Kili Ginger Lemon Drink ×20",     "price": "$6.75",  "stars": "★★★☆☆", "badge": "Review It",
     "img": "https://placehold.co/175x110/e0f7fa/006064?text=🍋+Ginger"},
    {"title": "Nature Valley Almond Bars ×12",        "price": "$7.25",  "stars": "★★★★★", "badge": "Fan Fave",
     "img": "https://placehold.co/175x110/f3e5f5/6a1b9a?text=🌾+Bars"},
    {"title": "Stash Premium Green Tea ×100",         "price": "$11.49", "stars": "★★★★☆", "badge": "Health Pick",
     "img": "https://placehold.co/175x110/e8f5e9/1b5e20?text=🍵+Tea"},
    {"title": "Lindt Excellence 70% Dark 3.5oz",      "price": "$5.99",  "stars": "★★★★★", "badge": "Best Seller",
     "img": "https://placehold.co/175x110/3e2723/ffcc80?text=🍫+Dark"},
    {"title": "PG Tips Pyramid Bags ×240",            "price": "$18.99", "stars": "★★★★★", "badge": "UK Classic",
     "img": "https://placehold.co/175x110/fff3e0/bf360c?text=🫖+PG+Tips"},
    {"title": "Lay's Classic Chips Variety ×28",      "price": "$19.49", "stars": "★★★★☆", "badge": "Value Pack",
     "img": "https://placehold.co/175x110/fffde7/f57f17?text=🥔+Chips"},
    {"title": "Pedigree Good Bites Skin & Coat ×40",  "price": "$12.29", "stars": "★★★★★", "badge": "Pet Fave",
     "img": "https://placehold.co/175x110/e1f5fe/01579b?text=🐾+Treats"},
    # duplicate set so CSS infinite loop looks seamless
    {"title": "Emerald Cocoa Roast Almonds 11oz",    "price": "$8.49",  "stars": "★★★★★", "badge": "Best Seller",
     "img": "https://placehold.co/175x110/e8f5e9/2e7d32?text=🍫+Almonds"},
    {"title": "PB2 Powdered Peanut Butter 16oz",      "price": "$9.99",  "stars": "★★★★★", "badge": "Top Pick",
     "img": "https://placehold.co/175x110/fff9e6/e65100?text=🥜+PB2"},
    {"title": "Senseo Coffee Pods Dark Roast ×36",    "price": "$14.99", "stars": "★★★★☆", "badge": "Popular",
     "img": "https://placehold.co/175x110/3e2723/fff8e1?text=☕+Coffee"},
    {"title": "Gold Kili Ginger Lemon Drink ×20",     "price": "$6.75",  "stars": "★★★☆☆", "badge": "Review It",
     "img": "https://placehold.co/175x110/e0f7fa/006064?text=🍋+Ginger"},
    {"title": "Nature Valley Almond Bars ×12",        "price": "$7.25",  "stars": "★★★★★", "badge": "Fan Fave",
     "img": "https://placehold.co/175x110/f3e5f5/6a1b9a?text=🌾+Bars"},
    {"title": "Stash Premium Green Tea ×100",         "price": "$11.49", "stars": "★★★★☆", "badge": "Health Pick",
     "img": "https://placehold.co/175x110/e8f5e9/1b5e20?text=🍵+Tea"},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — matplotlib figure → base64 PNG
# ═══════════════════════════════════════════════════════════════════════════════
def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    enc = __import__("base64").b64encode(buf.read()).decode()
    plt.close(fig)
    return enc


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def chart_cluster_donut():
    counts = df["cluster"].value_counts().sort_index()
    labels = [f"{PERSONAS[i]['icon']} {PERSONAS[i]['name']}" for i in counts.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=counts.values,
        hole=0.56,
        marker=dict(colors=[COLORS[i] for i in counts.index],
                    line=dict(color="#f5f5f5", width=3)),
        textinfo="percent",
        textfont=dict(size=12, color="#111827"),
        hovertemplate="<b>%{label}</b><br>Reviews: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=8, b=8, l=8, r=8), height=290,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", font=dict(size=11), x=1.0, y=0.5),
        annotations=[dict(
            text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>reviews</span>",
            x=0.5, y=0.5, font=dict(size=14, color="#111827"),
            showarrow=False,
        )],
    )
    return fig


def chart_sentiment_stacked():
    order   = ["Positive", "Neutral", "Negative"]
    pal     = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}
    data    = df.groupby(["cluster", "sentiment"]).size().unstack(fill_value=0)
    for c in order:
        if c not in data: data[c] = 0
    pct = data[order].div(data[order].sum(axis=1), axis=0) * 100
    xlabels = [f"{PERSONAS[i]['icon']} {PERSONAS[i]['name']}" for i in pct.index]
    fig = go.Figure()
    for s in order:
        fig.add_trace(go.Bar(
            name=s, x=xlabels, y=pct[s].values,
            marker_color=pal[s],
            text=[f"{v:.0f}%" for v in pct[s].values],
            textposition="inside", textfont=dict(size=11, color="white"),
            hovertemplate=f"<b>{s}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=290,
        margin=dict(t=8, b=8, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        xaxis=dict(tickfont=dict(size=11), gridcolor="#f3f4f6"),
        yaxis=dict(title="% of Reviews", ticksuffix="%",
                   gridcolor="#f3f4f6", tickfont=dict(size=11)),
    )
    return fig


def chart_rating_dist():
    fig = go.Figure()
    for cid, p in PERSONAS.items():
        sub = df[df["cluster"] == cid]
        vc  = sub["rating"].value_counts().sort_index()
        fig.add_trace(go.Bar(
            name=f"{p['icon']} {p['name']}",
            x=[f"★{i}" for i in vc.index],
            y=vc.values,
            marker_color=p["color"], opacity=0.87,
            hovertemplate="<b>" + p["name"] + "</b><br>%{x}: %{y:,}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(t=8, b=8, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.24, font=dict(size=11)),
        xaxis=dict(gridcolor="#f3f4f6"),
        yaxis=dict(title="Reviews", gridcolor="#f3f4f6"),
    )
    return fig


def chart_pca(sample_n=1200):
    s = df.sample(min(sample_n, len(df)), random_state=42)
    fig = px.scatter(
        s, x="pca_x", y="pca_y",
        color=s["cluster"].map(lambda c: f"{PERSONAS[c]['icon']} {PERSONAS[c]['name']}"),
        color_discrete_map={
            f"{PERSONAS[c]['icon']} {PERSONAS[c]['name']}": COLORS[c]
            for c in COLORS
        },
        opacity=0.5,
        labels={"pca_x": "PCA Component 1  (text similarity)",
                "pca_y": "PCA Component 2  (text similarity)",
                "color": "Persona"},
        custom_data=["rating", "sentiment"],
    )
    fig.update_traces(
        marker=dict(size=5),
        hovertemplate="<b>Rating:</b> %{customdata[0]}★  "
                      "<b>Sentiment:</b> %{customdata[1]}<extra></extra>",
    )
    fig.update_layout(
        height=440,
        margin=dict(t=8, b=8, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa",
        legend=dict(orientation="v", font=dict(size=11), x=1.01),
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(gridcolor="#e5e7eb", zeroline=False),
    )
    return fig


def chart_vader_violin():
    fig = go.Figure()
    for cid, p in PERSONAS.items():
        sub = df[df["cluster"] == cid]
        fig.add_trace(go.Violin(
            y=sub["vader"],
            name=f"{p['icon']} {p['name']}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=p["color"],
            opacity=0.6,
            line_color=p["color"],
        ))
    fig.update_layout(
        height=320,
        margin=dict(t=8, b=8, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="VADER Compound Score",
                   gridcolor="#e5e7eb",
                   zeroline=True, zerolinecolor="#d1d5db"),
        showlegend=False,
    )
    return fig


def chart_sentiment_pie_cluster(cid):
    sub = df[df["cluster"] == cid]["sentiment"].value_counts()
    pal = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}
    fig = go.Figure(go.Pie(
        labels=sub.index, values=sub.values,
        marker=dict(colors=[pal.get(l, "#9ca3af") for l in sub.index],
                    line=dict(color="white", width=2)),
        hole=0.48,
        textinfo="percent+label",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_wordcloud(cid):
    wc_stops = {
        "product","amazon","one","would","could","really","also","even",
        "this","that","they","them","have","with","just","very","from",
        "will","get","got","not","but","was","are","the","and","for",
        "you","good","great","like","food","taste","flavor","don","much",
        "when","more","about","what","than","been","some","other","only",
        "were","those","your","all","its","can","has","there","which",
        "how","into","been","then","they","their","these","time","use",
    }
    cmaps = {0: "Blues", 1: "Greens", 2: "Reds", 3: "Oranges"}
    text = " ".join(df[df["cluster"] == cid]["clean_text"].tolist())
    wc = WordCloud(
        width=800, height=320,
        background_color="white",
        stopwords=wc_stops,
        colormap=cmaps[cid],
        max_words=70,
        min_font_size=9,
        collocations=False,
        prefer_horizontal=0.8,
    ).generate(text)
    fig_wc, ax = plt.subplots(figsize=(10, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig_wc.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    b64 = fig_to_b64(fig_wc)
    return b64


# ═══════════════════════════════════════════════════════════════════════════════
#  CAROUSEL HTML
# ═══════════════════════════════════════════════════════════════════════════════
def render_carousel():
    cards = ""
    for p in PRODUCTS:
        cards += f"""
        <div class="product-card">
          <img src="{p['img']}" class="product-img"
               alt="{p['title']}"
               onerror="this.style.background='#e5e7eb';this.removeAttribute('src')"/>
          <div class="product-title">{p['title']}</div>
          <div class="product-stars">{p['stars']}</div>
          <div class="product-price">{p['price']}</div>
          <div><span class="product-badge">{p['badge']}</span></div>
        </div>"""
    return f"""
    <div class="carousel-outer">
      <div class="carousel-label">🛒 Frequently Reviewed Products</div>
      <div class="carousel-track">{cards}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_navbar():
    now  = datetime.datetime.now().strftime("%I:%M %p")
    date = datetime.datetime.now().strftime("%b %d, %Y")
    st.markdown(f"""
    <div class="top-navbar">
      <div>
        <div class="navbar-brand">🛒 Customer <span>Persona</span> Intelligence</div>
        <div class="navbar-tagline">NLP-Based Customer Analytics Platform</div>
      </div>
      <div class="navbar-search">
        <input type="text" placeholder="Search products, personas, keywords…"/>
        <button class="navbar-search-btn">🔍</button>
      </div>
      <div class="navbar-profile">
        <div class="navbar-info">
          <div class="navbar-name">MBA Analyst</div>
          <div class="navbar-meta">🕐 {now} &nbsp;·&nbsp; {date}</div>
          <div class="navbar-meta">📍 Hyderabad, IN</div>
        </div>
        <div class="navbar-avatar">A</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:22px 8px 14px 8px;text-align:center;">
          <div style="font-size:32px;margin-bottom:6px">🧠</div>
          <div style="font-size:15px;font-weight:700;color:#ffffff;
                      letter-spacing:-0.3px">NLP Persona Platform</div>
          <div style="font-size:11px;color:#9ca3af;margin-top:3px">
            MBA · Business Analytics · III</div>
        </div>
        <hr style="border:none;border-top:1px solid #232f3e;margin:4px 0 16px"/>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;
                    letter-spacing:0.6px;padding:0 8px 8px 8px">Navigation</div>
        """, unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = "Home"

        nav = [("🏠", "Home"), ("📊", "Analytics"),
               ("👤", "Personas"), ("🤖", "AI Assistant")]

        for icon, label in nav:
            active = st.session_state.page == label
            style = ("background:#ff9900 !important;color:#131921 !important;"
                     "font-weight:700 !important;" if active else "")
            if st.button(f"  {icon}  {label}", key=f"nav_{label}",
                         use_container_width=True):
                st.session_state.page = label
                st.rerun()
            if active:
                st.markdown(
                    f"<style>div[data-testid='stButton']:has(button[kind='secondary']"
                    f"[data-testid='baseButton-secondary']){{background:#ff9900}}</style>",
                    unsafe_allow_html=True)

        st.markdown("""
        <hr style="border:none;border-top:1px solid #232f3e;margin:16px 0 14px"/>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;
                    letter-spacing:0.6px;padding:0 8px 10px 8px">Dataset Stats</div>
        """, unsafe_allow_html=True)

        st.metric("Total Reviews",   f"{len(df):,}")
        st.metric("Unique Products", f"{df['ProductId'].nunique():,}")
        st.metric("Personas",        "4")
        st.metric("Avg ★ Rating",    f"{df['rating'].mean():.2f}")

        st.markdown("""
        <hr style="border:none;border-top:1px solid #232f3e;margin:14px 0 10px"/>
        <div style="font-size:10px;color:#6b7280;text-align:center;padding-bottom:14px;
                    line-height:1.6">
          K-Means · LDA · VADER · TF-IDF<br>
          <span style="color:#ff9900">NLP Final Project — Phase 4</span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
def chatbot(msg: str) -> str:
    m = msg.lower().strip()

    if any(w in m for w in ["best", "happiest", "highest", "top", "satisfied"]):
        sub = df[df["cluster"] == 1]
        return (
            "**💛 Quality Seekers** (Cluster 1) is your best-performing persona.\n\n"
            f"- **Reviews:** {len(sub):,} ({len(sub)/len(df)*100:.1f}% of dataset)\n"
            f"- **Avg Rating:** ★{sub['rating'].mean():.2f} — highest of all 4 clusters\n"
            f"- **VADER Score:** {sub['vader'].mean():.3f} — most positive language\n"
            "- **Focus:** PB2 powdered peanut butter, health & protein products\n\n"
            "These are your brand ambassadors — recruit them into a loyalty programme immediately."
        )

    if any(w in m for w in ["worst", "dissatisfied", "problem", "low", "unhappy",
                             "negative", "urgent", "attention", "fix"]):
        sub = df[df["cluster"] == 2]
        return (
            "**⚠️ Dissatisfied Users** (Cluster 2) needs urgent attention.\n\n"
            f"- **Reviews:** {len(sub):,} ({len(sub)/len(df)*100:.1f}% of dataset)\n"
            f"- **Avg Rating:** ★{sub['rating'].mean():.2f} — lowest of all clusters\n"
            f"- **Avg Review Length:** {sub['word_count'].mean():.0f} words "
            f"(unhappy customers write more)\n"
            "- **Root cause:** Ginger & herbal teas with flavour weaker than described\n\n"
            "**Immediate actions:**\n"
            "1. Add Mild/Medium/Bold flavour-intensity scale to herbal tea listings\n"
            "2. Trigger follow-up with replacement offer for all 1–2 star reviews\n"
            "3. Audit ginger tea SKUs for batch quality inconsistency"
        )

    if any(w in m for w in ["strategy", "recommend", "suggest", "improve",
                             "action", "plan", "advice"]):
        return (
            "**📋 Top 4 Strategic Priorities — Ranked by Urgency:**\n\n"
            "1. 🔴 **Dissatisfied Users** — Rewrite herbal tea descriptions, add "
            "flavour intensity scale, issue replacement offers for 1–2 star reviews\n\n"
            "2. 🟡 **Budget Buyers** — Subscribe & Save bundles, seasonal snack "
            "promotions, A/B test packaging\n\n"
            "3. 🟡 **Loyal Customers** — Cross-category recommendations, Subscribe & Save "
            "for grocery staples, delivery speed messaging\n\n"
            "4. 🟢 **Quality Seekers** — Ambassador programme, early product access, "
            "A+ content featuring their reviews"
        )

    if any(w in m for w in ["count", "size", "how many", "number", "total"]):
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** "
            f"{df[df['cluster']==i].shape[0]:,} reviews "
            f"({df[df['cluster']==i].shape[0]/len(df)*100:.1f}%)"
            for i in range(4)
        ])
        return f"**Dataset breakdown — {len(df):,} total reviews:**\n\n{lines}"

    if any(w in m for w in ["sentiment", "vader", "positive", "negative"]):
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** "
            f"VADER {df[df['cluster']==i]['vader'].mean():.3f} · "
            f"{(df[df['cluster']==i]['sentiment']=='Positive').mean()*100:.1f}% positive"
            for i in range(4)
        ])
        return f"**Sentiment scores per persona:**\n\n{lines}"

    if any(w in m for w in ["keyword", "topic", "word", "theme", "talk"]):
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** "
            f"{', '.join(TOP_KW[i][:6])}"
            for i in range(4)
        ])
        return f"**Top keywords per cluster (TF-IDF centroids):**\n\n{lines}"

    if any(w in m for w in ["rating", "star", "score", "avg"]):
        lines = "\n".join([
            f"- **{PERSONAS[i]['icon']} {PERSONAS[i]['name']}:** "
            f"★{df[df['cluster']==i]['rating'].mean():.2f} avg · "
            f"{df[df['cluster']==i]['word_count'].mean():.0f} words avg"
            for i in range(4)
        ])
        return f"**Average ratings and review length per persona:**\n\n{lines}"

    if any(w in m for w in ["hello", "hi", "hey", "help", "what can"]):
        return (
            "👋 Hello! I'm your **Customer Persona AI Assistant**.\n\n"
            "I've analysed **4,000 Amazon food reviews** using K-Means clustering, "
            "LDA topic modeling, and VADER sentiment analysis — discovering "
            "**4 customer personas**.\n\n"
            "**Try asking me:**\n"
            "- *Who is the best persona?*\n"
            "- *Which cluster needs urgent attention?*\n"
            "- *Give me strategy suggestions*\n"
            "- *What are the top keywords?*\n"
            "- *How many reviews per cluster?*\n"
            "- *What is the sentiment breakdown?*"
        )

    return (
        "I can answer questions about **personas, sentiment, strategy, keywords, "
        "ratings, or cluster sizes**. Try:\n"
        "- *'What is the best persona?'*\n"
        "- *'Which cluster needs fixing?'*\n"
        "- *'Give me strategy suggestions'*\n"
        "- *'What are the top keywords?'*"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Key Performance Indicators</div>',
                unsafe_allow_html=True)

    pct_pos = (df["sentiment"] == "Positive").mean() * 100
    pct_neg = (df["sentiment"] == "Negative").mean() * 100

    kpis = [
        ("#3b82f6", "👥", "Total Customers",    f"{len(df):,}",
         f"{df['ProductId'].nunique():,} unique products"),
        ("#10b981", "😊", "Positive Sentiment", f"{pct_pos:.1f}%",
         f"Negative: {pct_neg:.1f}%"),
        ("#f59e0b", "🗂️", "Number of Clusters", "4",
         "Budget · Quality · Dissatisfied · Loyal"),
    ]

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (color, icon, label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color}">
              <div>
                <span class="kpi-icon">{icon}</span>
                <div class="kpi-label">{label}</div>
              </div>
              <div>
                <div class="kpi-value" style="color:{color}">{value}</div>
                <div class="kpi-sub">{sub}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Carousel ──────────────────────────────────────────────────────────────
    st.markdown(render_carousel(), unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Cluster & Sentiment Overview</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="chart-card">
        <div class="chart-card-title">🍩 Cluster Distribution</div>""",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_cluster_donut(), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="chart-card">
        <div class="chart-card-title">📊 Sentiment per Persona (%)</div>""",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_stacked(), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Rating distribution ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">⭐ Star Rating Distribution per Persona</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(chart_rating_dist(), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗂️ Persona Summary Table</div>',
                unsafe_allow_html=True)
    rows = []
    for cid, p in PERSONAS.items():
        sub = df[df["cluster"] == cid]
        rows.append({
            "Persona":       f"{p['icon']}  {p['name']}",
            "Reviews":       f"{len(sub):,}",
            "Share %":       f"{len(sub)/len(df)*100:.1f}%",
            "Avg ★":         f"{sub['rating'].mean():.2f}",
            "% Positive":    f"{(sub['sentiment']=='Positive').mean()*100:.1f}%",
            "% Negative":    f"{(sub['sentiment']=='Negative').mean()*100:.1f}%",
            "Avg VADER":     f"{sub['vader'].mean():.3f}",
            "Avg Words":     f"{sub['word_count'].mean():.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔬 PCA Cluster Scatter Plot</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    TF-IDF vectors (500 dimensions) compressed to 2D using PCA.
    Each dot = one review. Proximity = textual similarity.
    Tight clusters = strong internal coherence. 1,200 reviews sampled for performance.
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(chart_pca(), use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Top keywords grid ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏷️ Top Keywords per Cluster</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for cid, p in PERSONAS.items():
        with cols[cid]:
            tags = "".join([
                f'<span class="kw-tag">{w}</span>'
                for w in TOP_KW[cid]
            ])
            st.markdown(f"""
            <div style="background:{p['bg']};border-radius:14px;padding:18px;
                        border-top:4px solid {p['color']};min-height:220px;">
              <div style="font-size:22px;margin-bottom:5px">{p['icon']}</div>
              <div style="font-size:13px;font-weight:700;color:{p['color']};
                          margin-bottom:12px">{p['name']}</div>
              <div>{tags}</div>
            </div>""", unsafe_allow_html=True)

    # ── VADER violin ──────────────────────────────────────────────────────────
    st.markdown("""<div class="section-title" style="margin-top:28px">
    📉 VADER Sentiment Score Distribution per Persona</div>""",
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Violin plots show the full distribution of VADER compound scores (−1 = very negative,
    +1 = very positive). The inner box shows median and quartiles.
    Dissatisfied Users has the widest spread — high variance in review sentiment.
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(chart_vader_violin(), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Methodology table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📐 NLP Methodology Summary</div>',
                unsafe_allow_html=True)
    methods = pd.DataFrame([
        ["1", "Text Preprocessing",       "NLTK · Regex",          "Tokenise, remove HTML, lemmatize, stopword filter"],
        ["2", "Feature Extraction",        "TF-IDF (500 features)",  "Convert text to 500-dim numeric vectors"],
        ["3", "Clustering",                "K-Means (k=4)",          "Group similar reviews into 4 personas"],
        ["4", "Topic Modeling",            "LDA (5 topics)",         "Find themes within each cluster"],
        ["5", "Sentiment Analysis",        "VADER",                  "Compound score per review (−1 → +1)"],
        ["6", "Dimensionality Reduction",  "PCA (2D)",               "Compress vectors to 2D for visualisation"],
    ], columns=["Step", "Technique", "Tool", "Purpose"])
    st.dataframe(methods, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PERSONAS
# ═══════════════════════════════════════════════════════════════════════════════
def page_personas():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Customer Persona Explorer</div>',
                unsafe_allow_html=True)

    selected = st.selectbox(
        "Select a customer persona:",
        [f"{PERSONAS[i]['icon']}  {PERSONAS[i]['name']}" for i in range(4)],
        key="persona_select",
    )
    cid = next(i for i in range(4)
               if f"{PERSONAS[i]['icon']}  {PERSONAS[i]['name']}" == selected)
    p   = PERSONAS[cid]
    sub = df[df["cluster"] == cid]

    pct_pos = (sub["sentiment"] == "Positive").mean() * 100
    pct_neg = (sub["sentiment"] == "Negative").mean() * 100
    pct_neu = (sub["sentiment"] == "Neutral").mean() * 100

    # ── Persona hero card ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="persona-hero" style="border-top-color:{p['color']}">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
        <span style="font-size:44px;line-height:1">{p['icon']}</span>
        <div>
          <div class="persona-name" style="color:{p['color']}">{p['name']}</div>
          <div class="persona-meta">
            Cluster {cid} &nbsp;·&nbsp; {len(sub):,} reviews
            ({len(sub)/len(df)*100:.1f}% of dataset)
          </div>
        </div>
      </div>
      <p class="persona-desc">{p['desc']}</p>
      <div style="margin-bottom:4px">
        <span class="badge" style="background:{p['color']}18;color:{p['color']}">
          ⭐ Avg Rating: {sub['rating'].mean():.2f}</span>
        <span class="badge" style="background:#dcfce7;color:#166534">
          😊 Positive: {pct_pos:.1f}%</span>
        <span class="badge" style="background:#fee2e2;color:#991b1b">
          😞 Negative: {pct_neg:.1f}%</span>
        <span class="badge" style="background:#fef9c3;color:#854d0e">
          😐 Neutral: {pct_neu:.1f}%</span>
        <span class="badge" style="background:#f3f4f6;color:#374151">
          📝 Avg Words: {sub['word_count'].mean():.0f}</span>
        <span class="badge" style="background:#f3f4f6;color:#374151">
          🎯 VADER: {sub['vader'].mean():.3f}</span>
      </div>
      <div class="strategy-box">
        <div class="strat-title">💡 Strategy Recommendation</div>
        {p['strategy']}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="chart-card">
        <div class="chart-card-title">Sentiment Breakdown</div>""",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_pie_cluster(cid),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="chart-card">
        <div class="chart-card-title">Star Rating Distribution</div>""",
                    unsafe_allow_html=True)
        vc = sub["rating"].value_counts().sort_index()
        fig_r = go.Figure(go.Bar(
            x=[f"★{i}" for i in vc.index], y=vc.values,
            marker_color=p["color"], marker_opacity=0.88,
            text=vc.values, textposition="outside",
        ))
        fig_r.update_layout(
            height=230, margin=dict(t=8, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6"),
            showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Word cloud ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">☁️ Word Cloud</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    with st.spinner("Generating word cloud…"):
        wc_b64 = make_wordcloud(cid)
    st.markdown(
        f'<img src="data:image/png;base64,{wc_b64}" '
        f'style="width:100%;border-radius:8px"/>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sample reviews ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📝 Sample Reviews</div>',
                unsafe_allow_html=True)

    score_filter = st.slider("Filter by star rating:", 1, 5, (1, 5),
                              key="score_slider")
    filtered = sub[
        (sub["rating"] >= score_filter[0]) &
        (sub["rating"] <= score_filter[1])
    ]
    if len(filtered) == 0:
        st.info("No reviews match this filter.")
    else:
        sample = filtered.sample(min(5, len(filtered)), random_state=42)
        for _, row in sample.iterrows():
            stars  = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
            summ   = str(row.get("Summary", "")).strip()[:70]
            text   = str(row["review_text"])[:280].strip()
            v      = row["vader"]
            vcol   = "#10b981" if v > 0.05 else ("#ef4444" if v < -0.05 else "#f59e0b")
            st.markdown(f"""
            <div class="review-item" style="border-left-color:{p['color']}">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
                <span class="review-stars" style="color:{p['color']}">{stars}</span>
                <span class="review-summary">{summ}</span>
                <span style="margin-left:auto;font-size:11px;font-weight:600;
                             color:{vcol}">VADER: {v:.3f}</span>
              </div>
              <div class="review-text">{text}…</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
def page_ai_assistant():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Persona Assistant</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="chat-info">
      <b>💡 Powered by NLP cluster analysis on 4,000 Amazon food reviews.</b><br>
      Ask me about personas, sentiment scores, business strategy, keyword themes,
      or cluster statistics. The answers are derived directly from the K-Means
      and VADER models built in this project.
    </div>""", unsafe_allow_html=True)

    # Initialise chat
    if "chat" not in st.session_state:
        st.session_state.chat = [
            ("assistant",
             "👋 Hello! I'm your **Customer Persona AI Assistant**.\n\n"
             "I've analysed **4,000 Amazon food reviews** and identified "
             "**4 customer personas** using K-Means clustering, LDA topic "
             "modeling, and VADER sentiment analysis.\n\n"
             "**Try asking:**\n"
             "- *Who is the best persona?*\n"
             "- *Which cluster needs urgent attention?*\n"
             "- *Give me strategy suggestions*\n"
             "- *What are the top keywords?*")
        ]

    # Quick action buttons
    st.markdown("**⚡ Quick Questions:**")
    q_cols = st.columns(4)
    quick  = ["Best persona?", "Worst cluster?",
              "Strategy suggestions", "Sentiment breakdown"]
    for col, q in zip(q_cols, quick):
        with col:
            if st.button(q, key=f"q_{q}", use_container_width=True):
                st.session_state.chat.append(("user", q))
                st.session_state.chat.append(("assistant", chatbot(q)))
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Display chat history
    for role, msg in st.session_state.chat:
        with st.chat_message(role,
                             avatar="👤" if role == "user" else "🤖"):
            st.markdown(msg)

    # User input
    user_in = st.chat_input(
        "Ask about personas, strategy, keywords, sentiment…")
    if user_in:
        st.session_state.chat.append(("user", user_in))
        st.session_state.chat.append(("assistant", chatbot(user_in)))
        st.rerun()

    # Clear
    if len(st.session_state.chat) > 1:
        if st.button("🗑️ Clear Chat", key="clear"):
            st.session_state.chat = st.session_state.chat[:1]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
render_navbar()
render_sidebar()

page = st.session_state.get("page", "Home")

if   page == "Home":         page_home()
elif page == "Analytics":    page_analytics()
elif page == "Personas":     page_personas()
elif page == "AI Assistant": page_ai_assistant()
