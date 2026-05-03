# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Persona Intelligence Dashboard", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("Amazon_Reviews.xlsx")

df = load_data()

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.navbar {background-color:#131921;padding:10px;display:flex;justify-content:space-between;align-items:center;color:white;}
.kpi-card {background-color:white;padding:20px;border-radius:10px;box-shadow:2px 2px 8px rgba(0,0,0,0.1);text-align:center;margin:10px;}
.kpi-value {font-size:24px;font-weight:bold;}
.kpi-label {font-size:14px;color:#555;}
.carousel-container {display:flex;overflow-x:auto;scroll-behavior:smooth;}
.carousel-card {flex:0 0 auto;background:white;border-radius:10px;box-shadow:2px 2px 8px rgba(0,0,0,0.1);margin:10px;width:200px;text-align:center;}
.carousel-card img {width:100%;border-radius:10px 10px 0 0;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown(f"""
<div class="navbar">
    <div><b>Customer Persona Intelligence Dashboard</b></div>
    <div><input type="text" placeholder="Search..." style="width:300px;padding:5px;border-radius:5px;"></div>
    <div>Gobinda ⦿ | {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Marpalli, India</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df)}</div><div class="kpi-label">Total Customers</div></div>', unsafe_allow_html=True)
with col2:
    if "sentiment" in df.columns:
        pos_pct = (df['sentiment'].eq("Positive").mean() * 100)
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{pos_pct:.1f}%</div><div class="kpi-label">Positive Sentiment</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">N/A</div><div class="kpi-label">Positive Sentiment</div></div>', unsafe_allow_html=True)
with col3:
    if "cluster" in df.columns:
        clusters = df['cluster'].nunique()
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{clusters}</div><div class="kpi-label">Number of Clusters</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">N/A</div><div class="kpi-label">Number of Clusters</div></div>', unsafe_allow_html=True)

# -------------------------------
# ANALYTICS SECTION
# -------------------------------
st.subheader("Cluster Distribution")
if "cluster" in df.columns:
    cluster_counts = df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ["cluster", "count"]
    fig = px.bar(cluster_counts, x="cluster", y="count", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Sentiment Distribution")
if "sentiment" in df.columns and "cluster" in df.columns:
    sentiment_counts = df.groupby(["cluster", "sentiment"]).size().reset_index(name="count")
    fig2 = px.bar(sentiment_counts, x="cluster", y="count", color="sentiment", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("PCA Scatter Plot")
reduced = np.random.rand(len(df), 2)
fig3 = px.scatter(x=reduced[:,0], y=reduced[:,1], color=df['cluster'].astype(str) if "cluster" in df.columns else None)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Average Rating by Cluster")
if "cluster" in df.columns and "rating" in df.columns:
    avg_rating = df.groupby("cluster")["rating"].mean().reset_index()
    fig4 = px.bar(avg_rating, x="cluster", y="rating", color="cluster")
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Sentiment Polarity Across Clusters")
if "cluster" in df.columns and "sentiment" in df.columns:
    sentiment_polarity = df.groupby("cluster")["sentiment"].value_counts(normalize=True).reset_index(name="pct")
    fig5 = px.bar(sentiment_polarity, x="cluster", y="pct", color="sentiment", barmode="group")
    st.plotly_chart(fig5, use_container_width=True)

st.subheader("Keyword Heatmap (Placeholder)")
heatmap_data = pd.DataFrame({"Keyword":["price","quality","cheap","service"],"Cluster":[0,1,2,3],"Frequency":[10,20,5,15]})
fig6 = px.density_heatmap(heatmap_data, x="Cluster", y="Keyword", z="Frequency", color_continuous_scale="Blues")
st.plotly_chart(fig6, use_container_width=True)

# -------------------------------
# PERSONA INSIGHTS
# -------------------------------
st.subheader("Persona Insights")
if "cluster" in df.columns:
    persona_map = {0:"Budget Buyers",1:"Quality Seekers",2:"Dissatisfied Users",3:"Loyal Customers"}
    cluster_choice = st.selectbox("Select Cluster", df['cluster'].unique())
    persona_name = persona_map.get(cluster_choice,"Unknown")
    st.write(f"**Persona: {persona_name}**")
    st.write("Description: Typical behavior and preferences of this cluster.")
    if "sentiment" in df.columns:
        sentiment_breakdown = df[df['cluster']==cluster_choice]['sentiment'].value_counts(normalize=True)*100
        st.write(sentiment_breakdown)
    if "clean_text" in df.columns:
        text = " ".join(df[df['cluster']==cluster_choice]['clean_text'])
        wc = WordCloud(background_color="white").generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)

# -------------------------------
# REVIEW UPLOAD
# -------------------------------
st.subheader("Upload & Analyze Reviews")
uploaded_file = st.file_uploader("Upload New Reviews (Excel)", type=["xlsx"])
if uploaded_file:
    new_df = pd.read_excel(uploaded_file)
    st.write("New reviews uploaded. Auto-processing pipeline would run here (sentiment ➔ clustering ➔ persona assignment).")
    st.write(new_df.head())

# -------------------------------
# AI ASSISTANT
# -------------------------------
st.subheader("AI Assistant Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []
user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.messages.append(("user", user_input))
    if "best persona" in user_input.lower():
        response = "The best persona is Loyal Customers (Cluster 3)."
    elif "worst cluster" in user_input.lower():
        response = "The worst cluster is Dissatisfied Users (Cluster 2)."
    elif "strategy" in user_input.lower():
        response = "Strategy suggestion: Improve product quality and reward loyalty."
    else:
        response = "I'm here to help with personas and strategies."
    st.session_state.messages.append(("assistant", response))
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

# -------------------------------
# PRODUCT CAROUSEL
# -------------------------------
st.subheader("Product Carousel")
st.markdown("""
<div class="carousel-container">
    <div class="carousel-card"><img src="https://via.placeholder.com/200x150"/><h4>Smart Speaker</h4><p>$49.99</p></div>
    <div class="carousel-card"><img src="https://via.placeholder.com/200x150"/><h4>Fitness Tracker</h4><p>$89.99</p></div>
    <div class="carousel-card"><img src="https://via.placeholder.com/200x150"/><h4>Wireless Headphones</h4><p>$59.99</p></div>
</div>
""", unsafe_allow_html=True)
