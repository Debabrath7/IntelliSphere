# ==============================================
# IntelliSphere: AI-Powered Insight Dashboard
# Author: Debabrath
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
from dateutil import tz
import backend_modules as bm
from backend_modules import (
    get_stock_data,
    get_trends_keywords,
    fetch_github_trending,
    fetch_arxiv_papers,
    get_news,
    analyze_headlines_sentiment,
)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide")

# ---------------------------
# Session state init
# ---------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
if "last_symbol" not in st.session_state:
    st.session_state["last_symbol"] = None
if "last_period" not in st.session_state:
    st.session_state["last_period"] = None
if "last_info" not in st.session_state:
    st.session_state["last_info"] = None

# ---------------------------
# CSS Styling (White Theme)
# ---------------------------
st.markdown("""
<style>
.stApp { background: #ffffff; color: #0b1220; font-family: 'Inter', sans-serif; }
h1,h2,h3,h4 { color: #0b1220; }
header, footer {visibility: hidden;}
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e6e9ee;
    padding: 14px 28px;
    display:flex;
    justify-content:space-between;
    align-items:center;
}
.brand { display:flex; gap:12px; align-items:center; }
.logo {
    width:42px; height:42px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(90deg,#00b894,#00a8ff);
    color:white; font-weight:700; font-size:17px;
}
.nav { display:flex; gap:10px; align-items:center; }
.nav button {
    border:none; background:#f3f4f6; border-radius:6px; padding:6px 14px;
    font-weight:600; color:#1e293b; cursor:pointer;
}
.nav button:hover { background:#e2e8f0; }
.nav button.active {
    background:#2563eb; color:white;
}
.card {
    background:#fbfdff; border:1px solid #e5e7eb; border-radius:10px;
    padding:18px; box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}
.metric-title { color:#6b7280; font-size:13px; margin-bottom:4px; }
.metric-value { font-weight:700; font-size:22px; color:#0b1220; }
.metric-sub { color:#4b5563; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Navigation Bar
# ---------------------------
def navbar():
    pages = [
        ("Home", "home"),
        ("Stocks", "stocks"),
        ("Trends", "trends"),
        ("Research", "research"),
        ("Skills", "skills"),
        ("News", "news"),
        ("Feedback", "feedback"),
    ]

    st.markdown("<div class='topbar'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,5])
    with col1:
        st.markdown("<div class='brand'><div class='logo'>IS</div><div><b>IntelliSphere</b><br><small style='color:#6b7280;'>AI-Powered Insights</small></div></div>", unsafe_allow_html=True)
    with col2:
        nav_cols = st.columns(len(pages))
        for (label, key), c in zip(pages, nav_cols):
            active = st.session_state.page == key
            if c.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Utility Functions
# ---------------------------
def humanize_number(num):
    try:
        num = float(num)
    except Exception:
        return "N/A"
    absn = abs(num)
    if absn >= 1e12:
        return f"{num/1e12:.2f} Tn"
    if absn >= 1e7:
        return f"{num/1e7:.2f} Cr"
    if absn >= 1e5:
        return f"{num/1e5:.2f} L"
    if absn >= 1e3:
        return f"{num/1e3:.2f} K"
    return f"{num:.2f}"

def format_dividend(div_y):
    if not div_y:
        return "N/A"
    d = float(div_y)
    if d < 1:
        d *= 100
    return f"{d:.2f}%"

# ---------------------------
# Home
# ---------------------------
def home():
    st.title("Welcome to IntelliSphere 🌐")
    st.markdown("Your unified AI dashboard for **stocks**, **trends**, **research**, and **news insights** — all in one modern platform.")
    st.info("Navigate using the top bar to explore each section.")

# ---------------------------
# Stocks Page
# ---------------------------
def stocks():
    st.header("Stock Insights Dashboard")

    col1, col2, col3 = st.columns([3,2,2])
    with col1:
        symbol = st.text_input("Enter company symbol (e.g., TCS or INFY.NS):")
    with col2:
        period = st.selectbox("Select time range:", ["1mo", "3mo", "6mo", "1y"], index=1)
    with col3:
        chart_type = st.radio("Chart Type:", ["Line", "Candlestick"], horizontal=True)

    c1, c2 = st.columns(2)
    with c1:
        show_ema = st.checkbox("Show EMA (12 & 26)", value=True)
    with c2:
        show_rsi = st.checkbox("Show RSI (14)", value=False)

    if st.button("Fetch Stock Data"):
        try:
            df = get_stock_data(symbol, period=period)
            if df is None or df.empty:
                st.warning("No data found.")
                return

            info = bm.yf.Ticker(symbol).info
            latest_price = df["Close"].iloc[-1]
            high_52 = info.get("fiftyTwoWeekHigh", 0)
            low_52 = info.get("fiftyTwoWeekLow", 0)
            pe = info.get("trailingPE", "N/A")
            mcap = info.get("marketCap", 0)
            div = format_dividend(info.get("dividendYield"))

            # cards
            cols = st.columns(4)
            cols[0].markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='metric-value'>₹{latest_price:.2f}</div></div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div class='card'><div class='metric-title'>P/E Ratio</div><div class='metric-value'>{pe}</div></div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div class='card'><div class='metric-title'>Market Cap</div><div class='metric-value'>{humanize_number(mcap)}</div></div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div class='card'><div class='metric-title'>Dividend Yield</div><div class='metric-value'>{div}</div></div>", unsafe_allow_html=True)

            # Chart
            if chart_type == "Candlestick":
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
            else:
                fig = px.line(df, x=df.index, y="Close")

            if show_ema:
                df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
                df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
                fig.add_scatter(x=df.index, y=df["EMA12"], mode="lines", name="EMA12")
                fig.add_scatter(x=df.index, y=df["EMA26"], mode="lines", name="EMA26")

            fig.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            if show_rsi:
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                fig_rsi = px.line(x=rsi.index, y=rsi, title="RSI (14)")
                fig_rsi.update_layout(template="plotly_white", height=200)
                st.plotly_chart(fig_rsi, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching stock data: {e}")

# ---------------------------
# Other Pages
# ---------------------------
def trends():
    st.header("Tech & Market Trends")
    st.info("Coming soon — AI-driven market and technology trends.")

def research():
    st.header("Research & Analytics")
    st.info("Coming soon — AI-curated research papers and insights.")

def skills():
    st.header("Skills & Learning")
    st.info("Coming soon — career growth and learning recommendations.")

def news():
    st.header("News & Sentiment Analysis")
    st.info("Coming soon — AI-based news sentiment tracker.")

def feedback():
    st.header("Feedback")
    name = st.text_input("Name:")
    feedback = st.text_area("Your Feedback:")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

# ---------------------------
# Render dashboard
# ---------------------------
def render_dashboard():
    navbar()
    st.markdown("<div style='padding:20px'>", unsafe_allow_html=True)
    page = st.session_state.page
    if page == "home":
        home()
    elif page == "stocks":
        stocks()
    elif page == "trends":
        trends()
    elif page == "research":
        research()
    elif page == "skills":
        skills()
    elif page == "news":
        news()
    elif page == "feedback":
        feedback()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    render_dashboard()
