# ==============================================================
# frontend.py — IntelliSphere Premium UI
# Author: Debabrath
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Backend imports
import backend_modules as bm
from backend_modules import (
    get_stock_data,
    get_intraday_data,
    fetch_github_trending,
    fetch_arxiv_papers,
    fetch_news_via_google_rss,
    analyze_headlines_sentiment,
)

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | AI Insights",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------
# GLOBAL STYLES
# --------------------------------------------------------------
st.markdown("""
<style>
.card{
    background:white;
    padding:16px;
    border-radius:14px;
    box-shadow:0 6px 24px rgba(0,0,0,0.06);
    margin-bottom:14px;
}
.metric-title{font-size:12px;color:#777}
.metric-value{font-size:22px;font-weight:700}
.big{font-size:30px;font-weight:800}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------
# PERMANENT SIDEBAR (FIX)
# --------------------------------------------------------------
def render_sidebar():
    with st.sidebar:

        # Logo (you can replace with your own)
        st.image("https://i.ibb.co/r2mBjDk/intellis.png", width=90)

        st.title("IntelliSphere")
        st.write("AI-Powered Insights Dashboard")

        page = st.radio(
            "Menu",
            ["Home", "Stocks", "Trends", "Research", "News", "Feedback"],
            index=1
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#999; font-size:13px;'>Made with ❤️ by <b>Debabrath</b></div>",
            unsafe_allow_html=True
        )

    return page


# --------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------
def home():
    st.markdown("<div class='card'><h2>Welcome to IntelliSphere</h2><p>Premium AI dashboard for stock insights.</p></div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------
# STOCKS PAGE (AUTO-UPDATE)
# --------------------------------------------------------------
def stocks():
    st.markdown("<div class='card'><h2>Stocks Dashboard</h2></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        symbol = st.text_input("Enter stock name / ticker / BSE code", "TCS").strip().upper()
    with c2:
        timeframe = st.selectbox(
            "Timeframe",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
        )
    with c3:
        chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"])

    # Auto trigger: no button, fetch instantly
    if not symbol:
        st.warning("Enter a stock symbol.")
        return

    df = get_stock_data(symbol, period=timeframe, interval="1d")

    if df is None or df.empty:
        st.error("No data found for this stock. Try a valid symbol like TCS or RELIANCE.")
        return

    df = df.sort_values("Date").reset_index(drop=True)

    # ------ Metrics ------
    latest = df["Close"].iloc[-1]
    today_high = df["High"].max()
    today_low = df["Low"].min()
    volume_today = df["Volume"].iloc[-1]

    # 52W
    df_52 = get_stock_data(symbol, "1y", "1d")
    if df_52 is not None and not df_52.empty:
        high52 = df_52["High"].max()
        low52 = df_52["Low"].min()
    else:
        high52 = low52 = None

    # SMA50
    sma50 = None
    if len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean().iloc[-1]

    # RSI
    def compute_rsi(series, n=14):
        delta = series.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = -delta.clip(upper=0).rolling(n).mean()
        rs = up / (down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        try:
            return float(rsi.iloc[-1])
        except:
            return None

    rsi = compute_rsi(df["Close"])

    # Render metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='big'>₹{latest:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='metric-title'>Today's High / Low</div><div class='metric-value'>₹{today_high:.2f} / ₹{today_low:.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='metric-title'>52W High</div><div class='metric-value'>{'₹'+format(high52, ',.2f') if high52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><div class='metric-title'>52W Low</div><div class='metric-value'>{'₹'+format(low52, ',.2f') if low52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='card'><div class='metric-title'>SMA50</div><div class='metric-value'>{('₹'+format(sma50,',.2f')) if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    # ---------------- Chart ----------------
    st.markdown("<div class='card'><h3>Price Chart</h3></div>", unsafe_allow_html=True)

    if chart_type == "Candlestick":
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        ))
    else:
        fig = px.line(df, x="Date", y="Close", title=f"{symbol} Price")

    if sma50:
        fig.add_scatter(x=df["Date"], y=df["Close"].rolling(50).mean(), name="SMA50")

    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # RSI CHART
    if rsi:
        with st.expander("RSI Indicator"):
            fig2 = px.line(df, x="Date", y=df["Close"].rolling(14).mean(), title="RSI Trend")
            st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------------------
# Other pages (simple)
# --------------------------------------------------------------
def trends():
    st.markdown("<div class='card'><h2>Trending Repos</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic / Language", "python")
    repos = bm.fetch_github_trending(q)
    for r in repos[:10]:
        st.write(f"**{r['name']}** — {r['description']}")


def research():
    st.markdown("<div class='card'><h2>Research Papers</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search topic", "machine learning")
    papers = bm.fetch_arxiv_papers(q)
    for p in papers:
        st.subheader(p["title"])
        st.write(p["summary"][:300] + "...")
        st.markdown(f"[Read more]({p['link']})")
        st.divider()


def news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic", "Stock Market")
    arts = bm.fetch_news_via_google_rss(q)
    s = bm.analyze_headlines_sentiment(arts)
    for a in s:
        st.markdown(f"**{a['title']}**")
        if a["sentiment"]:
            st.write(f"Sentiment: {a['sentiment']['label']} ({a['sentiment']['score']:.2f})")
        st.markdown(f"[Open]({a['link']})")
        st.divider()


def feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    rating = st.slider("Rate your experience", 1, 5, 4)
    comment = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thank you! Your feedback has been saved.")


# --------------------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------------------
page = render_sidebar()

if page == "Home": home()
elif page == "Stocks": stocks()
elif page == "Trends": trends()
elif page == "Research": research()
elif page == "News": news()
elif page == "Feedback": feedback()
