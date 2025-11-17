# ==============================================================
# frontend.py
# Author: Debabrath
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

# backend imports
import backend_modules as bm
from backend_modules import (
    resolve_ticker_candidates,
    get_stock_data,
    get_intraday_data,
    get_today_high_low,
    fetch_github_trending,
    fetch_arxiv_papers,
    fetch_news_via_google_rss,
    analyze_headlines_sentiment,
)

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | Insights",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------
st.markdown("""
<style>
.card{
    background:white;
    padding:14px;
    border-radius:12px;
    box-shadow:0 6px 24px rgba(0,0,0,0.05);
    margin-bottom:10px;
}
.metric-title{font-size:12px;color:#6b7280}
.metric-value{font-size:22px;font-weight:700}
.big{font-size:28px;font-weight:800}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Lottie loader
# ---------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
def home():
    c1,c2 = st.columns([3,1])
    with c1:
        st.markdown("<div class='card'><h2>Welcome to IntelliSphere</h2>Polished NSE+BSE dashboard</div>", unsafe_allow_html=True)
    with c2:
        ani = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
        if ani: st_lottie(ani, height=150)

# ---------------------------------------------------------
# STOCKS (FINAL FIXED VERSION)
# ---------------------------------------------------------
def stocks():

    st.markdown("<div class='card'><h2>Stocks Dashboard (Unified NSE+BSE)</h2></div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns([3,2,2,2])

    with c1:
        symbol_in = st.text_input("Ticker / Company / BSE Code", "BDL")

    with c2:
        timeframe = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","Intraday (1m, 1d)"])

    with c3:
        chart_type = st.selectbox("Chart", ["Candlestick","Line"])

    with c4:
        overlays = st.multiselect("Indicators", ["RSI","MACD","SMA50"], default=["SMA50"])

    if not st.button("Fetch Data"):
        return

    symbol = symbol_in.strip().upper()

    # ---------------------------------------------------------
    # FIX: USE BACKEND RESOLVER FOR NSE+BSE TICKERS
    # ---------------------------------------------------------
    candidates = resolve_ticker_candidates(symbol)

    # determine intraday or daily
    if timeframe.startswith("Intraday"):
        interval = "1m"
        period = "1d"
        use_intraday = True
    else:
        interval = "1d"
        period = timeframe
        use_intraday = False

    # ---------------------------------------------------------
    # FETCH ALL EXCHANGES SAFELY
    # ---------------------------------------------------------
    dfs = []
    for t in candidates:
        try:
            df = get_intraday_data(t,"1m","1d") if use_intraday else get_stock_data(t,period,interval)
            if df is not None and not df.empty:
                dfs.append((t,df))
        except:
            pass

    if not dfs:
        st.error("No data available for the provided ticker/ID on NSE or BSE. Try a different input like TCS or 500325.")
        return

    # ---------------------------------------------------------
    # SUPER-STABLE MERGE USING CONCAT (NEVER MERGE/JOIN)
    # ---------------------------------------------------------
    base_ticker, base_df = dfs[0]

    merged = base_df.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
    merged.columns = ["Open_0","High_0","Low_0","Close_0","Volume_0"]

    for i, (tk, df2) in enumerate(dfs[1:], start=1):
        tmp = df2.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
        tmp = tmp.add_suffix(f"_{i}")
        merged = pd.concat([merged, tmp], axis=1, join="outer")

    merged = merged.sort_index().fillna(method="ffill")

    # Combined Volume
    vol_cols = [c for c in merged.columns if c.startswith("Volume_")]
    close_cols = [c for c in merged.columns if c.startswith("Close_")]

    merged["TotalVolume"] = merged[vol_cols].sum(axis=1)

    # Combined Close using VWAP-like weighting
    numerator = np.zeros(len(merged))
    for vcol, ccol in zip(vol_cols, close_cols):
        numerator += merged[vcol].values * merged[ccol].values

    merged["Close"] = numerator / np.where(merged["TotalVolume"]>0, merged["TotalVolume"], np.nan)
    merged["Volume"] = merged["TotalVolume"]

    df = merged.reset_index()[["Date","Close","Volume"]].dropna()

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    latest_price = float(df["Close"].iloc[-1])
    today_vol = int(df["Volume"].iloc[-1])
    hi_today = df["Close"].max()
    lo_today = df["Close"].min()

    # 52 WEEK from NSE base
    try:
        df1y = get_stock_data(candidates[0], "1y", "1d")
        hi52 = float(df1y["Close"].max())
        lo52 = float(df1y["Close"].min())
    except:
        hi52 = lo52 = None

    # SMA50
    df_daily = df.set_index("Date").resample("D").agg({"Close": "last", "Volume": "sum"}).dropna()
    sma50 = df_daily["Close"].rolling(50).mean().iloc[-1] if len(df_daily)>=50 else None

    # RSI
    def compute_rsi(series, n=14):
        if len(series)<n+1: return None
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = -delta.clip(upper=0).rolling(n).mean()
        rs = gain/(loss+1e-9)
        return float((100 - 100/(1+rs)).iloc[-1])

    day_rsi = compute_rsi(df_daily["Close"])

    # ---------------------------------------------------------
    # METRIC CARDS
    # ---------------------------------------------------------
    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        st.markdown(f"<div class='card'><div class='metric-title'>Current Price</div>"
                    f"<div class='big'>₹{latest_price:,.2f}</div></div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='card'><div class='metric-title'>Today's High / Low</div>"
                    f"<div class='metric-value'>₹{hi_today:,.2f} / ₹{lo_today:,.2f}</div>"
                    f"<div class='metric-title'>Volume {today_vol:,}</div></div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='card'><div class='metric-title'>52W High</div>"
                    f"<div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"<div class='card'><div class='metric-title'>52W Low</div>"
                    f"<div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    with c5:
        st.markdown(f"<div class='card'><div class='metric-title'>SMA50</div>"
                    f"<div class='metric-value'>{('₹'+format(sma50,',.2f')) if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------
    colL, colR = st.columns([3,1])
    with colL:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df["Date"],
                open=df["Close"],
                high=df["Close"],
                low=df["Close"],
                close=df["Close"]
            )])
        else:
            fig = px.line(df, x="Date", y="Close")

        if sma50:
            fig.add_scatter(x=df_daily.index, y=df_daily["Close"].rolling(50).mean(), name="SMA50")

        fig.update_layout(template="plotly_white", height=480)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Indicators"):
            if "RSI" in overlays:
                st.write(f"RSI: {day_rsi:.2f}" if day_rsi else "RSI unavailable")

            if "MACD" in overlays:
                s=df["Close"]
                ema12=s.ewm(span=12).mean()
                ema26=s.ewm(span=26).mean()
                macd=ema12-ema26
                sig=macd.ewm(span=9).mean()

                fig2=go.Figure()
                fig2.add_scatter(x=df["Date"], y=macd, name="MACD")
                fig2.add_scatter(x=df["Date"], y=sig, name="Signal")
                fig2.update_layout(template="plotly_white", height=180)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- Day RSI: **{day_rsi:.2f}**" if day_rsi else "- RSI unavailable")
        st.markdown(f"- Combined Volume: **{today_vol:,}**")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Trends page
# ---------------------------------------------------------
def trends():
    st.markdown("<div class='card'><h2>Trending Repos</h2></div>", unsafe_allow_html=True)
    q=st.text_input("Topic","python")
    if st.button("Fetch"):
        repos=bm.fetch_github_trending(q)
        for r in repos[:10]:
            st.write(f"**{r['name']}** — {r['description']}")


# ---------------------------------------------------------
# Research page
# ---------------------------------------------------------
def research():
    st.markdown("<div class='card'><h2>Research Papers</h2></div>", unsafe_allow_html=True)
    q=st.text_input("Search","machine learning")
    if st.button("Search"):
        papers=bm.fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:300]+"...")
            st.markdown(f"[Open]({p['link']})")
            st.divider()


# ---------------------------------------------------------
# News page
# ---------------------------------------------------------
def news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q=st.text_input("Topic","Indian stock market")
    if st.button("Get"):
        arts=bm.fetch_news_via_google_rss(q)
        out=bm.analyze_headlines_sentiment(arts)
        for a in out:
            st.subheader(a["title"])
            if a["sentiment"]:
                st.write(a["sentiment"])
            st.markdown(f"[Read]({a['link']})")
            st.divider()


# ---------------------------------------------------------
# Feedback page
# ---------------------------------------------------------
def feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name=st.text_input("Name")
    rate=st.slider("Rate",1,5,4)
    com=st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thanks for your feedback!")


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------
with st.sidebar:
    page = st.radio("Menu", ["Home","Stocks","Trends","Research","News","Feedback"], index=1)

def render_dashboard():
    if page=="Home": home()
    elif page=="Stocks": stocks()
    elif page=="Trends": trends()
    elif page=="Research": research()
    elif page=="News": news()
    elif page=="Feedback": feedback()

if __name__ == "__main__":
    render_dashboard()
