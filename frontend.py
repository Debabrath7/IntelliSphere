# frontend.py
# IntelliSphere - Frontend (dark mode, intraday auto-switch for 1d, improved sidebar)
# Candlestick-only style as requested (cleaner)
# Author: adapted for Debabrath

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil import tz

# backend imports (must match backend_modules.py)
import backend_modules as bm
from backend_modules import (
    get_stock_data,
    get_intraday_data,
    get_today_high_low,
    fetch_github_trending,
    fetch_arxiv_papers,
    fetch_news_via_google_rss,
    analyze_headlines_sentiment,
    stock_summary,
    resolve_ticker_candidates
)

# Page config
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# Styles (dark friendly)
st.markdown("""
<style>
.stApp { background: #0f1720; color: #e6eef6; font-family: Inter, Roboto, Arial, sans-serif; }
h1,h2,h3{color:#e6eef6}
.topbar{background:transparent;padding:12px 0;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;gap:14px;align-items:center}
.logo{width:48px;height:48px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:linear-gradient(90deg,#00b894,#00a8ff);color:#fff;font-weight:700;box-shadow:0 6px 18px rgba(0,0,0,0.06);font-size:18px}
.card{background:#0b1220;border:1px solid rgba(255,255,255,0.03);border-radius:10px;padding:18px;box-shadow:0 6px 20px rgba(17,24,39,0.02)}
.metric-title{color:#9aa3ad;font-size:13px;margin-bottom:6px}
.metric-value{font-weight:700;font-size:20px;color:#e6eef6}
.muted{color:#9aa3ad;font-size:13px}
.small-muted{color:#9aa3ad;font-size:12px}
.sidebar-logo{display:block;margin-left:auto;margin-right:auto}
</style>
""", unsafe_allow_html=True)

# Sidebar custom UI
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; margin-top:10px; margin-bottom:12px;">
            <img src="https://i.imgur.com/Q96QJYV.png" width="72" style="border-radius:10px;display:block;margin-left:auto;margin-right:auto;">
            <h2 style="color:#e6eef6; margin-top:8px; text-align:center;">IntelliSphere</h2>
            <div style="color:#9aa3ad; font-size:13px; margin-top:-6px; text-align:center;">AI-Powered Insights</div>
        </div>
        <hr style="opacity:0.12">
    """, unsafe_allow_html=True)

    page = st.radio("",
        ["Home","Stocks","Trends","Research","News","Feedback"],
        index=1,
        label_visibility="collapsed"
    )

    st.markdown("""
        <div style="position:fixed; bottom:28px; width:85%; text-align:center; color:#9aa3ad; font-size:13px;">
            Made with ❤️ by <b>Debabrath</b>
        </div>
    """, unsafe_allow_html=True)

# Helpers
def ensure_date_col(df):
    if df is None:
        return df
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime":"Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        try:
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index":"Date"})
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass
    return df

def compute_rsi(series, n=14):
    s = series.dropna()
    if len(s) < n+1: return None
    delta = s.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100/(1+rs))
    return rsi

# Stocks page
def render_stock():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Stock Dashboard", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns([3,2,2,2])
    with c1:
        symbol = st.text_input("Stock Symbol / Company / Code", value="TCS")
    with c2:
        timeframe = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","6mo","1y","2y","5y"], index=0)
    with c3:
        chart_type = st.selectbox("Chart Type", ["Candlestick","Line"], index=0)
    with c4:
        overlays = st.multiselect("Indicators", ["RSI","MACD","SMA50"], default=["SMA50"])

    fetch = st.button("Fetch Stock Data")
    if not fetch:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    sym = (symbol or "").strip().upper()
    if not sym:
        st.error("Please enter a ticker or company symbol.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Auto-switch intraday for 1d
    if timeframe == "1d":
        period = "1d"
        interval = "1m"
        use_intraday = True
    else:
        period = timeframe
        interval = "1d"
        use_intraday = False

    # Fetch
    df = None
    try:
        if use_intraday:
            df = get_intraday_data(sym, interval=interval, period=period)
        else:
            df = get_stock_data(sym, period=period, interval=interval)
    except Exception:
        df = None

    if df is None or (hasattr(df,"empty") and df.empty):
        st.warning("Live sources unavailable — loading demo data.")
        try:
            if use_intraday:
                df = get_intraday_data(sym, interval=interval, period=period)
            else:
                df = get_stock_data(sym, period=period, interval=interval)
        except Exception:
            df = None

    if df is None or (hasattr(df,"empty") and df.empty):
        st.error("No data available for provided symbol. Try RELIANCE/TCS/INFY.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = ensure_date_col(df)
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values("Date").reset_index(drop=True)

    # metrics
    try:
        latest_price = float(df["Close"].dropna().iloc[-1])
    except:
        latest_price = None
    try:
        today_vol = int(df["Volume"].dropna().iloc[-1])
    except:
        today_vol = 0

    try:
        if use_intraday:
            df1y = get_stock_data(sym, period="1y", interval="1d")
        else:
            df1y = df.copy()
        if df1y is not None and not df1y.empty:
            hi52 = float(df1y["Close"].max())
            lo52 = float(df1y["Close"].min())
        else:
            hi52 = lo52 = None
    except:
        hi52 = lo52 = None

    try:
        if use_intraday:
            sma50_val = float(df["Close"].dropna().rolling(50).mean().iloc[-1]) if len(df["Close"].dropna())>=50 else None
        else:
            df_daily = df.set_index("Date").resample("D").last().dropna()
            sma50_val = float(df_daily["Close"].rolling(50).mean().iloc[-1]) if len(df_daily)>=50 else None
    except:
        sma50_val = None

    try:
        if use_intraday:
            rsi_series = compute_rsi(df["Close"].dropna(), n=14)
            day_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None
        else:
            series = df.set_index("Date").resample("D").last().dropna()["Close"]
            rsi_series = compute_rsi(series, n=14)
            day_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None
    except:
        day_rsi = None

    # top cards
    tc1,tc2,tc3,tc4 = st.columns(4)
    tc1.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='metric-value'>{('₹'+format(latest_price,',.2f')) if latest_price else 'N/A'}</div></div>", unsafe_allow_html=True)
    tc2.markdown(f"<div class='card'><div class='metric-title'>Today's H / L</div><div class='metric-value'>{('₹'+format(float(df['High'].max()),',.2f')) if not df['High'].isna().all() else 'N/A'} / {('₹'+format(float(df['Low'].min()),',.2f')) if not df['Low'].isna().all() else 'N/A'}</div></div>", unsafe_allow_html=True)
    tc3.markdown(f"<div class='card'><div class='metric-title'>52W High</div><div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    tc4.markdown(f"<div class='card'><div class='metric-title'>52W Low</div><div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # chart & right
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {sym}", unsafe_allow_html=True)

        df_plot = df.copy()
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(x=df_plot["Date"], open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], name="Price")])
        else:
            fig = px.line(df_plot, x="Date", y="Close", title=f"{sym} Price")

        if sma50_val is not None:
            if use_intraday:
                sma_series = df_plot["Close"].rolling(50).mean()
                fig.add_scatter(x=df_plot["Date"], y=sma_series, mode="lines", name="SMA50", line=dict(width=1))
            else:
                df_daily = df_plot.set_index("Date").resample("D").last().dropna()
                if len(df_daily) >= 50:
                    fig.add_scatter(x=df_daily.index, y=df_daily["Close"].rolling(50).mean(), mode="lines", name="SMA50", line=dict(width=1))

        fig.update_layout(template="plotly_dark", height=480, margin=dict(l=10,r=10,t=40,b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Indicators"):
            if "RSI" in overlays:
                st.write(f"RSI: **{day_rsi:.2f}**" if day_rsi else "RSI: N/A")
            if "MACD" in overlays:
                s = df_plot["Close"].dropna()
                if len(s) >= 26:
                    ema12 = s.ewm(span=12).mean()
                    ema26 = s.ewm(span=26).mean()
                    macd = ema12 - ema26
                    sig = macd.ewm(span=9).mean()
                    fig2 = go.Figure()
                    fig2.add_scatter(x=df_plot["Date"], y=macd, name="MACD")
                    fig2.add_scatter(x=df_plot["Date"], y=sig, name="Signal")
                    fig2.update_layout(template="plotly_dark", height=180)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.write("MACD: Insufficient data")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- RSI: **{day_rsi:.2f}**" if day_rsi else "- RSI: N/A")
        st.markdown(f"- Volume: **{today_vol:,}**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Other pages (minimal)
def render_trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","python")
    if st.button("Fetch"):
        repos = fetch_github_trending(q)
        for r in repos[:10]:
            st.markdown(f"**{r['name']}** — {r['description']}")

def render_research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search","machine learning")
    if st.button("Search"):
        papers = fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:300]+"...")
            st.markdown(f"[Open]({p['link']})")

def render_news():
    st.markdown("<div class='card'><h2>News</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","market")
    if st.button("Get"):
        arts = fetch_news_via_google_rss(q, max_items=8)
        sents = analyze_headlines_sentiment(arts)
        for a in sents:
            st.markdown(f"**{a['title']}**")
            if a.get("sentiment"):
                st.markdown(f"- {a['sentiment']['label']} ({a['sentiment']['score']:.2f})")
            st.markdown(f"[Open]({a['link']})")
            st.divider()

def render_feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    rating = st.slider("Rate:",1,5,4)
    comment = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thanks — saved locally.")

# Router / Main
def render_dashboard():
    if page == "Home":
        st.title("Welcome to IntelliSphere")
        st.markdown("Your AI dashboard for stocks, trends, research and news.")
    elif page == "Stocks":
        render_stock()
    elif page == "Trends":
        render_trends()
    elif page == "Research":
        render_research()
    elif page == "News":
        render_news()
    elif page == "Feedback":
        render_feedback()
    else:
        st.title("IntelliSphere")

if __name__ == "__main__":
    render_dashboard()
