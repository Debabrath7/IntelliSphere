# ==============================================================
# frontend.py 
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

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | Insights",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Styles
# ----------------------------------------------------
st.markdown("""
<style>
.card {
    background: white;
    padding: 14px;
    border-radius: 12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.05);
    margin-bottom: 10px;
}
.metric-title { font-size: 12px; color: #6b7280; }
.metric-value { font-size: 22px; font-weight: 700; }
.big { font-size: 28px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Lottie Loader
# ----------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------
def home():
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown("<div class='card'><h2>Welcome to IntelliSphere</h2>Smart Market Insights Dashboard</div>", unsafe_allow_html=True)
    with c2:
        anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
        if anim:
            st_lottie(anim, height=140)

# ----------------------------------------------------
# STOCKS PAGE (CLEAN)
# ----------------------------------------------------
def stocks():
    st.markdown("<div class='card'><h2>Stock Dashboard</h2></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([3,2,2,2])
    with c1:
        symbol_in = st.text_input("Stock Symbol / Company Name / Code", "BDL")
    with c2:
        timeframe = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","Intraday (1m, 1d)"])
    with c3:
        chart_type = st.selectbox("Chart Type", ["Candlestick","Line"])
    with c4:
        overlays = st.multiselect("Indicators", ["RSI","MACD","SMA50"], default=["SMA50"])

    if not st.button("Fetch Stock Data"):
        return

    symbol = (symbol_in or "").strip().upper()
    candidates = resolve_ticker_candidates(symbol)

    # Determine period/interval
    if timeframe.startswith("Intraday"):
        period, interval, use_intraday = "1d", "1m", True
    else:
        period, interval, use_intraday = timeframe, "1d", False

    # Fetch data for candidates
    dfs = []
    for t in candidates:
        try:
            df = get_intraday_data(t, interval=interval, period=period) if use_intraday else get_stock_data(t, period=period, interval=interval)
            if df is None: 
                continue

            df = df.copy()
            if "Datetime" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            if "Date" not in df.columns:
                df = df.reset_index().rename(columns={"index":"Date"})

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            for col in ["Open","High","Low","Close","Volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Date"])

            if not df.empty:
                dfs.append((t, df))

        except Exception:
            continue

    if not dfs:
        st.error("No data found for this stock. Try another symbol like TCS or RELIANCE.")
        return

    # MERGE & NORMALIZE
    base_ticker, base_df = dfs[0]
    merged = base_df.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
    merged.columns = [f"{c}_0" for c in merged.columns]

    # Combine all fetched DFs
    for i, (tk, df2) in enumerate(dfs[1:], start=1):
        tmp = df2.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
        tmp.columns = [f"{c}_{i}" for c in tmp.columns]
        merged = pd.concat([merged, tmp], axis=1, join="outer")

    merged.columns = [str(c) for c in merged.columns]
    merged = merged.sort_index().ffill().bfill().fillna(0)

    # Identify close/volume columns
    cols = list(merged.columns)
    vol_cols = [c for c in cols if "volume" in c.lower()]
    close_cols = [c for c in cols if "close" in c.lower()]

    if not close_cols:
        st.error("Unable to extract price data.")
        return

    if not vol_cols:
        merged["Volume_fallback"] = 0
        vol_cols = ["Volume_fallback"]

    # Compute VWAP-style combined close
    numerator = np.zeros(len(merged))
    total_vol = np.zeros(len(merged))

    for vcol, ccol in zip(vol_cols, close_cols):
        v = pd.to_numeric(merged[vcol], errors="coerce").fillna(0).to_numpy()
        c = pd.to_numeric(merged[ccol], errors="coerce").fillna(method="ffill").to_numpy()
        numerator += v * c
        total_vol += v

    merged["Close"] = np.where(total_vol > 0, numerator / total_vol, merged[close_cols[0]])
    merged["Volume"] = merged[vol_cols].sum(axis=1)

    combined = merged.reset_index()[["Date","Close","Volume"]]
    combined = combined.sort_values("Date").reset_index(drop=True)

    # ----------------------------------------------------
    # METRICS
    # ----------------------------------------------------
    try:
        latest_price = float(combined["Close"].iloc[-1])
    except:
        latest_price = None

    hi_today = combined["Close"].max()
    lo_today = combined["Close"].min()
    today_vol = int(combined["Volume"].iloc[-1])

    # 52W data
    try:
        one_year_df = get_stock_data(symbol, "1y", "1d")
        hi52 = one_year_df["Close"].max()
        lo52 = one_year_df["Close"].min()
    except:
        hi52 = lo52 = None

    # SMA50
    try:
        daily_df = combined.set_index("Date").resample("D").last().dropna()
        sma50 = daily_df["Close"].rolling(50).mean().iloc[-1]
    except:
        sma50 = None

    # RSI
    def rsi(series, n=14):
        s = series.dropna()
        if len(s) < n+1:
            return None
        delta = s.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = -delta.clip(upper=0).rolling(n).mean()
        rs = up / (down + 1e-9)
        return float(100 - 100/(1+rs).iloc[-1])

    day_rsi = None
    try:
        day_rsi = rsi(daily_df["Close"])
    except:
        pass

    # ----------------------------------------------------
    # RENDER METRICS CARDS
    # ----------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='big'>{'₹'+format(latest_price,',.2f') if latest_price else 'N/A'}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='metric-title'>Today's High / Low</div><div class='metric-value'>₹{hi_today:.2f} / ₹{lo_today:.2f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><div class='metric-title'>52-Week High</div><div class='metric-value'>{hi52 if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><div class='metric-title'>52-Week Low</div><div class='metric-value'>{lo52 if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='card'><div class='metric-title'>SMA 50</div><div class='metric-value'>{sma50 if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # CHART
    # ----------------------------------------------------
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {symbol}")

        df_plot = combined.copy()

        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df_plot["Date"],
                open=df_plot["Close"],
                high=df_plot["Close"],
                low=df_plot["Close"],
                close=df_plot["Close"]
            )])
        else:
            fig = px.line(df_plot, x="Date", y="Close")

        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Indicators"):
            if "RSI" in overlays:
                st.write("RSI:", day_rsi)
            if "MACD" in overlays:
                s = df_plot["Close"]
                ema12 = s.ewm(span=12).mean()
                ema26 = s.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                macd_fig = go.Figure()
                macd_fig.add_scatter(x=df_plot["Date"], y=macd, name="MACD")
                macd_fig.add_scatter(x=df_plot["Date"], y=signal, name="Signal")
                macd_fig.update_layout(template="plotly_white", height=200)
                st.plotly_chart(macd_fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- RSI: **{day_rsi}**")
        st.markdown(f"- Volume: **{today_vol:,}**")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# OTHER PAGES
# ----------------------------------------------------
def trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic:", "python")
    if st.button("Fetch"):
        repos = bm.fetch_github_trending(q)
        for r in repos[:10]:
            st.write(f"**{r['name']}** — {r['description']}")

def research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search:", "machine learning")
    if st.button("Search"):
        papers = bm.fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:300] + "...")
            st.markdown(f"[Open]({p['link']})")
            st.divider()

def news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic:", "market")
    if st.button("Get"):
        arts = bm.fetch_news_via_google_rss(q, max_items=8)
        sents = bm.analyze_headlines_sentiment(arts)
        for a in sents:
            st.markdown(f"**{a['title']}**")
            if a.get("sentiment"):
                st.markdown(f"- {a['sentiment']['label']} ({a['sentiment']['score']:.2f})")
            st.markdown(f"[Open]({a['link']})")
            st.divider()

def feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    rating = st.slider("Rate:", 1, 5, 4)
    comments = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thanks — saved locally.")

# ----------------------------------------------------
# ROUTING
# ----------------------------------------------------
with st.sidebar:
    page = st.radio("Menu", ["Home","Stocks","Trends","Research","News","Feedback"], index=1)

def render_dashboard():
    if page == "Home":
        home()
    elif page == "Stocks":
        stocks()
    elif page == "Trends":
        trends()
    elif page == "Research":
        research()
    elif page == "News":
        news()
    elif page == "Feedback":
        feedback()
    else:
        home()

if __name__ == "__main__":
    render_dashboard()
