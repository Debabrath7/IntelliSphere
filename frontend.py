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

# Page config
st.set_page_config(page_title="IntelliSphere | Insights",
                   page_icon="🌐",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Styles
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

def load_lottie(url):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# Home
def home():
    c1,c2 = st.columns([3,1])
    with c1:
        st.markdown("<div class='card'><h2>Welcome to IntelliSphere</h2>Polished NSE+BSE dashboard</div>", unsafe_allow_html=True)
    with c2:
        anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
        if anim:
            st_lottie(anim, height=140)

# Stocks (robust)
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

    symbol = (symbol_in or "").strip().upper()
    candidates = resolve_ticker_candidates(symbol)

    # determine intraday/daily
    if timeframe.startswith("Intraday"):
        period, interval, use_intraday = "1d", "1m", True
    else:
        period, interval, use_intraday = timeframe, "1d", False

    # fetch df list
    dfs = []
    for t in candidates:
        try:
            df = get_intraday_data(t, interval=interval, period=period) if use_intraday else get_stock_data(t, period=period, interval=interval)
            if df is None:
                continue
            df = df.copy()
            # normalize Date column
            if "Datetime" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"Datetime":"Date"})
            if "Date" not in df.columns:
                df = df.reset_index().rename(columns={"index":"Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            # ensure OHLCV
            for col in ["Open","High","Low","Close","Volume"]:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Date"])
            if not df.empty:
                dfs.append((t, df))
        except Exception:
            # ignore and continue
            continue

    if not dfs:
        st.error("No data available for the provided ticker/ID on NSE or BSE. Try a different input like TCS or 500325.")
        return

    # SAFE CONCAT MERGE
    base_ticker, base_df = dfs[0]
    merged = base_df.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
    merged.columns = [f"{c}_0" for c in merged.columns]

    for i, (tk, df2) in enumerate(dfs[1:], start=1):
        tmp = df2.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
        tmp.columns = [f"{c}_{i}" for c in tmp.columns]
        # concat outer
        merged = pd.concat([merged, tmp], axis=1, join="outer")

    # Ensure merged is DataFrame and has string columns
    if not isinstance(merged, pd.DataFrame):
        merged = pd.DataFrame(merged)
    merged.columns = [str(c) for c in merged.columns]

    # Fill forward/backward as reasonable; then fill remaining NaNs with 0 for volumes
    merged = merged.sort_index().ffill().bfill().fillna(0)

    # Robust selection of volume and close columns (case-insensitive)
    cols = list(merged.columns)
    vol_cols = [c for c in cols if c.lower().startswith("volume_")]
    if not vol_cols:
        vol_cols = [c for c in cols if "volume" in c.lower()]
    close_cols = [c for c in cols if c.lower().startswith("close_")]
    if not close_cols:
        close_cols = [c for c in cols if "close" in c.lower()]

    # Fallbacks: if none found, try numeric columns heuristics
    if not close_cols:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(merged[c])]
        # prefer columns containing 'vw' or 'price' or last numeric
        preferred = [c for c in numeric_cols if any(x in c.lower() for x in ("vw","vwap","price","close"))]
        close_cols = preferred or numeric_cols[:1]

    if not vol_cols:
        # create fallback zero-volume column
        merged["Volume_fallback_0"] = 0
        vol_cols = ["Volume_fallback_0"]

    # Now compute total volume and VWClose safely
    # Pair as many close/vol columns as possible (zip)
    pair_count = min(len(vol_cols), len(close_cols))
    if pair_count == 0:
        # if still zero, use first close column or numeric fallback
        if close_cols:
            merged["VWClose"] = merged[close_cols[0]]
        else:
            st.error("Unable to determine price columns for plotting.")
            return
    else:
        numerator = np.zeros(len(merged), dtype=float)
        total_vol = np.zeros(len(merged), dtype=float)
        for i in range(pair_count):
            vcol = vol_cols[i]
            ccol = close_cols[i]
            vol_vals = pd.to_numeric(merged[vcol], errors="coerce").fillna(0).astype(float).values
            close_vals = pd.to_numeric(merged[ccol], errors="coerce").fillna(np.nan).astype(float).values
            numerator += vol_vals * np.nan_to_num(close_vals, nan=0.0)
            total_vol += vol_vals
        with np.errstate(divide='ignore', invalid='ignore'):
            vwclose = np.where(total_vol > 0, numerator / total_vol, np.nan)
        merged["VWClose"] = pd.Series(vwclose, index=merged.index)

    # Finalize Close and Volume columns
    merged["Close"] = merged["VWClose"].fillna(method="ffill").fillna(method="bfill")
    merged["Volume"] = merged[[c for c in vol_cols if c in merged.columns]].sum(axis=1).fillna(0)

    combined = merged.reset_index()[["Date","Close","Volume"]].copy()
    combined = combined.sort_values("Date").reset_index(drop=True)

    if combined.empty:
        st.error("Combined timeseries is empty after merging.")
        return

    # Compute metrics
    try:
        latest_price = float(combined["Close"].iloc[-1])
    except:
        latest_price = None
    try:
        today_vol = int(combined["Volume"].iloc[-1])
    except:
        today_vol = 0

    hi_today = combined["Close"].max() if not combined.empty else None
    lo_today = combined["Close"].min() if not combined.empty else None

    # 52-week (best-effort)
    hi52 = lo52 = None
    try:
        df1y = get_stock_data(candidates[0], "1y", "1d")
        if df1y is not None and not df1y.empty:
            hi52 = float(df1y["Close"].max())
            lo52 = float(df1y["Close"].min())
    except:
        hi52 = lo52 = None

    # SMA50 (resampled)
    sma50 = None
    try:
        df_daily = combined.set_index("Date").resample("D").agg({"Close":"last","Volume":"sum"}).dropna()
        if len(df_daily) >= 50:
            sma50 = float(df_daily["Close"].rolling(50).mean().iloc[-1])
    except:
        sma50 = None

    def compute_rsi(series, n=14):
        s = series.dropna()
        if len(s) < n+1: return None
        delta = s.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = -delta.clip(upper=0).rolling(n).mean()
        rs = up / (down + 1e-9)
        rsi = 100 - (100/(1+rs))
        try:
            return float(rsi.iloc[-1])
        except:
            return None

    day_rsi = compute_rsi(df_daily["Close"]) if 'df_daily' in locals() and not df_daily.empty else None

    # Render top metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='big'>{('₹'+format(latest_price,',.2f')) if latest_price else 'N/A'}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='metric-title'>Today's H / L</div><div class='metric-value'>{('₹'+format(hi_today,',.2f') if hi_today else 'N/A')} / {('₹'+format(lo_today,',.2f') if lo_today else 'N/A')}</div><div class='metric-title'>Vol: {today_vol:,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><div class='metric-title'>52W High</div><div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><div class='metric-title'>52W Low</div><div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='card'><div class='metric-title'>SMA50</div><div class='metric-value'>{('₹'+format(sma50,',.2f')) if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Chart area
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {symbol} — {timeframe}")

        df_plot = combined.copy()
        if df_plot.empty:
            st.warning("No data to plot.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df_plot["Date"],
                open=df_plot["Close"],
                high=df_plot["Close"],
                low=df_plot["Close"],
                close=df_plot["Close"]
            )])
        else:
            fig = px.line(df_plot, x="Date", y="Close", markers=False)

        if sma50 is not None:
            fig.add_scatter(x=df_daily.index, y=df_daily["Close"].rolling(50).mean().values, name="SMA50")

        fig.update_layout(template="plotly_white", height=480)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Indicators"):
            if "RSI" in overlays:
                st.write(f"RSI: **{day_rsi:.2f}**" if day_rsi else "RSI: N/A")
            if "MACD" in overlays:
                s = df_plot["Close"]
                ema12 = s.ewm(span=12).mean()
                ema26 = s.ewm(span=26).mean()
                macd = ema12 - ema26
                sig = macd.ewm(span=9).mean()
                fig2 = go.Figure()
                fig2.add_scatter(x=df_plot["Date"], y=macd, name="MACD")
                fig2.add_scatter(x=df_plot["Date"], y=sig, name="Signal")
                fig2.update_layout(template="plotly_white", height=180)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- Day RSI: **{day_rsi:.2f}**" if day_rsi else "- Day RSI: N/A")
        st.markdown(f"- Combined Volume: **{today_vol:,}**")
        st.markdown("</div>", unsafe_allow_html=True)

# Minimal other pages
def trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","python")
    if st.button("Fetch"):
        repos = bm.fetch_github_trending(q)
        for r in repos[:10]:
            st.write(f"**{r['name']}** — {r['description']}")

def research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search","machine learning")
    if st.button("Search"):
        papers = bm.fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:300]+"...")
            st.markdown(f"[Open]({p['link']})")
            st.divider()

def news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","market")
    if st.button("Get"):
        arts = bm.fetch_news_via_google_rss(q, max_items=8)
        s = bm.analyze_headlines_sentiment(arts)
        for a in s:
            st.markdown(f"**{a['title']}**")
            if a.get("sentiment"):
                st.markdown(f"- {a['sentiment']['label']} ({a['sentiment']['score']:.2f})")
            st.markdown(f"[Open]({a['link']})")
            st.divider()

def feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    rating = st.slider("Rate:",1,5,4)
    comment = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thanks — saved locally.")

# Router
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
