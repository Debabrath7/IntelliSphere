# ==============================================================
# frontend.py — polished Streamlit UI
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

import backend_modules as bm
from backend_modules import (
    clean_ticker,
    get_stock_data,
    get_intraday_data,
    get_today_high_low,
    fetch_github_trending,
    fetch_arxiv_papers,
    fetch_news_via_google_rss,
    analyze_headlines_sentiment
)

# ---------------------------------------------------------
# Page config + Sidebar
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | Exam Ready",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("""
        <div style='display:flex;align-items:center;gap:10px'>
            <div style='background:linear-gradient(90deg,#7b61ff,#00d2ff);
            width:38px;height:38px;border-radius:8px;color:white;
            display:flex;align-items:center;justify-content:center;font-weight:800'>
            IS</div>
            <div>
                <div style='font-weight:700'>IntelliSphere</div>
                <div style='font-size:12px;color:#6b7280'>Super-Fancy UI</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["Home","Stocks","Trends","Research","News","Feedback"], index=1)

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
# Custom card styling
# ---------------------------------------------------------
st.markdown("""
<style>
.card{
    background:white;padding:14px;border-radius:12px;
    box-shadow:0 8px 30px rgba(0,0,0,0.05);margin-bottom:12px;
}
.metric-title{font-size:12px;color:#6b7280}
.metric-value{font-size:20px;font-weight:700}
.big{font-size:28px;font-weight:800}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Home
# ---------------------------------------------------------
def home():
    col1,col2 = st.columns([3,1])
    with col1:
        st.markdown("<div class='card'><h2>Welcome — IntelliSphere (Exam Ready UI)</h2>"
                    "A polished multi-exchange dashboard with intraday charts, indicators & trends."
                    "</div>", unsafe_allow_html=True)
    with col2:
        anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
        if anim:
            st_lottie(anim, height=160)

# ---------------------------------------------------------
# Stocks Page (FULLY FIXED VERSION)
# ---------------------------------------------------------
def stocks():

    st.markdown("<div class='card'><h2>Stocks — Unified NSE + BSE</h2></div>", unsafe_allow_html=True)

    # Inputs
    c1,c2,c3,c4 = st.columns([3,2,2,2])
    with c1:
        raw = st.text_input("Enter Ticker / BSE Code:", "BDL")
    with c2:
        timeframe = st.selectbox("Timeframe",
                                 ["Intraday (1d,1m)","1d","5d","1mo","3mo","6mo","1y","2y","5y"])
    with c3:
        chart_mode = st.selectbox("Chart", ["Candlestick","Line"])
    with c4:
        overlays = st.multiselect("Overlays", ["MACD","RSI","SMA50"], default=["SMA50"])

    btn = st.button("Load Data")
    if not btn:
        return

    # CLEAN TICKER
    base = clean_ticker(raw.strip().upper())

    # Generate possible tickers
    if base.isdigit():
        candidates = [f"{base}.BO", base]
    else:
        candidates = [base, base+".NS", base+".BO"]

    # Determine period
    if timeframe.startswith("Intraday"):
        period, interval, use_intraday = "1d", "1m", True
    else:
        period, interval, use_intraday = timeframe, "1d", False

    # ---------------------------------------------------------
    # FETCH DATA FROM ALL EXCHANGES
    # ---------------------------------------------------------
    dfs = []

    for t in candidates:
        try:
            df = get_intraday_data(t, interval="1m", period="1d") if use_intraday \
                 else get_stock_data(t, period=period, interval=interval)

            if df is not None and not df.empty:
                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"])
                for col in ["Open","High","Low","Close","Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan
                dfs.append((t, df))
        except:
            pass

    if not dfs:
        st.error("No data available for ticker.")
        return

    # =========================================================
    # ⭐ FINAL FIX: SAFE MERGING USING CONCAT (NO MERGE/JOIN) ⭐
    # =========================================================

    base_ticker, base_df = dfs[0]
    merged = base_df.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
    merged.columns = ["Open_0","High_0","Low_0","Close_0","Volume_0"]

    for i, (tk, df2) in enumerate(dfs[1:], start=1):
        temp = df2.set_index("Date")[["Open","High","Low","Close","Volume"]]
        temp = temp.add_suffix(f"_{i}")  # avoids ALL collisions
        merged = pd.concat([merged, temp], axis=1, join="outer")

    merged = merged.sort_index().fillna(method="ffill")

    # Combine volumes
    vol_cols = [c for c in merged.columns if c.startswith("Volume_")]
    close_cols = [c for c in merged.columns if c.startswith("Close_")]
    merged["TotalVolume"] = merged[vol_cols].sum(axis=1)

    # Compute VWAP Close
    numerator = np.zeros(len(merged))
    for vcol, ccol in zip(vol_cols, close_cols):
        numerator += merged[vcol].values * merged[ccol].values

    merged["VWClose"] = numerator / np.where(merged["TotalVolume"]>0, merged["TotalVolume"], np.nan)
    merged["Close"] = merged["VWClose"].fillna(method="ffill")
    merged["Volume"] = merged["TotalVolume"].fillna(0)

    combined = merged.reset_index()[["Date","Close","Volume"]]
    combined = combined.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # METRICS: PRICE, 52W, SMA50, RSI, MFI
    # ---------------------------------------------------------
    latest_price = float(combined["Close"].iloc[-1])
    combined_vol_today = int(combined["Volume"].iloc[-1])

    # 52W
    try:
        df_1y = get_stock_data(candidates[0], period="1y", interval="1d")
        hi52 = float(df_1y["Close"].max())
        lo52 = float(df_1y["Close"].min())
    except:
        hi52 = lo52 = None

    # SMA50
    try:
        daily = combined.set_index("Date").resample("D").agg({"Close": "last", "Volume": "sum"}).dropna()
        sma50 = daily["Close"].rolling(50).mean().iloc[-1] if len(daily)>=50 else None
    except:
        sma50 = None

    # RSI
    def compute_rsi(series, n=14):
        series = series.dropna()
        if len(series) < n+1: return None
        delta = series.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = -delta.clip(upper=0).rolling(n).mean()
        rs = up/(down+1e-9)
        return float((100 - 100/(1+rs)).iloc[-1])

    day_rsi = compute_rsi(daily["Close"]) if 'daily' in locals() else None

    # MFI (approx)
    def compute_mfi(df, window=14):
        if len(df) < window + 1: return None
        df = df.copy()
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        tp = (df["High"]+df["Low"]+df["Close"])/3
        mf = tp * df["Volume"]
        pos=[];neg=[]
        for i in range(1,len(tp)):
            if tp.iloc[i]>tp.iloc[i-1]: pos.append(mf.iloc[i])
            else: neg.append(mf.iloc[i])
        pos = pd.Series(pos).rolling(window).sum()
        neg = pd.Series(neg).rolling(window).sum()
        mfi = 100 - (100/(1+(pos/(neg+1e-9))))
        return float(mfi.iloc[-1])

    day_mfi = compute_mfi(combined)

    # ---------------------------------------------------------
    # METRIC CARDS
    # ---------------------------------------------------------
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.markdown(f"<div class='card'><div class='metric-title'>Current Price</div>"
                    f"<div class='big'>₹{latest_price:,.2f}</div>"
                    f"<div class='metric-title'>Unified NSE+BSE</div></div>", unsafe_allow_html=True)

    with col2:
        hi_today = combined["Close"].max()
        lo_today = combined["Close"].min()
        st.markdown(f"<div class='card'><div class='metric-title'>Today's High / Low</div>"
                    f"<div class='metric-value'>₹{hi_today:,.2f} / ₹{lo_today:,.2f}</div>"
                    f"<div class='metric-title'>Volume: {combined_vol_today:,}</div></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='card'><div class='metric-title'>52W High</div>"
                    f"<div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<div class='card'><div class='metric-title'>52W Low</div>"
                    f"<div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    with col5:
        st.markdown(f"<div class='card'><div class='metric-title'>SMA50</div>"
                    f"<div class='metric-value'>{('₹'+format(sma50,',.2f')) if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------
    left,right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {raw.upper()} — {timeframe}")

        if chart_mode == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=combined["Date"],
                open=combined["Close"],
                high=combined["Close"],
                low=combined["Close"],
                close=combined["Close"]
            )])
            if sma50:
                fig.add_scatter(x=daily.index, y=daily["Close"].rolling(50).mean(),
                                name="SMA50")
        else:
            fig = px.line(combined, x="Date", y="Close", markers=False)
            if sma50:
                fig.add_scatter(x=daily.index, y=daily["Close"].rolling(50).mean(),
                                name="SMA50")

        fig.update_layout(template="plotly_white", height=480)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Indicators"):
            if "RSI" in overlays:
                st.write(f"RSI: **{day_rsi:.2f}**" if day_rsi else "RSI: N/A")

            if "MACD" in overlays:
                s = combined["Close"]
                ema12 = s.ewm(span=12).mean()
                ema26 = s.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                fig_i = go.Figure()
                fig_i.add_scatter(x=combined["Date"], y=macd, name="MACD")
                fig_i.add_scatter(x=combined["Date"], y=signal, name="Signal")
                fig_i.update_layout(template="plotly_white", height=180)
                st.plotly_chart(fig_i, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- Day RSI: **{day_rsi:.2f}**" if day_rsi else "- Day RSI: N/A")
        st.markdown(f"- Day MFI: **{day_mfi:.2f}**" if day_mfi else "- Day MFI: N/A")
        st.markdown(f"- Combined Volume: **{combined_vol_today:,}**")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Trends
# ---------------------------------------------------------
def trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic:", "python")
    if st.button("Fetch"):
        repos = fetch_github_trending(q)
        for r in repos[:10]:
            st.markdown(f"- **{r['name']}** — {r['description']}")

# ---------------------------------------------------------
# Research
# ---------------------------------------------------------
def research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search papers:", "machine learning")
    if st.button("Search"):
        papers = fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:400]+"...")
            st.markdown(f"[Read]({p['link']})")

# ---------------------------------------------------------
# News
# ---------------------------------------------------------
def news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic:", "Indian stock market")
    if st.button("Get"):
        arts = fetch_news_via_google_rss(q, max_items=8)
        s = analyze_headlines_sentiment(arts)
        for a in s:
            st.markdown(f"**{a['title']}**")
            if a.get("sentiment"):
                st.markdown(f"- {a['sentiment']['label']} ({a['sentiment']['score']:.2f})")
            st.markdown(f"[Open]({a['link']})")
            st.divider()

# ---------------------------------------------------------
# Feedback
# ---------------------------------------------------------
def feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    rate = st.slider("Rate:",1,5,4)
    com = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thank you!")

# ---------------------------------------------------------
# Router
# ---------------------------------------------------------
def render_dashboard():
    if page=="Home": home()
    elif page=="Stocks": stocks()
    elif page=="Trends": trends()
    elif page=="Research": research()
    elif page=="News": news()
    elif page=="Feedback": feedback()

if __name__ == "__main__":
    render_dashboard()
