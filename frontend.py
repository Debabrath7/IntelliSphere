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
import math

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
        r = requests.get(url, timeout=6)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ---------------------------------------------------------
# Custom styling
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
# Stocks Page (robust)
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
    base = clean_ticker(str(raw).strip().upper())

    # Generate candidate tickers
    if base.isdigit():
        candidates = [f"{base}.BO", base]
    else:
        candidates = [base, f"{base}.NS", f"{base}.BO"]

    # Determine period
    if timeframe.startswith("Intraday"):
        period, interval, use_intraday = "1d", "1m", True
    else:
        period, interval, use_intraday = timeframe, "1d", False

    # Fetch data from candidates
    dfs = []
    info_dicts = {}
    for t in candidates:
        try:
            df = get_intraday_data(t, interval="1m", period="1d") if use_intraday else get_stock_data(t, period=period, interval=interval)
            if df is not None and not df.empty:
                df = df.copy()
                # Normalize date column
                if "Datetime" in df.columns and "Date" not in df.columns:
                    df = df.rename(columns={"Datetime":"Date"})
                if "Date" not in df.columns:
                    df = df.reset_index().rename(columns={"index":"Date"})
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                # Ensure OHLCV
                for col in ["Open","High","Low","Close","Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan
                # Keep only relevant columns and drop empty dates
                df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Date"])
                if not df.empty:
                    dfs.append((t, df))
                    # safe info fetch best-effort
                    try:
                        info_dicts[t] = bm.safe_get_info(t) if hasattr(bm, "safe_get_info") else {}
                    except:
                        info_dicts[t] = {}
        except Exception:
            # ignore single-source failures; continue with other candidates
            continue

    if not dfs:
        st.error("No data available for the provided ticker/ID on NSE or BSE. Try a different input like TCS or 500325.")
        return

    # ---------------------------------------------------------
    # SAFE CONCAT MERGE: build merged DataFrame with suffixed columns
    # ---------------------------------------------------------
    # Start with first dataframe
    base_ticker, base_df = dfs[0]
    merged = base_df.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
    # provide suffix _0
    merged.columns = [f"{col}_0" for col in merged.columns]

    # For each additional data source, suffix and concat by index
    for i, (tk, df2) in enumerate(dfs[1:], start=1):
        temp = df2.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()
        # rename with suffix to avoid collisions
        temp.columns = [f"{col}_{i}" for col in temp.columns]
        # outer concat along columns aligned by index (Date)
        merged = pd.concat([merged, temp], axis=1, join="outer")

    # Ensure merged is DataFrame
    if not isinstance(merged, pd.DataFrame):
        merged = pd.DataFrame(merged)

    # Normalize column names to strings (avoid weird dtypes)
    merged.columns = [str(c) for c in merged.columns]

    # Fill forward to cover missing rows (aligning timestamps)
    merged = merged.sort_index().fillna(method="ffill").fillna(0)

    # ---------------------------------------------------------
    # Identify volume and close columns robustly (case-insensitive)
    # ---------------------------------------------------------
    cols = list(merged.columns)

    vol_cols = [c for c in cols if c.lower().startswith("volume_")]
    if not vol_cols:
        vol_cols = [c for c in cols if "volume" in c.lower()]

    close_cols = [c for c in cols if c.lower().startswith("close_")]
    if not close_cols:
        close_cols = [c for c in cols if "close" in c.lower()]

    # If still empty, try to infer numeric columns as closes (fallback)
    if not close_cols:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(merged[c])]
        # prefer columns with 'vw' or 'vwp' or 'price' in name
        guessed = [c for c in numeric_cols if any(x in c.lower() for x in ["vw", "vwap", "price", "close"])]
        if guessed:
            close_cols = guessed
        elif numeric_cols:
            # take the last numeric column as fallback
            close_cols = [numeric_cols[-1]]

    # If no volume columns, create a zero volume column to avoid division by zero
    if not vol_cols:
        merged["Volume_0_fallback"] = 0
        vol_cols = ["Volume_0_fallback"]

    # Ensure the number of close and volume columns match for pairing
    # We'll pair by position; use min length
    pair_count = min(len(vol_cols), len(close_cols))
    if pair_count == 0:
        # cannot compute VWAP; create Close as first available numeric column if possible
        if close_cols:
            merged["VWClose"] = merged[close_cols[0]]
        else:
            # fallback: use the first numeric column in merged
            numeric_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
            if numeric_cols:
                merged["VWClose"] = merged[numeric_cols[0]]
            else:
                st.error("Unable to determine price columns for this ticker.")
                return
    else:
        # compute numerator using paired columns
        numerator = np.zeros(len(merged), dtype=float)
        total_vol = np.zeros(len(merged), dtype=float)
        for idx in range(pair_count):
            vcol = vol_cols[idx]
            ccol = close_cols[idx]
            # coerce to numeric
            vol_vals = pd.to_numeric(merged[vcol], errors="coerce").fillna(0).values
            close_vals = pd.to_numeric(merged[ccol], errors="coerce").fillna(np.nan).values
            numerator += vol_vals * np.nan_to_num(close_vals, nan=0.0)
            total_vol += vol_vals
        # avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            vwclose = np.where(total_vol > 0, numerator / total_vol, np.nan)
        merged["VWClose"] = pd.Series(vwclose, index=merged.index)

    # final combined Close and Volume columns
    merged["Close"] = merged["VWClose"].fillna(method="ffill")
    # if TotalVolume exists, use it; else sum vol_cols
    if any(c in merged.columns for c in ["TotalVolume"]):
        merged["Volume"] = merged["TotalVolume"].fillna(0)
    else:
        merged["Volume"] = merged[[c for c in vol_cols if c in merged.columns]].sum(axis=1).fillna(0)

    combined = merged.reset_index()[["Date","Close","Volume"]].copy()
    combined = combined.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Compute metrics: latest price, 52W, SMA50, RSI, MFI
    # ---------------------------------------------------------
    try:
        latest_price = float(combined["Close"].iloc[-1])
    except Exception:
        latest_price = None

    try:
        combined_vol_today = int(combined["Volume"].iloc[-1])
    except Exception:
        combined_vol_today = 0

    # 52-week high/low
    hi52 = lo52 = None
    try:
        df_1y = get_stock_data(candidates[0], period="1y", interval="1d")
        if df_1y is not None and not df_1y.empty:
            hi52 = float(df_1y["Close"].max())
            lo52 = float(df_1y["Close"].min())
    except Exception:
        hi52 = lo52 = None

    # SMA50 (resample to daily)
    try:
        daily = combined.set_index("Date").resample("D").agg({"Close":"last","Volume":"sum"}).dropna()
        sma50 = float(daily["Close"].rolling(50).mean().iloc[-1]) if len(daily) >= 50 else None
    except Exception:
        sma50 = None

    # RSI
    def compute_rsi(series, n=14):
        s = series.dropna()
        if len(s) < n+1:
            return None
        delta = s.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = -delta.clip(upper=0).rolling(n).mean()
        rs = up/(down + 1e-9)
        rsi = 100 - (100/(1+rs))
        try:
            return float(rsi.iloc[-1])
        except:
            return None

    day_rsi = compute_rsi(daily["Close"]) if 'daily' in locals() and not daily.empty else None

    # MFI (approx)
    def compute_mfi(df, window=14):
        if df is None or len(df) < window + 1:
            return None
        tmp = df.copy()
        if "High" not in tmp.columns or "Low" not in tmp.columns:
            tmp["High"] = tmp["Close"]
            tmp["Low"] = tmp["Close"]
        tp = (tmp["High"] + tmp["Low"] + tmp["Close"]) / 3
        mf = tp * tmp["Volume"].fillna(0)
        pos = []; neg = []
        for i in range(1, len(tp)):
            if tp.iloc[i] > tp.iloc[i-1]:
                pos.append(mf.iloc[i]); neg.append(0)
            else:
                pos.append(0); neg.append(mf.iloc[i])
        pos = pd.Series(pos).rolling(window).sum()
        neg = pd.Series(neg).rolling(window).sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            mfr = pos / (neg + 1e-9)
        mfi = 100 - (100/(1 + mfr))
        try:
            return float(mfi.iloc[-1])
        except:
            return None

    day_mfi = compute_mfi(combined)

    # ---------------------------------------------------------
    # Render metric cards
    # ---------------------------------------------------------
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>Current Price</div><div class='big'>{'₹'+format(latest_price,',.2f') if latest_price else 'N/A'}</div>")
        st.markdown("<div class='metric-title'>Combined NSE+BSE</div></div>", unsafe_allow_html=True)

    with col2:
        hi_today = combined["Close"].max() if not combined.empty else None
        lo_today = combined["Close"].min() if not combined.empty else None
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>Today's High / Low</div><div class='metric-value'>{('₹'+format(hi_today,',.2f') if hi_today else 'N/A')} / {('₹'+format(lo_today,',.2f') if lo_today else 'N/A')}</div>")
        st.markdown(f"<div class='metric-title'>Volume: {combined_vol_today:,}</div></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>52W High</div><div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>52W Low</div><div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    with col5:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>SMA50</div><div class='metric-value'>{('₹'+format(sma50,',.2f')) if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Chart area
    # ---------------------------------------------------------
    left,right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {raw.upper()} — {timeframe}")

        df_plot = combined.copy()
        if df_plot.empty:
            st.warning("No combined time series to plot.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if chart_mode == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df_plot["Date"],
                open=df_plot["Close"],
                high=df_plot["Close"],
                low=df_plot["Close"],
                close=df_plot["Close"],
                name="Price"
            )])
            if sma50 is not None:
                fig.add_trace(go.Scatter(x=daily.index, y=daily["Close"].rolling(50).mean().values, name="SMA50"))
        else:
            fig = px.line(df_plot, x="Date", y="Close", title="", markers=False)
            if sma50 is not None:
                fig.add_scatter(x=daily.index, y=daily["Close"].rolling(50).mean().values, name="SMA50")

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
                signal = macd.ewm(span=9).mean()
                fig2 = go.Figure()
                fig2.add_scatter(x=df_plot["Date"], y=macd, name="MACD")
                fig2.add_scatter(x=df_plot["Date"], y=signal, name="Signal")
                fig2.update_layout(template="plotly_white", height=180)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- Day RSI: **{day_rsi:.2f}**" if day_rsi else "- Day RSI: N/A")
        st.markdown(f"- Day MFI: **{day_mfi:.2f}**" if day_mfi else "- Day MFI: N/A")
        st.markdown(f"- Combined Volume: **{combined_vol_today:,}**")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Trends / Research / News / Feedback — minimal but functional
# ---------------------------------------------------------
def trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic:", "python")
    if st.button("Fetch"):
        repos = fetch_github_trending(q)
        for r in repos[:10]:
            st.markdown(f"- **{r['name']}** — {r['description']}")

def research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search papers:", "machine learning")
    if st.button("Search"):
        papers = fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:400]+"...")
            st.markdown(f"[Read]({p['link']})")

def news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic:", "market")
    if st.button("Get"):
        arts = fetch_news_via_google_rss(q, max_items=8)
        s = analyze_headlines_sentiment(arts)
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
    comments = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thanks — feedback saved locally.")

# ---------------------------------------------------------
# Router
# ---------------------------------------------------------
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
