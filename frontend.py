# ==============================================================
# frontend.py — polished Streamlit UI (Super Fancy: Lottie + cards)
# Author: assistant (adapted for Debabrath)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from streamlit_lottie import st_lottie
import json
import requests
import math

# backend
import backend_modules as bm
from backend_modules import (
    clean_ticker,
    get_stock_data,
    get_intraday_data,
    get_today_high_low,
    stock_summary,
    fetch_github_trending,
    fetch_arxiv_papers,
    fetch_news_via_google_rss,
    analyze_headlines_sentiment
)

# -------------------------
# Page config + sidebar
# -------------------------
st.set_page_config(page_title="IntelliSphere | Exam Ready", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# Sidebar navigation
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px'>"
                "<div style='background:linear-gradient(90deg,#7b61ff,#00d2ff);width:38px;height:38px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:center;font-weight:800'>IS</div>"
                "<div><div style='font-weight:700'>IntelliSphere</div><div style='font-size:12px;color:#6b7280'>Super-Fancy UI</div></div></div>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Home","Stocks","Trends","Research","News","Feedback"], index=1)

# utility: load lottie from URL
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

# small style tweak
st.markdown("""
    <style>
    .card{background:#ffffff;border-radius:12px;padding:14px;box-shadow:0 10px 30px rgba(15,23,42,0.06);margin-bottom:14px}
    .metric-title{color:#6b7280;font-size:12px}
    .metric-value{font-size:20px;font-weight:700;color:#0b1220}
    .big{font-size:28px;font-weight:800}
    </style>
""", unsafe_allow_html=True)

# ---------- Home ----------
def home():
    col1,col2 = st.columns([3,1])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Welcome — IntelliSphere (Exam UI)")
        st.write("Showcase-ready, polished dashboard with combined NSE+BSE data, intraday candlesticks, indicators and multiple time-range analytics.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
        if lottie:
            st_lottie(lottie, height=160)

# ---------- Stocks Page ----------
def stocks():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Stocks — Unified NSE + BSE")
    st.markdown("</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns([3,2,1.5,1.5])
    with c1:
        raw = st.text_input("Enter Ticker (symbol or BSE id):", value="BDL")
    with c2:
        timeframe = st.selectbox("Timeframe", ["Intraday (1d,1m)","1d","5d","1mo","3mo","6mo","1y","2y","5y"], index=0)
    with c3:
        chart_mode = st.selectbox("Chart", ["Candlestick","Line"], index=0)
    with c4:
        overlays = st.multiselect("Overlays", ["MACD","RSI","SMA50"], default=["SMA50"])

    btn = st.button("Load")

    if not btn and "last_loaded" in st.session_state and st.session_state["last_loaded"].get("query")==raw and st.session_state["last_loaded"].get("timeframe")==timeframe:
        combined = st.session_state["last_loaded"]["combined"]
        candidates = st.session_state["last_loaded"]["candidates"]
        info_dicts = st.session_state["last_loaded"]["info"]
    else:
        # Build candidate tickers (try symbol, symbol.NS, symbol.BO; if numeric assume BSE id)
        q = str(raw).strip()
        base = clean_ticker(q)
        candidates = []
        if base.isdigit():
            candidates = [f"{base}.BO", base]
        else:
            candidates = [base, base + ".NS", base + ".BO"]
        # determine period/interval
        use_intraday = timeframe.startswith("Intraday")
        interval = "1m" if use_intraday else "1d"
        period = "1d" if use_intraday else timeframe
        # fetch from candidates in parallel
        dfs = []
        info_dicts = {}
        for t in candidates:
            try:
                df_t = get_intraday_data(t, interval=interval, period=period) if use_intraday else get_stock_data(t, period=period, interval=interval)
                if df_t is not None and not df_t.empty:
                    df_t = df_t.copy()
                    if "Date" not in df_t.columns and "Datetime" in df_t.columns:
                        df_t = df_t.rename(columns={"Datetime":"Date"})
                    df_t["Date"] = pd.to_datetime(df_t["Date"])
                    for c in ["Open","High","Low","Close","Volume"]:
                        if c not in df_t.columns:
                            df_t[c] = np.nan
                    dfs.append(df_t[["Date","Open","High","Low","Close","Volume"]])
                    # get safe info
                    info_dicts[t] = {}
                    try:
                        info_dicts[t] = bm.safe_get_info(t) if hasattr(bm, "safe_get_info") else {}
                    except Exception:
                        info_dicts[t] = {}
            except Exception:
                continue
        if not dfs:
            st.error("No data found. Try symbol like TCS or numeric BSE id.")
            return
        # combine (VW close + combined volume)
        # simple combine method: align on Date index, sum volumes, compute VW close
        merged = dfs[0].set_index("Date").copy()
        merged.columns = [f"Open_0","High_0","Low_0","Close_0","Volume_0"]
        for i,d in enumerate(dfs[1:], start=1):
            tmp = d.set_index("Date")
            tmp.columns = [f"Open_{i}","High_{i}","Low_{i}","Close_{i}","Volume_{i}"]
            merged = merged.join(tmp, how="outer")
        merged = merged.sort_index().fillna(method="ffill").fillna(0)
        close_cols = [c for c in merged.columns if c.startswith("Close_")]
        vol_cols = [c for c in merged.columns if c.startswith("Volume_")]
        merged["TotalVolume"] = merged[vol_cols].sum(axis=1)
        # weighted close
        numerator = np.zeros(len(merged))
        for i,cc in enumerate(close_cols):
            numerator += (merged[cc].fillna(0).values * merged[vol_cols[i]].values)
        with np.errstate(divide='ignore', invalid='ignore'):
            vw = np.where(merged["TotalVolume"].values>0, numerator / (merged["TotalVolume"].values), np.nan)
        merged["VWClose"] = pd.Series(vw, index=merged.index).fillna(method="ffill")
        combined = merged.reset_index()[["Date","VWClose","TotalVolume"]].rename(columns={"VWClose":"Close","TotalVolume":"Volume"})

        # store
        st.session_state["last_loaded"] = {"query": raw, "timeframe": timeframe, "combined": combined, "candidates": candidates, "info": info_dicts}

    # compute metrics
    combined = combined.sort_values("Date").reset_index(drop=True)
    latest = combined.iloc[-1]
    latest_price = float(latest["Close"]) if not np.isnan(latest["Close"]) else None
    combined_vol_today = int(latest["Volume"]) if not np.isnan(latest["Volume"]) else 0

    # 52w high/low: try to fetch 1y daily for first candidate if possible
    hi52 = "N/A"; lo52="N/A"
    try:
        df_1y = get_stock_data(candidates[0], period="1y", interval="1d")
        if df_1y is not None and not df_1y.empty:
            hi52 = float(df_1y["Close"].max())
            lo52 = float(df_1y["Close"].min())
    except Exception:
        pass

    # SMA50: compute on daily series (resample from combined if intraday)
    try:
        daily = combined.copy()
        if timeframe.startswith("Intraday"):
            # resample: daily close
            daily = daily.set_index("Date").resample("D").agg({"Close":"last","Volume":"sum"}).dropna().reset_index()
        if len(daily) >= 50:
            sma50 = float(daily["Close"].rolling(50).mean().iloc[-1])
        else:
            sma50 = None
    except Exception:
        sma50 = None

    # Day RSI & MFI (use daily series)
    def compute_rsi(series, period=14):
        series = series.dropna()
        if len(series) < period+1:
            return None
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / (ma_down + 1e-9)
        rsi = 100 - (100/(1+rs))
        return float(rsi.iloc[-1])
    def compute_mfi(df, period=14):
        if df is None or len(df) < period+1:
            return None
        typical = (df.get("High", df["Close"]) + df.get("Low", df["Close"]) + df["Close"]) / 3
        money = typical * df["Volume"].fillna(0)
        pos = []
        neg = []
        for i in range(1,len(typical)):
            if typical.iat[i] > typical.iat[i-1]:
                pos.append(money.iat[i]); neg.append(0)
            else:
                pos.append(0); neg.append(money.iat[i])
        pos = pd.Series(pos).rolling(period).sum()
        neg = pd.Series(neg).rolling(period).sum()
        mfr = pos/(neg + 1e-9)
        mfi = 100 - (100/(1+mfr))
        return float(mfi.iloc[-1]) if not mfi.empty else None

    try:
        day_rsi = compute_rsi(daily["Close"]) if 'daily' in locals() else None
    except Exception:
        day_rsi = None
    try:
        # daily needs High/Low — attempt to create proxys if not present
        daily_with_hl = daily.copy()
        if "High" not in daily_with_hl.columns:
            daily_with_hl["High"] = daily_with_hl["Close"]
        if "Low" not in daily_with_hl.columns:
            daily_with_hl["Low"] = daily_with_hl["Close"]
        day_mfi = compute_mfi(daily_with_hl)
    except Exception:
        day_mfi = None

    # Beta and identifiers using safe info (best-effort)
    beta_val = None
    id_display = []
    info_dicts = st.session_state["last_loaded"]["info"] if "last_loaded" in st.session_state else {}
    for k,inf in info_dicts.items():
        if not inf:
            continue
        if beta_val is None:
            beta_val = inf.get("beta") or inf.get("beta_1y") or inf.get("beta50")
        id_display.append({"ticker": k, "shortName": inf.get("shortName"), "isin": inf.get("isin"), "exchange": inf.get("exchange")})

    # Ranges: compute percent change relative to past
    def pct_range(now, past):
        try:
            return (now - past) / (past + 1e-9) * 100
        except Exception:
            return None

    # get long series for ranges (5y)
    df_long = get_stock_data(candidates[0], period="5y", interval="1d") or pd.DataFrame()
    ranges = {}
    mapping = {"1 day":1, "1 week":5, "1 month":21, "3 month":63, "6 month":126, "1 year":252, "2 year":504, "5 year":1260}
    if not df_long.empty:
        df_long = df_long.sort_values("Date").reset_index(drop=True)
        for name,days in mapping.items():
            if len(df_long) > days:
                past = df_long["Close"].iloc[-1-days]
                now = df_long["Close"].iloc[-1]
                ranges[name] = pct_range(now, past)
            else:
                ranges[name] = None
    else:
        for name in mapping.keys():
            ranges[name] = None

    # ---- Render top metrics ----
    col1,col2,col3,col4,col5 = st.columns([2,1.2,1.2,1.2,1.2])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280'>Current Price</div><div class='big'>₹{latest_price:,.2f}</div><div style='font-size:12px;color:#6b7280'>Combined NSE+BSE</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280'>Today's H / L</div><div class='metric-value'>₹{combined['Close'].max():,.2f} / ₹{combined['Close'].min():,.2f}</div><div style='font-size:12px;color:#6b7280'>Vol: {combined_vol_today:,}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280'>52W High</div><div class='metric-value'>{('₹{:, .2f}'.format(hi52) if isinstance(hi52,(int,float)) else 'N/A')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280'>52W Low</div><div class='metric-value'>{('₹{:, .2f}'.format(lo52) if isinstance(lo52,(int,float)) else 'N/A')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col5:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280'>SMA50</div><div class='metric-value'>{('₹{:, .2f}'.format(sma50) if sma50 else 'N/A')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart + indicators
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {raw.upper()} — {timeframe}")
        df_plot = combined.copy()
        if chart_mode == "Candlestick":
            # if no OHLC available we show Close as candle body (still visual)
            fig = go.Figure(data=[go.Candlestick(x=df_plot["Date"], open=df_plot["Close"], high=df_plot["Close"], low=df_plot["Close"], close=df_plot["Close"], name="Price")])
            if "SMA50" in overlays and sma50:
                df_plot["SMA50"] = df_plot["Close"].rolling(50).mean()
                fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["SMA50"], name="SMA50"))
        else:
            fig = px.line(df_plot, x="Date", y="Close", title=f"{raw.upper()} Price", markers=False)
            if "SMA50" in overlays and sma50:
                df_plot["SMA50"] = df_plot["Close"].rolling(50).mean()
                fig.add_scatter(x=df_plot["Date"], y=df_plot["SMA50"], mode="lines", name="SMA50")
        fig.update_layout(template="plotly_white", height=480, margin=dict(l=8,r=8,t=40,b=8))
        st.plotly_chart(fig, use_container_width=True)
        # expanders for indicators
        with st.expander("Indicators"):
            if "RSI" in overlays:
                st.write(f"RSI (current): **{day_rsi:.2f}**" if day_rsi else "RSI: N/A")
            if "MACD" in overlays:
                # compute simple MACD
                series = df_plot["Close"]
                ema12 = series.ewm(span=12, adjust=False).mean()
                ema26 = series.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df_plot["Date"], y=macd, name="MACD"))
                fig_macd.add_trace(go.Scatter(x=df_plot["Date"], y=macd_signal, name="Signal"))
                fig_macd.update_layout(template="plotly_white", height=160)
                st.plotly_chart(fig_macd, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Quick Stats")
        st.markdown(f"- Day RSI: **{day_rsi:.2f}**" if day_rsi else "- Day RSI: N/A")
        st.markdown(f"- Day MFI: **{day_mfi:.2f}**" if day_mfi else "- Day MFI: N/A")
        st.markdown(f"- Beta (1Y): **{beta_val:.2f}**" if beta_val else "- Beta (1Y): N/A")
        st.markdown(f"- Combined Volume (today): **{combined_vol_today:,}**")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Ranges")
        for k,v in ranges.items():
            st.markdown(f"- **{k}:** {v:.2f}%" if v is not None else f"- **{k}:** N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    # bottom actions
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    b1,b2,b3 = st.columns(3)
    with b1:
        if st.button("Download Combined CSV"):
            tmp = combined.copy()
            tmp["Date"] = tmp["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.download_button("Download CSV", tmp.to_csv(index=False), file_name=f"{raw}_combined.csv", mime="text/csv")
    with b2:
        if st.button("Show Raw Exchange Info"):
            st.json(info_dicts)
    with b3:
        if st.button("Refresh"):
            st.session_state.pop("last_loaded", None)
            st.experimental_rerun()

# ---------- Other pages ----------
def trends():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Trends")
    q = st.text_input("Topic:", value="python")
    if st.button("Fetch"):
        repos = fetch_github_trending(q)
        for r in repos[:10]:
            st.markdown(f"- **{r.get('name')}** — {r.get('description')}")
    st.markdown("</div>", unsafe_allow_html=True)

def research():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Research")
    q = st.text_input("Search papers:", value="machine learning")
    if st.button("Search"):
        papers = fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p.get("title"))
            st.write(p.get("summary")[:400] + "...")
            st.markdown(f"[Read more]({p.get('link')})")
    st.markdown("</div>", unsafe_allow_html=True)

def news():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## News & Sentiment")
    q = st.text_input("Topic/Company:", value="Indian Stock Market")
    n = st.slider("Articles:", 3, 15, 6)
    if st.button("Get News"):
        arts = fetch_news_via_google_rss(q, max_items=n)
        s = analyze_headlines_sentiment(arts)
        for a in s:
            st.markdown(f"**{a['title']}**")
            if a.get("sentiment"):
                lab = a["sentiment"]["label"]; scr = a["sentiment"]["score"]
                st.markdown(f"- Sentiment: **{lab}** ({scr:.2f})")
            if a.get("link"): st.markdown(f"[Read more]({a['link']})")
            st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

def feedback():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Feedback")
    name = st.text_input("Name")
    rating = st.slider("Rate (1-5):",1,5,4)
    comment = st.text_area("Comments")
    if st.button("Submit"):
        st.success("Thanks — feedback recorded (local).")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Main router ----------
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
