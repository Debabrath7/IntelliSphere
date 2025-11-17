# frontend.py
# IntelliSphere - Frontend (dark mode, intraday auto-switch for 1d)
# Author: adapted for Debabrath
# Replace your existing frontend.py with this file.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Styles (dark friendly + header tweak)
st.markdown("""
<style>
.stApp { background: #0f1720; color: #e6eef6; font-family: Inter, Roboto, Arial, sans-serif; }
h1,h2,h3{color:#e6eef6}
.topbar{background:transparent;padding:12px 0;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;gap:14px;align-items:center}
.logo{width:44px;height:44px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:linear-gradient(90deg,#00b894,#00a8ff);color:#fff;font-weight:700;box-shadow:0 6px 18px rgba(0,0,0,0.06);font-size:18px}
.nav{display:flex;gap:14px;align-items:center}
.card{background:#0b1220;border:1px solid rgba(255,255,255,0.03);border-radius:10px;padding:18px;box-shadow:0 6px 20px rgba(17,24,39,0.02)}
.metric-title{color:#9aa3ad;font-size:13px;margin-bottom:6px}
.metric-value{font-weight:700;font-size:20px;color:#e6eef6}
.metric-sub{color:#9aa3ad;font-size:13px;margin-top:6px}
.divider{height:1px;background:#0b1220;margin:14px 0;border-radius:2px}
.muted{color:#9aa3ad;font-size:13px}
.small-muted{color:#9aa3ad;font-size:12px}
</style>
""", unsafe_allow_html=True)

# Header
def header_bar():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.markdown('<div class="brand"><div class="logo">IS</div><div><div style="font-weight:700;font-size:20px;color:#e6eef6">IntelliSphere</div><div class="muted" style="font-size:12px;margin-top:2px">AI-Powered Insights</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Helpers
def humanize_number(num):
    try:
        num = float(num)
    except Exception:
        return "N/A"
    n = abs(num)
    if n >= 1e12: return f"{num/1e12:.2f} Tn"
    if n >= 1e7: return f"{num/1e7:.2f} Cr"
    if n >= 1e5: return f"{num/1e5:.2f} L"
    if n >= 1e3: return f"{num/1e3:.2f} K"
    return f"{num:.2f}"

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

    # AUTO-SWITCH: if timeframe == "1d", request intraday 1m
    if timeframe == "1d":
        period = "1d"
        interval = "1m"
        use_intraday = True
    else:
        period = timeframe
        interval = "1d"
        use_intraday = False

    # Fetch data using backend
    df = None
    try:
        if use_intraday:
            df = get_intraday_data(sym, interval=interval, period=period)
        else:
            df = get_stock_data(sym, period=period, interval=interval)
    except Exception:
        df = None

    # If none, try demo fallback via backend again (backend handles demo)
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
        st.error("No data available for provided symbol. Try a demo symbol (e.g., RELIANCE, TCS).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = ensure_date_col(df)
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            st.error("Unexpected data format from backend.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    # Ensure OHLCV columns exist
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values("Date").reset_index(drop=True)

    # Scalars
    try:
        latest_price = float(df["Close"].dropna().iloc[-1])
    except Exception:
        latest_price = None
    try:
        today_vol = int(df["Volume"].dropna().iloc[-1])
    except Exception:
        today_vol = 0

    # 52-week: try backend daily 1y if intraday
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

    # SMA50
    try:
        if use_intraday:
            sma50_val = float(df["Close"].dropna().rolling(50).mean().iloc[-1]) if len(df["Close"].dropna())>=50 else None
        else:
            df_daily = df.set_index("Date").resample("D").last().dropna()
            sma50_val = float(df_daily["Close"].rolling(50).mean().iloc[-1]) if len(df_daily)>=50 else None
    except Exception:
        sma50_val = None

    # RSI
    try:
        if use_intraday:
            rsi_series = compute_rsi(df["Close"].dropna(), n=14)
            day_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None
        else:
            series = df.set_index("Date").resample("D").last().dropna()["Close"]
            rsi_series = compute_rsi(series, n=14)
            day_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None
    except Exception:
        day_rsi = None

    # Top metric cards
    ct1,ct2,ct3,ct4 = st.columns(4)
    ct1.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='metric-value'>{('₹'+format(latest_price,',.2f')) if latest_price else 'N/A'}</div></div>", unsafe_allow_html=True)
    ct2.markdown(f"<div class='card'><div class='metric-title'>Today's H / L</div><div class='metric-value'>{('₹'+format(float(df['High'].max()),',.2f')) if not df['High'].isna().all() else 'N/A'} / {('₹'+format(float(df['Low'].min()),',.2f')) if not df['Low'].isna().all() else 'N/A'}</div></div>", unsafe_allow_html=True)
    ct3.markdown(f"<div class='card'><div class='metric-title'>52W High</div><div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    ct4.markdown(f"<div class='card'><div class='metric-title'>52W Low</div><div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Chart + right panel
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {sym}", unsafe_allow_html=True)

        df_plot = df.copy()
        if df_plot.empty:
            st.warning("No data to plot.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Plot
        try:
            if chart_type == "Candlestick":
                fig = go.Figure(data=[go.Candlestick(x=df_plot["Date"], open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], name="Price")])
            else:
                fig = px.line(df_plot, x="Date", y="Close", title=f"{sym} Price", markers=False)

            # Add SMA (series or single value)
            if sma50_val is not None:
                if use_intraday:
                    sma_series = df_plot["Close"].rolling(50).mean()
                    fig.add_scatter(x=df_plot["Date"], y=sma_series, mode="lines", name="SMA50", line=dict(width=1))
                else:
                    df_daily = df_plot.set_index("Date").resample("D").last().dropna()
                    if len(df_daily) >= 50:
                        fig.add_scatter(x=df_daily.index, y=df_daily["Close"].rolling(50).mean(), mode="lines", name="SMA50", line=dict(width=1))

            fig.update_layout(template="plotly_dark", height=480, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Could not render chart: " + str(e))

        # Indicators expander
        with st.expander("Indicators"):
            if "RSI" in overlays:
                if day_rsi is not None:
                    st.write(f"RSI: **{day_rsi:.2f}**")
                else:
                    st.write("RSI: N/A")
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

# Minimal other pages
def render_trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    lang = st.text_input("Language or topic:", "python")
    if st.button("Fetch Trending Repos"):
        with st.spinner("Fetching trending repos..."):
            repos = fetch_github_trending(lang)
        if not repos:
            st.warning("No trending repositories right now.")
        else:
            for r in repos[:10]:
                st.markdown(f"**{r['name']}** — {r.get('description','')}")
                st.caption(f"Stars: {r.get('stars','0')}")
                st.divider()

def render_research():
    st.markdown("<div class='card'><h2>Research & Education</h2></div>", unsafe_allow_html=True)
    topic = st.text_input("Search papers on:", "machine learning")
    if st.button("Fetch Papers"):
        with st.spinner("Fetching papers..."):
            papers = fetch_arxiv_papers(topic, max_results=6)
        if not papers:
            st.warning("No papers found.")
        else:
            for p in papers:
                st.subheader(p["title"])
                st.write(p["summary"][:600] + "...")
                st.markdown(f"[Read more]({p['link']})")
                st.divider()

def render_news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","market")
    if st.button("Get"):
        arts = fetch_news_via_google_rss(q, max_items=8)
        sents = analyze_headlines_sentiment(arts)
        for a in sents:
            col1,col2 = st.columns([4,1])
            with col1:
                st.markdown(f"**{a['title']}**")
                if a.get("link"):
                    st.markdown(f"[Read more]({a['link']})")
                st.caption(a['published'].strftime("%b %d, %Y %H:%M"))
            with col2:
                lab = a['sentiment']['label'] if a.get("sentiment") else "NEUTRAL"
                sc = a['sentiment']['score'] if a.get("sentiment") else 0.0
                badge = "🟢" if "POS" in (lab or "") else "🔴" if "NEG" in (lab or "") else "⚪"
                st.markdown(f"**{badge} {lab}**\n\n{sc:.2f}")
            st.divider()

def render_feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    rating = st.slider("Rate IntelliSphere (1-5):", 1, 5, 4)
    comments = st.text_area("Comments")
    if st.button("Submit Feedback"):
        row = {"Name": name, "Rating": rating, "Comments": comments, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            import os
            if not os.path.exists("feedback.csv"):
                pd.DataFrame([row]).to_csv("feedback.csv", index=False)
            else:
                df_fb = pd.read_csv("feedback.csv")
                df_fb = pd.concat([df_fb, pd.DataFrame([row])], ignore_index=True)
                df_fb.to_csv("feedback.csv", index=False)
            st.success("Thanks — feedback submitted!")
        except Exception:
            st.error("Could not save feedback.")
    st.markdown("</div>", unsafe_allow_html=True)

# Router / Main
def render_dashboard():
    header_bar()
    page = st.sidebar.radio("Menu", ["Home","Stocks","Trends","Research","News","Feedback"], index=1)
    st.markdown("<div style='padding:20px'>", unsafe_allow_html=True)
    if page == "Home":
        st.title("Welcome to IntelliSphere")
        st.markdown("Your AI dashboard for stocks, trends, research and news.")
        st.success("All systems operational!")
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
        st.write("Page not found.")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    render_dashboard()
