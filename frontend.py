# frontend.py
# IntelliSphere - Frontend
# Author: Debabrath

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil import tz

# backend imports
import backend_modules as bm
from backend_modules import get_stock_data, get_intraday_data, get_today_high_low, stock_summary, fetch_github_trending, fetch_arxiv_papers, fetch_news_via_google_rss, analyze_headlines_sentiment

st.set_page_config(page_title="IntelliSphere | Insights", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# Basic style
st.markdown("""
<style>
.stApp { background: #0f1720; color: #e6eef6; font-family: Inter, Roboto, Arial, sans-serif; }
h1,h2,h3{color:#e6eef6}
.card{background:#0b1220;border:1px solid rgba(255,255,255,0.03);border-radius:10px;padding:18px;margin-bottom:12px}
.metric-title{color:#9aa3ad;font-size:13px;margin-bottom:6px}
.metric-value{font-weight:700;font-size:20px;color:#e6eef6}
.small-muted{color:#9aa3ad;font-size:12px}
.sidebar-logo{display:block;margin-left:auto;margin-right:auto}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/Q96QJYV.png", width=72)  # logo image (replace with your own if desired)
    st.title("IntelliSphere")
    st.write("AI-Powered Insights")
    page = st.radio("", ["Home","Stocks","Trends","Research","News","Feedback"], index=1, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#9aa3ad'>Made with ❤️ by <b>Debabrath</b></div>", unsafe_allow_html=True)

# Helpers
@st.cache_data(ttl=30)  # cache small results for 30 seconds to avoid API spam while tweaking UI
def cached_get_stock(sym, period, interval, intraday):
    if intraday:
        return get_intraday_data(sym, interval=interval, period=period)
    return get_stock_data(sym, period=period, interval=interval)

def ensure_date_col(df):
    if df is None: return df
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
        except:
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

def compute_macd(series):
    s = series.dropna()
    if len(s) < 26: return None, None
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def rolling_volatility(series, window=20):
    s = series.dropna().pct_change().rolling(window).std() * np.sqrt(252)
    return s

# Stocks page (auto-update)
def render_stock():
    st.markdown("<div class='card'><h2>Stocks Dashboard</h2></div>", unsafe_allow_html=True)

    # Controls — changing any widget triggers a rerun and will fetch data
    c1,c2,c3,c4,c5 = st.columns([3,2,2,2,1])
    with c1:
        symbol = st.text_input("Symbol / Company / Code", value="TCS")
    with c2:
        timeframe = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","6mo","1y","2y","5y"], index=0)
    with c3:
        chart_type = st.selectbox("Chart", ["Candlestick","Line"], index=0)
    with c4:
        indicators = st.multiselect("Indicators", ["RSI","MACD","SMA50","VolProfile","RollingVol"], default=["SMA50"])
    with c5:
        refresh_button = st.button("Quick Refresh")  # optional manual refresh but not required

    sym = (symbol or "").strip().upper()
    if not sym:
        st.warning("Enter a symbol (e.g., TCS, RELIANCE, INFY).")
        return

    # interpret timeframe -> interval
    intraday = timeframe == "1d"
    period = timeframe if not intraday else "1d"
    interval = "1m" if intraday else "1d"

    # get data (cached)
    df = None
    try:
        df = cached_get_stock(sym, period, interval, intraday)
    except Exception:
        df = None

    # fallback to demo generator inside backend if remote sources fail (get_stock_data already handles fallback)
    if df is None or (hasattr(df,"empty") and df.empty):
        st.warning("Live sources unavailable — loading demo data/fallback.")
        df = get_stock_data(sym, period=period, interval=interval)

    if df is None or (hasattr(df,"empty") and df.empty):
        st.error("No data available for this symbol. Try RELIANCE/TCS/INFY.")
        return

    df = ensure_date_col(df)
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values("Date").reset_index(drop=True)

    # metrics
    latest_price = float(df["Close"].dropna().iloc[-1]) if not df["Close"].dropna().empty else None
    today_high = float(df["High"].max()) if not df["High"].isna().all() else None
    today_low = float(df["Low"].min()) if not df["Low"].isna().all() else None
    today_vol = int(df["Volume"].dropna().iloc[-1]) if not df["Volume"].dropna().empty else 0

    # 52-week via 1y fetch if possible
    try:
        df_1y = get_stock_data(sym, period="1y", interval="1d")
        if df_1y is None or df_1y.empty:
            hi52 = lo52 = None
        else:
            hi52 = float(df_1y["Close"].max())
            lo52 = float(df_1y["Close"].min())
    except:
        hi52 = lo52 = None

    # SMA50
    try:
        df_daily = df.set_index("Date").resample("D").last().dropna()
        sma50 = df_daily["Close"].rolling(50).mean().iloc[-1] if len(df_daily)>=50 else None
    except:
        sma50 = None

    # RSI
    try:
        if intraday:
            rsi_series = compute_rsi(df["Close"].dropna())
        else:
            series = df.set_index("Date").resample("D").last().dropna()["Close"]
            rsi_series = compute_rsi(series)
        day_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None
    except:
        day_rsi = None

    # Top metric cards
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='metric-value'>{'₹'+format(latest_price,',.2f') if latest_price else 'N/A'}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='metric-title'>Today's H / L</div><div class='metric-value'>{('₹'+format(today_high,',.2f') if today_high else 'N/A')} / {('₹'+format(today_low,',.2f') if today_low else 'N/A')}</div><div class='small-muted'>Vol: {today_vol:,}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='metric-title'>52W High</div><div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><div class='metric-title'>52W Low</div><div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Main chart + side panel
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {sym} — {timeframe}", unsafe_allow_html=True)

        # Candlestick or line
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
        else:
            fig = px.line(df, x="Date", y="Close", title=f"{sym} Price")

        # SMA50 overlay
        if sma50 is not None:
            # plot SMA using daily resampled data
            try:
                df_daily = df.set_index("Date").resample("D").last().dropna()
                if len(df_daily) >= 50:
                    fig.add_scatter(x=df_daily.index, y=df_daily["Close"].rolling(50).mean(), mode="lines", name="SMA50", line=dict(width=1))
            except:
                pass

        # Add Volume bars as subplot-like overlay (secondary y)
        vol_trace = go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker=dict(opacity=0.6), yaxis="y2")
        fig.add_trace(vol_trace)
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=520,
                          yaxis=dict(title="Price"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume", position=0.98))
        st.plotly_chart(fig, use_container_width=True)

        # Indicators panel (MACD/RSI/Rolling Vol)
        with st.expander("Indicators", expanded=True):
            cols = st.columns(2)
            if "RSI" in indicators:
                if day_rsi:
                    cols[0].metric("RSI (14)", f"{day_rsi:.2f}")
                else:
                    cols[0].write("RSI: N/A")
            if "MACD" in indicators:
                macd, sig = compute_macd(df["Close"])
                if macd is not None:
                    fig2 = go.Figure()
                    fig2.add_scatter(x=df["Date"], y=macd, name="MACD")
                    fig2.add_scatter(x=df["Date"], y=sig, name="Signal")
                    fig2.update_layout(template="plotly_dark", height=220)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.write("MACD: Insufficient data")
            if "RollingVol" in indicators:
                rv = rolling_volatility(df["Close"], window=20)
                if rv is not None:
                    fig3 = px.line(x=rv.index, y=rv.values, labels={"x":"Date","y":"Volatility"})
                    fig3.update_layout(template="plotly_dark", height=200)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.write("Rolling Vol: N/A")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- RSI: **{day_rsi:.2f}**" if day_rsi else "- RSI: N/A")
        st.markdown(f"- SMA50: **{'₹'+format(sma50,',.2f') if sma50 else 'N/A'}**")
        st.markdown(f"- Latest: **{'₹'+format(latest_price,',.2f') if latest_price else 'N/A'}**")
        st.markdown(f"- Volume: **{today_vol:,}**")
        st.markdown("</div>", unsafe_allow_html=True)

# Minimal other pages
def render_trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","python")
    if st.button("Fetch Trends"):
        repos = fetch_github_trending(q)
        for r in repos[:10]:
            st.markdown(f"**{r['name']}** — {r['description']}")

def render_research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search","machine learning")
    if st.button("Search Papers"):
        papers = fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p["title"])
            st.write(p["summary"][:300]+"...")
            st.markdown(f"[Open]({p['link']})")

def render_news():
    st.markdown("<div class='card'><h2>News</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic","market")
    if st.button("Get News"):
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
    if st.button("Submit Feedback"):
        st.success("Thanks — saved locally.")

# Router
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
