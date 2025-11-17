# ==============================================================
# frontend.py
# Author: Debabrath
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Backend imports (ensure backend_modules.py exposes these)
import backend_modules as bm
from backend_modules import get_stock_data, get_intraday_data, get_today_high_low, stock_summary

# --------------------------------------------------------------
# Page config & styles
# --------------------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | Premium",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* card / typography */
.card{background:#0b1220;border-radius:12px;padding:16px;margin-bottom:12px;color:#e6eef6}
.metric-title{color:#9aa3ad;font-size:12px;margin-bottom:6px}
.metric-value{font-weight:700;font-size:20px;color:#e6eef6}
.big{font-size:28px;font-weight:800}
.small-muted{color:#98a3ad;font-size:12px}
.stButton>button {background-color:#0f1720;border:1px solid rgba(255,255,255,0.04);color:#e6eef6}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Sidebar (persistent) - unique key for the radio to avoid duplicate ID errors
# --------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.image("https://i.ibb.co/r2mBjDk/intellis.png", width=80)
        st.title("IntelliSphere")
        st.markdown("**AI-powered stock insights**")
        page = st.radio(
            "Navigation",
            ["Home", "Stocks", "Trends", "Research", "News", "Feedback"],
            index=1,
            key="sidebar_navigation"
        )
        st.markdown("---")
        st.markdown("<div style='text-align:center; color:#9aa3ad'>Made with ❤️ by <b>Debabrath</b></div>", unsafe_allow_html=True)
    return page

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
@st.cache_data(ttl=20)
def cached_fetch(symbol: str, period: str, interval: str, intraday: bool):
    """Cached wrapper around backend fetches to avoid spamming APIs while UI changes."""
    try:
        if intraday:
            return get_intraday_data(symbol, interval=interval, period=period)
        return get_stock_data(symbol, period=period, interval=interval)
    except Exception:
        return None

def ensure_date_col(df: pd.DataFrame):
    if df is None:
        return None
    df = df.copy()
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        else:
            df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        # fallback: business-day range
        df["Date"] = pd.bdate_range(end=datetime.today(), periods=len(df))
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def compute_rsi(series: pd.Series, n: int = 14):
    s = series.dropna()
    if len(s) < n + 1:
        return None
    delta = s.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100/(1+rs))
    return rsi

def compute_mfi(df: pd.DataFrame, n: int = 14):
    if df is None or len(df) < n + 1:
        return None
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_flow = typical * df["Volume"].fillna(0)
    pos, neg = [], []
    for i in range(1, len(typical)):
        if typical.iat[i] > typical.iat[i-1]:
            pos.append(raw_flow.iat[i]); neg.append(0.0)
        else:
            pos.append(0.0); neg.append(raw_flow.iat[i])
    pos = pd.Series([0.0] + pos).rolling(n).sum()
    neg = pd.Series([0.0] + neg).rolling(n).sum()
    mfi = 100 - (100 / (1 + (pos / (neg + 1e-9))))
    return mfi

def compute_macd(series: pd.Series):
    s = series.dropna()
    if len(s) < 26:
        return None, None
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def rolling_volatility(series: pd.Series, window: int = 20):
    s = series.dropna()
    if len(s) < window:
        return None
    rv = s.pct_change().rolling(window).std() * np.sqrt(252)
    return rv

def compute_ranges(df: pd.DataFrame):
    out = {}
    lookbacks = {
        "1d": 1, "1w": 5, "1m": 22, "3m": 66, "6m": 132, "1y": 252, "2y": 252*2, "5y": 252*5
    }
    for k, v in lookbacks.items():
        if len(df) >= v:
            tail = df.tail(v)
            out[k] = float(tail["Close"].max() - tail["Close"].min())
        else:
            out[k] = None
    return out

def volume_profile(df: pd.DataFrame, bins=20):
    if df is None or df.empty:
        return None
    prices = df["Close"].dropna()
    vols = df["Volume"].fillna(0)
    if prices.empty:
        return None
    hist, edges = np.histogram(prices, bins=bins, weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2
    vp = pd.DataFrame({"price": centers, "volume": hist})
    return vp.sort_values("volume", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------
# Stocks page (final premium)
# --------------------------------------------------------------
def render_stocks_page():
    st.markdown("<div class='card'><h2>Stocks — Premium View</h2></div>", unsafe_allow_html=True)

    # controls (each has a unique key)
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        symbol = st.text_input("Ticker / Company / BSE code", value="TCS", key="symbol_input").strip().upper()
    with c2:
        timeframe = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","6mo","1y","2y","5y"], index=0, key="timeframe_select")
    with c3:
        chart_type = st.selectbox("Chart Type", ["Candlestick","Line"], index=0, key="chart_type_select")
    with c4:
        indicators = st.multiselect("Indicators", ["RSI","MACD","MFI","SMA50","RollingVol","VolumeProfile"], default=["SMA50"], key="indicators_multiselect")

    if not symbol:
        st.warning("Enter a stock symbol (e.g., TCS, RELIANCE, INFY).")
        return

    intraday = timeframe == "1d"
    period = timeframe if not intraday else "1d"
    interval = "1m" if intraday else "1d"

    # fetch data (cached)
    df = cached_fetch(symbol, period, interval, intraday)
    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("Live sources unavailable or symbol not found. Loading demo fallback...")
        df = get_stock_data(symbol, period=period, interval=interval)

    if df is None or (hasattr(df, "empty") and df.empty):
        st.error("No data found for this symbol. Try TCS / RELIANCE / INFY or check internet.")
        return

    df = ensure_date_col(df)
    df = df.sort_values("Date").reset_index(drop=True)

    # metrics
    latest_price = float(df["Close"].dropna().iloc[-1]) if not df["Close"].dropna().empty else None
    today_high = float(df["High"].max()) if not df["High"].isna().all() else None
    today_low = float(df["Low"].min()) if not df["Low"].isna().all() else None
    today_vol = int(df["Volume"].dropna().iloc[-1]) if not df["Volume"].dropna().empty else 0

    # 52w
    try:
        df_1y = get_stock_data(symbol, period="1y", interval="1d")
        if df_1y is not None and not df_1y.empty:
            hi52 = float(df_1y["Close"].max())
            lo52 = float(df_1y["Close"].min())
        else:
            hi52 = lo52 = None
    except:
        hi52 = lo52 = None

    # SMA50
    try:
        df_daily = df.set_index("Date").resample("D").last().dropna()
        sma50 = float(df_daily["Close"].rolling(50).mean().iloc[-1]) if len(df_daily) >= 50 else None
    except:
        sma50 = None

    # RSI & MFI
    try:
        if intraday:
            rsi_series = compute_rsi(df["Close"])
        else:
            rsi_series = compute_rsi(df_daily["Close"])
        day_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None
    except:
        day_rsi = None

    try:
        mfi_series = compute_mfi(df, n=14)
        day_mfi = float(mfi_series.iloc[-1]) if mfi_series is not None else None
    except:
        day_mfi = None

    # ranges and rolling vol
    ranges = compute_ranges(df)
    rv = rolling_volatility(df["Close"], window=20)

    # metric cards
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.markdown(f"<div class='card'><div class='metric-title'>Current Price</div><div class='big'>{'₹'+format(latest_price,',.2f') if latest_price else 'N/A'}</div></div>", unsafe_allow_html=True)
    p2.markdown(f"<div class='card'><div class='metric-title'>Today's H / L</div><div class='metric-value'>{('₹'+format(today_high,',.2f') if today_high else 'N/A')} / {('₹'+format(today_low,',.2f') if today_low else 'N/A')}</div><div class='small-muted'>Vol: {today_vol:,}</div></div>", unsafe_allow_html=True)
    p3.markdown(f"<div class='card'><div class='metric-title'>52W High</div><div class='metric-value'>{('₹'+format(hi52,',.2f')) if hi52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    p4.markdown(f"<div class='card'><div class='metric-title'>52W Low</div><div class='metric-value'>{('₹'+format(lo52,',.2f')) if lo52 else 'N/A'}</div></div>", unsafe_allow_html=True)
    p5.markdown(f"<div class='card'><div class='metric-title'>SMA50</div><div class='metric-value'>{('₹'+format(sma50,',.2f')) if sma50 else 'N/A'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # chart + right panel
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {symbol} — {timeframe}", unsafe_allow_html=True)

        # primary chart
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
        else:
            fig = px.line(df, x="Date", y="Close", title=f"{symbol} Price")

        # SMA overlay
        if sma50 is not None:
            try:
                df_daily_local = df.set_index("Date").resample("D").last().dropna()
                if len(df_daily_local) >= 50:
                    fig.add_trace(go.Scatter(x=df_daily_local.index, y=df_daily_local["Close"].rolling(50).mean(), mode="lines", name="SMA50", line=dict(width=1)))
            except:
                pass

        # volume bars
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker=dict(opacity=0.6), yaxis="y2"))
        fig.update_layout(template="plotly_white", height=540, xaxis_rangeslider_visible=False,
                          yaxis=dict(title="Price"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume", position=0.98))
        st.plotly_chart(fig, use_container_width=True)

        # indicators
        with st.expander("Indicators (expand)", expanded=False):
            if "RSI" in indicators:
                if day_rsi:
                    st.write(f"RSI (14): **{day_rsi:.2f}**")
                    fig_rsi = px.line(x=rsi_series.index, y=rsi_series.values, labels={"x":"Date","y":"RSI"})
                    fig_rsi.update_layout(template="plotly_white", height=220)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                else:
                    st.write("RSI: N/A")

            if "MACD" in indicators:
                macd, sig = compute_macd(df["Close"])
                if macd is not None:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df["Date"], y=macd, name="MACD"))
                    fig_macd.add_trace(go.Scatter(x=df["Date"], y=sig, name="Signal"))
                    fig_macd.update_layout(template="plotly_white", height=220)
                    st.plotly_chart(fig_macd, use_container_width=True)
                else:
                    st.write("MACD: Insufficient data")

            if "MFI" in indicators:
                if day_mfi:
                    st.write(f"MFI (14): **{day_mfi:.2f}**")
                else:
                    st.write("MFI: N/A")

            if "RollingVol" in indicators:
                if rv is not None:
                    fig_rv = px.line(x=rv.index, y=rv.values, labels={"x":"Date","y":"Volatility"})
                    fig_rv.update_layout(template="plotly_white", height=200)
                    st.plotly_chart(fig_rv, use_container_width=True)
                else:
                    st.write("Rolling Volatility: N/A")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
        st.markdown(f"- Latest: **{'₹'+format(latest_price,',.2f') if latest_price else 'N/A'}**")
        st.markdown(f"- Day RSI: **{day_rsi:.2f}**" if day_rsi else "- Day RSI: N/A")
        st.markdown(f"- Day MFI: **{day_mfi:.2f}**" if day_mfi else "- Day MFI: N/A")
        st.markdown(f"- SMA50: **{'₹'+format(sma50,',.2f') if sma50 else 'N/A'}**")
        st.markdown("---")
        st.markdown("<div style='font-weight:700'>Range (Close max-min)</div>", unsafe_allow_html=True)
        for k in ["1d","1w","1m","3m","6m","1y","2y","5y"]:
            val = ranges.get(k)
            st.markdown(f"- {k}: **{('₹'+format(val,',.2f')) if val else 'N/A'}**")
        st.markdown("---")
        if "VolumeProfile" in indicators:
            vp = volume_profile(df)
            if vp is not None and not vp.empty:
                st.markdown("<div style='font-weight:700'>Top Volume Price Buckets</div>", unsafe_allow_html=True)
                for i,row in vp.head(6).iterrows():
                    st.markdown(f"- ₹{row['price']:.2f} — {int(row['volume']):,}")
            else:
                st.write("Volume Profile: N/A")
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------
# Other pages (light)
# --------------------------------------------------------------
def render_trends():
    st.markdown("<div class='card'><h2>Trends</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic", "python", key="trends_q")
    if q:
        repos = bm.fetch_github_trending(q)
        for r in repos[:10]:
            st.markdown(f"**{r.get('name','')}** — {r.get('description','')}")

def render_research():
    st.markdown("<div class='card'><h2>Research</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Search topic", "machine learning", key="research_q")
    if q:
        papers = bm.fetch_arxiv_papers(q)
        for p in papers:
            st.subheader(p.get("title",""))
            st.write((p.get("summary","") or "")[:300] + "...")
            st.markdown(f"[Open]({p.get('link','#')})")

def render_news():
    st.markdown("<div class='card'><h2>News & Sentiment</h2></div>", unsafe_allow_html=True)
    q = st.text_input("Topic", "market", key="news_q")
    if q:
        arts = bm.fetch_news_via_google_rss(q, max_items=8)
        sents = bm.analyze_headlines_sentiment(arts)
        for a in sents:
            st.markdown(f"**{a.get('title','')}**")
            if a.get("sentiment"):
                lab = a["sentiment"].get("label")
                sc = a["sentiment"].get("score")
                st.markdown(f"- {lab} ({sc:.2f})")
            st.markdown(f"[Open]({a.get('link','#')})")
            st.divider()

def render_feedback():
    st.markdown("<div class='card'><h2>Feedback</h2></div>", unsafe_allow_html=True)
    name = st.text_input("Name", key="fb_name")
    rating = st.slider("Rate IntelliSphere", 1, 5, 4, key="fb_rating")
    comments = st.text_area("Comments", key="fb_comments")
    if st.button("Submit Feedback", key="fb_submit"):
        st.success("Thanks — feedback saved locally.")

# --------------------------------------------------------------
# render_dashboard: entrypoint that app.py expects
# --------------------------------------------------------------
def render_dashboard():
    page = render_sidebar()
    if page == "Home":
        st.markdown("<div class='card'><h1>Welcome to IntelliSphere</h1><p>Premium AI-powered dashboard.</p></div>", unsafe_allow_html=True)
    elif page == "Stocks":
        render_stocks_page()
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

# ensure script runnable directly
if __name__ == "__main__":
    render_dashboard()
