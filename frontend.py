# frontend.py
# IntelliSphere - Frontend (robust, session_state-safe)
# Author: adapted for Debabrath (fixed AttributeError issue)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
from dateutil import tz

# backend helpers
from backend_modules import (
    get_stock_data,
    get_trends_keywords,
    fetch_github_trending,
    fetch_arxiv_papers,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)
import backend_modules as bm

# Page config
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# Safe session_state initialization (use only dict-style access later)
if "nav" not in st.session_state:
    st.session_state["nav"] = "home"
if "last_symbol" not in st.session_state:
    st.session_state["last_symbol"] = None
if "last_ticker" not in st.session_state:
    st.session_state["last_ticker"] = None
if "last_period" not in st.session_state:
    st.session_state["last_period"] = None
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
if "last_info" not in st.session_state:
    st.session_state["last_info"] = None

# Styles (white trendline-like)
st.markdown("""
<style>
.stApp { background: #ffffff; color: #0b1220; font-family: Inter, Roboto, Arial, sans-serif; }
h1,h2,h3{color:#0b1220}
.topbar{background:#fff;border-bottom:1px solid #e6e9ee;padding:12px 26px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;gap:14px;align-items:center}
.logo{width:44px;height:44px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:linear-gradient(90deg,#00b894,#00a8ff);color:#fff;font-weight:700;box-shadow:0 6px 18px rgba(0,0,0,0.06);font-size:18px}
.nav{display:flex;gap:14px;align-items:center}
.nav button{background:transparent;border:1px solid #e6eef6;padding:6px 12px;cursor:pointer;color:#2b6cb0;font-weight:600;border-radius:8px}
.nav button.active{background:#f3f7fb;color:#0b1220}
.card{background:#fbfdff;border:1px solid #e9eef6;border-radius:10px;padding:18px;box-shadow:0 6px 20px rgba(17,24,39,0.02)}
.metric-title{color:#6b7280;font-size:13px;margin-bottom:6px}
.metric-value{font-weight:700;font-size:22px;color:#0b1220}
.metric-sub{color:#4b5563;font-size:13px;margin-top:6px}
.divider{height:1px;background:#eef2f7;margin:14px 0;border-radius:2px}
.muted{color:#6b7280;font-size:13px}
@media (max-width:900px){.topbar{flex-direction:column;gap:8px;align-items:flex-start}.nav{flex-wrap:wrap}}
</style>
""", unsafe_allow_html=True)

# Header (buttons update session_state via streamlit buttons)
def header_bar():
    nav_items = [("Home","home"),("Stocks","stock"),("Trends","trends"),("Research","research"),
                 ("Skills","skills"),("News","news"),("Feedback","feedback")]
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.markdown('<div class="brand"><div class="logo">IS</div><div><div style="font-weight:700;font-size:16px">IntelliSphere</div><div class="muted" style="font-size:12px;margin-top:2px">AI-Powered Insights</div></div></div>', unsafe_allow_html=True)
    cols = st.columns(len(nav_items))
    for i,(label,key) in enumerate(nav_items):
        if cols[i].button(label, key=f"nav_{key}"):
            st.session_state["nav"] = key
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

def format_dividend(div_y):
    if div_y is None:
        return "N/A"
    try:
        d = float(div_y)
    except Exception:
        return "N/A"
    if d == 0: return "0.00%"
    if 0 < d < 0.5: pct = d*100
    elif 0.5 <= d <= 100: pct = d
    else: pct = d/100.0
    return f"{pct:.2f}%"

def safe_get_info(ticker):
    try:
        return bm.yf.Ticker(ticker).info
    except Exception:
        return {}

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

def upcoming_events_from_info(info):
    events = []
    now = datetime.now(timezone.utc)
    try:
        earn_ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
        if earn_ts:
            d = datetime.fromtimestamp(int(earn_ts), tz=timezone.utc)
            if d > now:
                events.append(("Earnings", d))
    except Exception:
        pass
    try:
        div_ts = info.get("dividendDate")
        if div_ts:
            d = datetime.fromtimestamp(int(div_ts), tz=timezone.utc)
            if d > now:
                events.append(("Dividend", d))
    except Exception:
        pass
    return events

# Stocks page
def render_stock():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Stock Insights Dashboard")

    c1,c2,c3 = st.columns([4,2,2])
    with c1:
        symbol = st.text_input("Enter company symbol (e.g., TCS or INFY.NS):", value=(st.session_state.get("last_symbol") or "TCS"))
    with c2:
        period = st.selectbox("Select time range:", ["1d","5d","1mo","3mo","6mo","1y"], index=2)
    with c3:
        chart_type = st.radio("Chart type:", ["Candlestick","Line"], index=0, horizontal=True)

    e1,e2 = st.columns(2)
    with e1:
        show_ema = st.checkbox("Show EMA (12 & 26)", value=True)
    with e2:
        show_rsi = st.checkbox("Show RSI (14)", value=False)

    fetch = st.button("Fetch Stock Data")

    symbol_in = (symbol.strip().upper() if symbol else "")
    need_fetch = False
    if fetch:
        need_fetch = True
    elif st.session_state.get("last_df") is None:
        need_fetch = True
    elif st.session_state.get("last_symbol") != symbol_in or st.session_state.get("last_period") != period:
        need_fetch = True

    df = None
    info = {}
    ticker_used = None

    if need_fetch:
        candidates = [symbol_in]
        if "." not in symbol_in:
            candidates.append(symbol_in + ".NS")
        for t in candidates:
            try:
                df_try = get_stock_data(t, period=period)
                if df_try is not None and not df_try.empty:
                    info_try = safe_get_info(t) or {}
                    df = df_try.copy()
                    info = info_try
                    ticker_used = t
                    break
            except Exception:
                continue
        if df is None:
            st.error("⚠️ No data available for provided symbol. Try another ticker (e.g., TCS, INFY.NS).")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.session_state["last_symbol"] = symbol_in
        st.session_state["last_ticker"] = ticker_used
        st.session_state["last_period"] = period
        st.session_state["last_df"] = df.copy()
        st.session_state["last_info"] = info.copy() if isinstance(info, dict) else {}
    else:
        df = st.session_state.get("last_df")
        info = st.session_state.get("last_info", {})
        ticker_used = st.session_state.get("last_ticker", symbol_in)

    if df is None or (hasattr(df, "empty") and df.empty):
        st.error("No data available to render.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # normalize and safe-guard columns
    df = ensure_date_col(df)
    # ensure df is DataFrame
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            st.error("Unexpected data format from backend.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            # assign a column of NaNs matching df length
            try:
                df[col] = np.nan
            except Exception:
                # fallback: create Series then assign
                df = df.join(pd.Series([np.nan]*len(df), name=col))

    df = df.sort_values("Date").reset_index(drop=True)

    # safe scalars
    try:
        latest_price = float(df["Close"].dropna().iloc[-1])
    except Exception:
        latest_price = None
    try:
        first_price = float(df["Close"].dropna().iloc[0])
    except Exception:
        first_price = None
    pct_change = ((latest_price - first_price) / first_price * 100) if (latest_price is not None and first_price and first_price != 0) else 0.0

    # info fields
    pe_ratio = info.get("trailingPE", "N/A")
    market_cap = info.get("marketCap", 0)
    volume_info = info.get("volume", np.nan)
    high_52w = info.get("fiftyTwoWeekHigh", "N/A")
    low_52w = info.get("fiftyTwoWeekLow", "N/A")
    dividend_y = info.get("dividendYield", None)
    dividend_display = format_dividend(dividend_y)

    # top small cards
    def render_small_card(title, value, sub=None):
        st.markdown(f"""<div class="card" style="flex:1;margin-right:8px">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub if sub else ''}</div>
            </div>""", unsafe_allow_html=True)

    cols_top = st.columns(4)
    render_small_card("Current Price", f"₹{latest_price:,.2f}" if latest_price is not None else "N/A", f"{pct_change:.2f}% since period start")
    render_small_card("P/E Ratio", f"{pe_ratio}", f"52W High: {high_52w}")
    render_small_card("Market Cap", humanize_number(market_cap), f"52W Low: {low_52w}")
    render_small_card("Volume", humanize_number(volume_info), f"Dividend Yield: {dividend_display}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # technical summary
    technical_lines = []
    try:
        if show_rsi and "Close" in df.columns:
            closes = df["Close"].dropna()
            if len(closes) >= 15:
                delta = closes.diff().dropna()
                up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                down = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
                rs = up / (down + 1e-9)
                rsi_val = 100 - (100 / (1 + rs))
                technical_lines.append(f"RSI(14): {rsi_val:.1f}")
        if "Close" in df.columns and len(df) >= 30:
            sma20 = df["Close"].rolling(20).mean().iloc[-1]
            sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            if sma50:
                technical_lines.append("Momentum: bullish (SMA20 > SMA50)" if sma20> sma50 else "Momentum: bearish (SMA20 < SMA50)")
    except Exception:
        pass

    if technical_lines:
        st.info(" • ".join(technical_lines))
    else:
        st.caption("Technical view: short-term data available.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Chart + right panel
    left_col, right_col = st.columns([3,1])
    with left_col:
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
            if len(df) >= 20:
                df["SMA20"] = df["Close"].rolling(20).mean()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], mode="lines", name="SMA20", line=dict(dash="dot", width=1)))
            if show_ema:
                df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
                df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA12"], mode="lines", name="EMA12", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA26"], mode="lines", name="EMA26", line=dict(width=1)))
            fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.line(df, x="Date", y="Close", title=f"{(st.session_state.get('last_symbol') or symbol_in).upper()} Price", markers=True)
            if show_ema:
                df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
                df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
                fig.add_scatter(x=df["Date"], y=df["EMA12"], mode="lines", name="EMA12")
                fig.add_scatter(x=df["Date"], y=df["EMA26"], mode="lines", name="EMA26")
            fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        if show_rsi and "Close" in df.columns and len(df) >= 15:
            closes = df["Close"].dropna()
            delta = closes.diff().dropna()
            up = delta.clip(lower=0).rolling(14).mean()
            down = -delta.clip(upper=0).rolling(14).mean()
            rs = up/(down + 1e-9)
            rsi_series = 100 - (100/(1+rs))
            rsi_fig = px.line(x=df["Date"].iloc[-len(rsi_series):], y=rsi_series.values, labels={"x":"Date","y":"RSI"})
            rsi_fig.update_layout(template="plotly_white", height=140, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280'>{(st.session_state.get('last_symbol') or symbol_in).upper()} | {ticker_used}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:28px;font-weight:700;margin-top:6px'>₹{(float(latest_price) if latest_price is not None else 0):,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#10b981;font-weight:600;margin-top:6px'>{pct_change:.2f}%</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        try:
            sma_lines = []
            for w in [5,10,20,30,50,100,150,200]:
                if len(df) >= w:
                    val = df["Close"].rolling(w).mean().iloc[-1]
                    sma_lines.append((f"{w} Day SMA", f"{val:.2f}"))
            sma_html = "<table style='width:100%;font-size:13px'>"
            for name,val in sma_lines:
                sma_html += f"<tr><td style='color:#6b7280'>{name}</td><td style='text-align:right;font-weight:700'>{val}</td></tr>"
            sma_html += "</table>"
            st.markdown(sma_html, unsafe_allow_html=True)
        except Exception:
            pass
        st.markdown("</div>", unsafe_allow_html=True)

    # upcoming events (future only)
    events = upcoming_events_from_info(info)
    if events:
        ev_text = " • ".join([f"{typ} on {dt.astimezone(tz.tzlocal()).strftime('%d %b %Y')}" for typ,dt in events])
        st.info(ev_text)

    # Technical indicators grid
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("### Technical Indicators")
    left, right = st.columns([2,1])
    with left:
        indicators = []
        try:
            closes = df["Close"].dropna()
            if len(closes) >= 14:
                delta = closes.diff().dropna()
                up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                down = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
                rs = up / (down + 1e-9)
                rsi_val = 100 - (100 / (1 + rs))
                indicators.append(("Day RSI (14)", f"{rsi_val:.1f}", "Mid-range" if 30<=rsi_val<=70 else "Overbought" if rsi_val>70 else "Oversold"))
            if len(closes) >= 26:
                ema12 = closes.ewm(span=12, adjust=False).mean()
                ema26 = closes.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                indicators.append(("MACD (12,26,9)", f"{macd.iloc[-1]:.2f}", f"Signal: {macd_signal.iloc[-1]:.2f}"))
            if "High" in df.columns and "Low" in df.columns and len(df) >= 14:
                high = df["High"]; low = df["Low"]; prev_close = df["Close"].shift(1)
                tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                indicators.append(("ATR (14)", f"{atr:.2f}", "Average True Range"))
        except Exception:
            pass

        cols = st.columns(2)
        idx = 0
        for title,val,sub in indicators:
            c = cols[idx % 2]
            c.markdown(f"<div class='card' style='margin-bottom:12px'><div class='metric-title'>{title}</div><div class='metric-value'>{val}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)
            idx += 1

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;margin-bottom:8px'>Quick Interpretation</div>", unsafe_allow_html=True)
        interpret = []
        try:
            if 'rsi_val' in locals():
                if rsi_val > 70:
                    interpret.append("RSI indicates Overbought")
                elif rsi_val < 30:
                    interpret.append("RSI indicates Oversold")
                else:
                    interpret.append("RSI is mid-range")
            if 'macd' in locals() and 'macd_signal' in locals():
                interpret.append("MACD above signal — bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "MACD below signal — bearish")
        except Exception:
            pass
        if interpret:
            for line in interpret:
                st.markdown(f"- {line}")
        else:
            st.markdown("No quick interpretation available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Simple other pages (kept functional)
def render_trends():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Tech Trends")
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
    st.markdown("</div>", unsafe_allow_html=True)

def render_research():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Research & Education")
    topic = st.text_input("Search papers on:", "machine learning")
    if st.button("Fetch Papers"):
        with st.spinner("Fetching papers..."):
            papers = fetch_arxiv_papers(topic, max_results=6)
        if not papers:
            st.warning("No papers found.")
        else:
            for p in papers:
                st.subheader(p["title"])
                st.caption(", ".join(p["authors"]))
                st.write(p["summary"][:600] + "...")
                st.markdown(f"[Read more]({p['link']})")
                st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

def render_skills():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Skill & Job Trends")
    skills = st.text_input("Enter skills (comma-separated):", "Python, SQL")
    if st.button("Analyze"):
        keys = [k.strip() for k in skills.split(",")]
        with st.spinner("Fetching trends..."):
            trends = get_trends_keywords(keys)
        if trends:
            df = pd.DataFrame([{"Skill": k, "Change (%)": trends[k]["pct_change"]} for k in keys if k in trends])
            fig = px.bar(df, x="Skill", y="Change (%)", color="Skill")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Trends unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_news():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## News & Sentiment")
    query = st.text_input("Enter topic or company:", "Indian Stock Market")
    num_articles = st.slider("Num articles:", 3, 15, 6)
    if st.button("Get News"):
        with st.spinner("Fetching news..."):
            arts = get_news(query, max_items=num_articles)
            sents = analyze_headlines_sentiment(arts)
        for a in sents:
            col1,col2 = st.columns([4,1])
            with col1:
                st.markdown(f"**{a['title']}**")
                if a.get("link"):
                    st.markdown(f"[Read more]({a['link']})")
                st.caption(a['published'].strftime("%b %d, %Y %H:%M"))
            with col2:
                lab = a['sentiment']['label']
                sc = a['sentiment']['score']
                badge = "🟢" if "POS" in lab else "🔴" if "NEG" in lab else "⚪"
                st.markdown(f"**{badge} {lab}**\n\n{sc:.2f}")
            st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

def render_feedback():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Feedback")
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

# Main render
def render_dashboard():
    header_bar()
    page = st.session_state.get("nav", "home")
    st.markdown("<div style='padding:20px'>", unsafe_allow_html=True)
    if page == "home":
        st.title("Welcome to IntelliSphere")
        st.markdown("Your AI dashboard for stocks, trends, research and news.")
        st.success("All systems operational!")
    elif page == "stock":
        render_stock()
    elif page == "trends":
        render_trends()
    elif page == "research":
        render_research()
    elif page == "skills":
        render_skills()
    elif page == "news":
        render_news()
    elif page == "feedback":
        render_feedback()
    else:
        st.write("Page not found.")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    render_dashboard()
