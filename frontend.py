# ==============================================
# IntelliSphere Frontend — Trendlyne-like Technical Dashboard
# Author: Generated for Debabrath
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import backend_modules as bm
from backend_modules import (
    fetch_github_trending,
    fetch_arxiv_papers,
    get_trends_keywords,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)

# -------------------- Page config --------------------
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide")
# -------------------- Ensure session keys --------------------
def _init_state():
    if "nav" not in st.session_state: st.session_state["nav"] = "home"
    if "last_symbol" not in st.session_state: st.session_state["last_symbol"] = None
    if "last_df" not in st.session_state: st.session_state["last_df"] = None
    if "last_info" not in st.session_state: st.session_state["last_info"] = None
    if "last_period" not in st.session_state: st.session_state["last_period"] = None
    if "tech_cache" not in st.session_state: st.session_state["tech_cache"] = {}
_init_state()

# -------------------- Styling: neon + persistent active nav --------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(-45deg,#060b17,#071227,#031321,#051321); background-size:400% 400%; animation: bg 20s ease infinite; color:#eafcff; }
    @keyframes bg {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
    h1,h2,h3{ color:#00e6ff !important; text-shadow:0 0 12px rgba(0,230,255,0.6)}
    .navbar{display:flex; gap:10px; justify-content:center; margin-bottom:18px;}
    .navbtn{background:#0f1720; border:1px solid #00e6ff; color:#00e6ff; padding:8px 16px; border-radius:999px; cursor:pointer;}
    .navbtn.active{background:linear-gradient(90deg,#00e6ff,#7a00ff); color:#021; box-shadow:0 8px 30px rgba(0,230,255,0.08);}
    .metric-card{background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); padding:12px; border-radius:10px}
    .small-muted{color:#9fbfcf; font-size:13px}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- NAVBAR --------------------
def navbar():
    items = [("Home","home"),("Stocks","stock"),("Trends","trends"),("Research","research"),("Skills","skills"),("News","news"),("Feedback","feedback")]
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    cols = st.columns(len(items))
    for i,(label,key) in enumerate(items):
        is_active = (st.session_state["nav"]==key)
        btn = cols[i].button(label, key=f"nav_{key}")
        if btn:
            st.session_state["nav"] = key
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Indicator helpers --------------------
def humanize(n):
    try:
        n = float(n)
        if n >= 1e12: return f"{n/1e12:.2f} Tn"
        if n >= 1e7: return f"{n/1e7:.2f} Cr"
        if n >= 1e5: return f"{n/1e5:.2f} L"
        if n >= 1e3: return f"{n/1e3:.2f} K"
        return str(round(n,2))
    except:
        return "N/A"

def dividend_yield_from_info(info, current_price):
    """Best-effort dividend yield calculation:
       prefer dividendRate/currentPrice, fallback to dividendYield field (with autoscale fix)."""
    try:
        div_rate = info.get("dividendRate")
        if div_rate and current_price and current_price>0:
            return round((float(div_rate)/float(current_price))*100, 2)
        raw = info.get("dividendYield")
        if raw is None: return "N/A"
        v = float(raw)
        # if value > 10, assume it's been returned as 'percentage' (e.g., 58 meaning 58%), convert to fraction
        if v > 10:
            v = v / 100.0
        return round(v*100, 2)
    except:
        return "N/A"

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(span=period, adjust=False).mean()
    ma_down = down.ewm(span=period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_atr(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def compute_mfi(df, period=14):
    typical = (df['High'] + df['Low'] + df['Close'])/3
    money_flow = typical * df['Volume']
    posmf = []
    negmf = []
    for i in range(1,len(typical)):
        if typical.iloc[i] > typical.iloc[i-1]:
            posmf.append(money_flow.iloc[i])
            negmf.append(0)
        else:
            posmf.append(0)
            negmf.append(money_flow.iloc[i])
    posmf = pd.Series([0]+posmf)
    negmf = pd.Series([0]+negmf)
    mf_ratio = posmf.rolling(window=period, min_periods=1).sum() / (negmf.rolling(window=period, min_periods=1).sum()+1e-9)
    mfi = 100 - (100 / (1 + mf_ratio))
    return mfi

def compute_adx(df, period=14):
    # Basic ADX implementation
    high = df['High']; low = df['Low']; close = df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move>down_move)&(up_move>0), up_move, 0.0)
    minus_dm = np.where((down_move>up_move)&(down_move>0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (tr_smooth + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (tr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0)

def compute_roc(series, period=12):
    return (series.pct_change(periods=period))*100

# -------------------- Predictive Technical Summary --------------------
def technical_summary(df, info):
    # compute indicators
    close = df['Close']
    rsi = compute_rsi(close).iloc[-1]
    macd, signal, hist = compute_macd(close)
    macd_latest = macd.iloc[-1]; signal_latest = signal.iloc[-1]
    ema12 = compute_ema(close,12).iloc[-1]
    ema26 = compute_ema(close,26).iloc[-1]

    # RSI zone
    if rsi < 30:
        rsi_tag = ("Oversold", "green")
    elif rsi > 70:
        rsi_tag = ("Overbought", "red")
    else:
        rsi_tag = ("Neutral", "yellow")

    # MACD momentum
    macd_tag = ("Bullish" if macd_latest > signal_latest else "Bearish") if not np.isnan(macd_latest) else ("Neutral")

    # EMA trend
    ema_tag = ("Bullish" if ema12 > ema26 else "Bearish")

    # Compose short natural language insight
    insight = []
    if rsi_tag[0]=="Oversold":
        insight.append("RSI indicates oversold conditions — possible rebound.")
    elif rsi_tag[0]=="Overbought":
        insight.append("RSI indicates overbought conditions — watch for pullback.")

    if macd_tag=="Bullish":
        insight.append("MACD above signal line — momentum supports upside.")
    elif macd_tag=="Bearish":
        insight.append("MACD below signal line — momentum weakening.")

    if ema_tag=="Bullish":
        insight.append("Short-term EMA above long-term EMA — short-term bullish bias.")
    else:
        insight.append("Short-term EMA below long-term EMA — short-term bearish bias.")

    return {
        "rsi": round(float(rsi),2) if not np.isnan(rsi) else None,
        "macd": round(float(macd_latest),4) if not np.isnan(macd_latest) else None,
        "macd_signal": round(float(signal_latest),4) if not np.isnan(signal_latest) else None,
        "ema12": round(float(ema12),4) if not np.isnan(ema12) else None,
        "ema26": round(float(ema26),4) if not np.isnan(ema26) else None,
        "insight": " ".join(insight)
    }

# -------------------- Sections --------------------
def render_home():
    st.title("🌐 IntelliSphere — Trendlyne-style Dashboard")
    st.write("Welcome — use the Stocks tab for a full technical view similar to Trendlyne.")
    st.divider()

def render_stock():
    st.header("💹 Stock Insights (Technical Dashboard)")

    # inputs
    symbol = st.text_input("Enter company symbol:", st.session_state.get("last_symbol") or "TCS")
    period = st.selectbox("Select time range:", ["1d","5d","1mo","3mo","6mo","1y"], index=2)
    fetch_btn = st.button("Fetch Stock Data")
    chart_type = st.radio("Chart Type", ["Candlestick","Line"], index=0, horizontal=True)
    show_ema = st.checkbox("Show EMA (12 & 26)", value=True)
    show_rsi = st.checkbox("Show RSI (14)", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_technical_dashboard = st.checkbox("Show full Technical Dashboard (SMA/EMA/MACD/RSI/ADX/ATR/ROC)", value=True)

    # Fetching logic & caching in session_state
    need_fetch = fetch_btn or (symbol != st.session_state.get("last_symbol")) or (period != st.session_state.get("last_period"))
    if need_fetch:
        tickers = [symbol.strip().upper(), symbol.strip().upper()+".NS"]
        df = None; info = {}
        for t in tickers:
            try:
                df = bm.get_stock_data(t, period=period)
                if df is not None and not df.empty:
                    info = bm.yf.Ticker(t).info
                    used_ticker = t
                    break
            except Exception:
                continue
        if df is None or df.empty:
            st.error("No data found for this symbol / period.")
            return
        # store
        st.session_state["last_symbol"] = symbol
        st.session_state["last_df"] = df
        st.session_state["last_info"] = info
        st.session_state["last_period"] = period

    df = st.session_state.get("last_df")
    info = st.session_state.get("last_info")
    if df is None or info is None:
        st.info("Enter a symbol and click Fetch Stock Data.")
        return

    # clean dataframe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Datetime":"Date","date":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # metrics
    current_price = float(df["Close"].iloc[-1])
    first_price = float(df["Close"].iloc[0])
    change_pct = round((current_price - first_price)/ (first_price+1e-9) * 100,2)
    pe = info.get("trailingPE","N/A")
    market_cap = humanize(info.get("marketCap",0))
    volume = humanize(info.get("volume",0))
    fiftyTwoHigh = info.get("fiftyTwoWeekHigh","N/A")
    fiftyTwoLow = info.get("fiftyTwoWeekLow","N/A")
    # accurate dividend yield: prefer dividendRate/currentPrice else robust fallback
    div_yield = dividend_yield_from_info(info, current_price)

    # Top overview cards (Trendlyne-like)
    top_cols = st.columns([2,2,2,2])
    with top_cols[0]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader(f"₹{round(current_price,2)}")
        st.write(f"**{change_pct}%** since period start")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_cols[1]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader(f"P/E: {pe}")
        st.write(f"52W High: {fiftyTwoHigh}")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_cols[2]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader(f"Market Cap: {market_cap}")
        st.write(f"52W Low: {fiftyTwoLow}")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_cols[3]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader(f"Volume: {volume}")
        st.write(f"Dividend Yield: {div_yield}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # compute on-demand indicators and cache them (avoid recompute)
    cache_key = f"{st.session_state['last_symbol']}_{st.session_state['last_period']}"
    if cache_key not in st.session_state["tech_cache"]:
        tech = {}
        tech["ema12"] = compute_ema(df["Close"],12)
        tech["ema26"] = compute_ema(df["Close"],26)
        tech["rsi14"] = compute_rsi(df["Close"],14)
        tech["macd"], tech["macd_signal"], tech["macd_hist"] = compute_macd(df["Close"])
        tech["atr14"] = compute_atr(df,14)
        tech["mfi14"] = compute_mfi(df,14)
        tech["adx14"] = compute_adx(df,14)
        tech["roc12"] = compute_roc(df["Close"],12)
        # simple SMAs
        tech["sma5"] = df["Close"].rolling(5).mean()
        tech["sma10"] = df["Close"].rolling(10).mean()
        tech["sma20"] = df["Close"].rolling(20).mean()
        tech["sma50"] = df["Close"].rolling(50).mean()
        tech["sma200"] = df["Close"].rolling(200).mean()
        st.session_state["tech_cache"][cache_key] = tech
    else:
        tech = st.session_state["tech_cache"][cache_key]

    # Technical summary card
    t_summary = technical_summary(df, info)
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"**Technical Summary:** {t_summary['insight']}")
    st.markdown(f"**RSI(14):** {t_summary['rsi']}   •   **MACD:** {t_summary['macd']} (signal {t_summary['macd_signal']})")
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    # Upcoming events: only future events (earnings/dividend)
    upcoming = []
    try:
        if info.get("earningsTimestampStart"):
            dt = pd.to_datetime(info.get("earningsTimestampStart"), unit="s")
            if dt.date() >= datetime.now().date():
                upcoming.append(("Earnings", dt))
        elif info.get("earningsTimestamp"):
            dt = pd.to_datetime(info.get("earningsTimestamp"), unit="s")
            if dt.date() >= datetime.now().date():
                upcoming.append(("Earnings", dt))
        if info.get("dividendDate"):
            dt = pd.to_datetime(info.get("dividendDate"), unit="s")
            if dt.date() >= datetime.now().date():
                upcoming.append(("Dividend", dt))
    except Exception:
        pass
    if upcoming:
        for ev,dt in upcoming:
            st.info(f"🔔 Upcoming: **{ev}** on **{dt.strftime('%d %b %Y')}**")

    # -------------------- Multi-panel charts (synchronized) --------------------
    rows = 3 if (show_rsi or show_macd) else 1
    row_heights = [0.5, 0.25, 0.25] if rows==3 else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_heights=row_heights)

    # Main price panel
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines+markers", name="Close"), row=1, col=1)

    # overlays
    if show_ema:
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["ema12"], name="EMA12", line=dict(dash="dot", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["ema26"], name="EMA26", line=dict(dash="dash", width=1.5)), row=1, col=1)
    # SMAs - subtle
    fig.add_trace(go.Scatter(x=df["Date"], y=tech["sma20"], name="SMA20", line=dict(dash="dot", width=1), opacity=0.6), row=1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)

    # RSI panel
    if show_rsi:
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["rsi14"], name="RSI(14)", line=dict(color="#ff7f0e")), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1)
        fig.update_yaxes(range=[0,100], row=2, col=1, title_text="RSI")

    # MACD panel
    if show_macd:
        fig.add_trace(go.Bar(x=df["Date"], y=tech["macd_hist"], name="MACD Hist", marker=dict(color=np.where(tech["macd_hist"]>=0,"#2ca02c","#d62728"))), row=rows, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["macd"], name="MACD Line", line=dict(width=1)), row=rows, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["macd_signal"], name="Signal Line", line=dict(width=1, dash="dot")), row=rows, col=1)
        fig.update_yaxes(title_text="MACD", row=rows, col=1)

    fig.update_layout(template="plotly_dark", title=f"{symbol.upper()} Technical View ({st.session_state['last_period']})", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- Technical indicator grid (Trendlyne style cards) --------------------
    if show_technical_dashboard:
        st.subheader("Technical Indicators")
        g1,g2,g3,g4 = st.columns(4)
        g1.metric("RSI(14)", f"{round(tech['rsi14'].iloc[-1],2)}")
        g2.metric("MFI(14)", f"{round(tech['mfi14'].iloc[-1],2)}")
        g3.metric("ADX(14)", f"{round(tech['adx14'].iloc[-1],2)}")
        g4.metric("ATR(14)", f"{round(tech['atr14'].iloc[-1],4)}")

        g5,g6,g7,g8 = st.columns(4)
        g5.metric("MACD", f"{round(tech['macd'].iloc[-1],4)}")
        g6.metric("MACD Signal", f"{round(tech['macd_signal'].iloc[-1],4)}")
        g7.metric("ROC(12)", f"{round(tech['roc12'].iloc[-1],2)}")
        g8.metric("Short/Long EMA", f"{round(tech['ema12'].iloc[-1],2)} / {round(tech['ema26'].iloc[-1],2)}")

        # quick textual interpretations
        st.markdown("### Quick Interpretations")
        interpret = []
        rsi_val = tech['rsi14'].iloc[-1]
        if rsi_val < 30:
            interpret.append("🟢 RSI indicates oversold — possible rebound.")
        elif rsi_val > 70:
            interpret.append("🔴 RSI indicates overbought — watch for correction.")
        else:
            interpret.append("⚪ RSI neutral.")

        if tech['macd'].iloc[-1] > tech['macd_signal'].iloc[-1]:
            interpret.append("🟢 MACD above signal — bullish momentum.")
        else:
            interpret.append("🔴 MACD below signal — bearish momentum.")

        if tech['adx14'].iloc[-1] > 25:
            interpret.append("🟡 ADX indicates a strong trend.")
        else:
            interpret.append("⚪ ADX indicates weak/sideways trend.")

        st.write(" • ".join(interpret))

    # end render_stock

def render_trends():
    st.header("💻 Tech & Startup Trends")
    lang = st.text_input("Language / Topic:", "python")
    if st.button("Fetch Trending Repos"):
        repos = fetch_github_trending(lang)
        if not repos: st.warning("No trending repos found")
        for r in repos[:8]:
            st.markdown(f"**[{r['name']}]({'https://github.com/'+r['name']})**  ⭐ {r['stars']}")
            st.caption(r['description'] or "_No description_")
            st.divider()

def render_research():
    st.header("📚 Research & Learning")
    topic = st.text_input("Search topic:", "machine learning")
    if st.button("Fetch Papers"):
        papers = fetch_arxiv_papers(topic, max_results=5)
        for p in papers:
            st.subheader(p['title'])
            st.caption(", ".join(p['authors']))
            st.write(p['summary'][:600] + "...")
            st.markdown(f"[Read full paper]({p['link']})")
            st.divider()
    st.subheader("Recommended Courses")
    for link in recommend_learning_resources(topic).get("courses", []):
        st.markdown(f"- [{link}]({link})")

def render_skills():
    st.header("🔍 Skill & Job Trends")
    skills = st.text_input("Skills (comma-separated):","Python, SQL")
    if st.button("Analyze"):
        keys = [k.strip() for k in skills.split(",")]
        trends = get_trends_keywords(keys)
        df = pd.DataFrame([{"Skill":k,"Change (%)":v["pct_change"]} for k,v in trends.items()])
        fig = go.Figure([go.Bar(x=df['Skill'], y=df['Change (%)'])])
        st.plotly_chart(fig, use_container_width=True)

def render_news():
    st.header("📰 News & Sentiment")
    q = st.text_input("Topic:", "Indian Stock Market")
    if st.button("Get News"):
        arts = get_news(q, max_items=6)
        sent = analyze_headlines_sentiment(arts)
        for a in sent:
            c1,c2 = st.columns([4,1])
            with c1:
                st.markdown(f"**{a['title']}**")
                st.caption(a['published'].strftime("%b %d, %Y %H:%M"))
            with c2:
                lab = a['sentiment']['label']
                sc = a['sentiment']['score']
                color = "🟢" if "POS" in lab else "🔴" if "NEG" in lab else "⚪"
                st.markdown(f"**{color} {lab} ({sc:.2f})**")
            st.divider()

def render_feedback():
    st.header("💬 Feedback")
    name = st.text_input("Name")
    rating = st.slider("Rate (1-5):",1,5,4)
    comments = st.text_area("Comments")
    if st.button("Submit"):
        row = {"Name":name,"Rating":rating,"Comments":comments,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            df = pd.read_csv("feedback.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Name","Rating","Comments","Timestamp"])
        df = pd.concat([df,pd.DataFrame([row])], ignore_index=True)
        df.to_csv("feedback.csv", index=False)
        st.success("Thanks for the feedback!")

# -------------------- Main renderer --------------------
def render_dashboard():
    navbar()
    page = st.session_state.get("nav","home")
    if page=="home":
        render_home()
    elif page=="stock":
        render_stock()
    elif page=="trends":
        render_trends()
    elif page=="research":
        render_research()
    elif page=="skills":
        render_skills()
    elif page=="news":
        render_news()
    elif page=="feedback":
        render_feedback()
    else:
        render_home()

# run
if __name__ == "__main__":
    render_dashboard()
