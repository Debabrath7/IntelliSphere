# ==============================================
# IntelliSphere Frontend — polished Trendlyne-inspired UI
# Author: Debabrath (refactor)
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import backend_modules as bm
from backend_modules import (
    fetch_github_trending,
    fetch_arxiv_papers,
    get_trends_keywords,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)

# -------------------- Page config & safe session init --------------------
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide")

def _init_state():
    if "nav" not in st.session_state: st.session_state["nav"] = "home"
    if "last_symbol" not in st.session_state: st.session_state["last_symbol"] = None
    if "last_df" not in st.session_state: st.session_state["last_df"] = None
    if "last_info" not in st.session_state: st.session_state["last_info"] = None
    if "last_period" not in st.session_state: st.session_state["last_period"] = None
    if "tech_cache" not in st.session_state: st.session_state["tech_cache"] = {}
_init_state()

# -------------------- Styling: dark neon + boxed cards --------------------
st.markdown("""
<style>
/* background and subtle animated gradient */
.stApp {
  background: linear-gradient(120deg,#06111a 0%, #071827 35%, #031022 100%);
  color: #eafcff;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
/* header */
h1,h2,h3 { color: #00e6ff !important; text-shadow:0 0 8px rgba(0,230,255,0.12) }
/* nav */
.navbar{display:flex; gap:12px; justify-content:center; margin:14px 0;}
.navbtn{background:#0f1720; border:1px solid rgba(0,230,255,0.18); color:#8fefff; padding:8px 14px; border-radius:999px; font-weight:600}
.navbtn.active{background:linear-gradient(90deg,#00e6ff,#7a00ff); color:#02111a; box-shadow:0 10px 30px rgba(0,230,255,0.06)}
/* metric boxes */
.metric-box{background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.008)); border:1px solid rgba(255,255,255,0.03); border-radius:12px; padding:14px;}
.metric-title{color:#9feafc; font-size:13px}
.metric-value{color:#ffffff; font-size:24px; font-weight:700}
.small-muted{color:#9fbfcf; font-size:13px}
/* info box */
.upcoming{background:#0f3b3b; padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.02)}
/* responsive */
@media (max-width: 900px) {
  .navbar { flex-wrap:wrap; }
}
</style>
""", unsafe_allow_html=True)

# -------------------- Utilities & indicators --------------------
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
    try:
        div_rate = info.get("dividendRate")
        if div_rate and current_price and current_price>0:
            return round((float(div_rate)/float(current_price))*100, 2)
        raw = info.get("dividendYield")
        if raw is None:
            return "N/A"
        v = float(raw)
        if v > 10:  # likely percent expressed as whole number
            v = v / 100.0
        return round(v * 100, 2)
    except:
        return "N/A"

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(span=period, adjust=False).mean()
    ma_down = down.ewm(span=period, adjust=False).mean()
    rs = ma_up/(ma_down + 1e-9)
    rsi = 100 - (100/(1+rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

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
            posmf.append(money_flow.iloc[i]); negmf.append(0)
        else:
            posmf.append(0); negmf.append(money_flow.iloc[i])
    posmf = pd.Series([0]+posmf); negmf = pd.Series([0]+negmf)
    mf_ratio = posmf.rolling(window=period, min_periods=1).sum() / (negmf.rolling(window=period, min_periods=1).sum()+1e-9)
    mfi = 100 - (100 / (1 + mf_ratio))
    return mfi

def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    up_move = high.diff(); down_move = -low.diff()
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
    return series.pct_change(periods=period) * 100

# -------------------- Layout: navbar --------------------
def navbar():
    items = [("Home","home"),("Stocks","stock"),("Trends","trends"),("Research","research"),("Skills","skills"),("News","news"),("Feedback","feedback")]
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    cols = st.columns(len(items))
    for i,(label,key) in enumerate(items):
        is_active = (st.session_state["nav"] == key)
        if cols[i].button(label, key=f"nav_{key}"):
            st.session_state["nav"] = key
        # attempt to style active visually by writing an extra small HTML marker
        if is_active:
            cols[i].markdown(f"<div style='text-align:center;margin-top:4px'><small style='color:#00e6ff'>●</small></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Stock renderer (clean boxed layout) --------------------
def render_stock():
    st.header("💹 Stock Insights")
    # inputs
    symbol = st.text_input("Enter company symbol:", st.session_state.get("last_symbol") or "TCS")
    period = st.selectbox("Select time range:", ["1d","5d","1mo","3mo","6mo","1y"], index=2)
    chart_type = st.radio("Chart Type", ["Candlestick","Line"], index=0, horizontal=True)
    show_ema = st.checkbox("Show EMA (12 & 26)", value=True)
    show_rsi = st.checkbox("Show RSI (14)", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_full_tech = st.checkbox("Show full technical dashboard", value=True)
    fetch = st.button("Fetch Stock Data")

    need_fetch = fetch or (symbol != st.session_state.get("last_symbol")) or (period != st.session_state.get("last_period"))
    if need_fetch:
        tickers = [symbol.strip().upper(), symbol.strip().upper()+".NS"]
        df = None; info = {}
        used_ticker = None
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
            st.error("No data found for this symbol/period. Try another symbol or period.")
            return
        st.session_state["last_symbol"] = symbol
        st.session_state["last_df"] = df
        st.session_state["last_info"] = info
        st.session_state["last_period"] = period

    df = st.session_state.get("last_df")
    info = st.session_state.get("last_info")
    if df is None or info is None:
        st.info("Enter a symbol and click Fetch Stock Data.")
        return

    # normalize columns (yfinance sometimes returns multiindex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Datetime":"Date","date":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # top metrics
    current_price = float(df["Close"].iloc[-1])
    first_price = float(df["Close"].iloc[0])
    change_pct = round((current_price - first_price)/(first_price+1e-9)*100, 2)
    pe = info.get("trailingPE","N/A")
    market_cap = humanize(info.get("marketCap",0))
    vol = humanize(info.get("volume",0))
    fiftyTwoHigh = info.get("fiftyTwoWeekHigh","N/A")
    fiftyTwoLow = info.get("fiftyTwoWeekLow","N/A")
    day_high = info.get("dayHigh", df["High"].iloc[-1] if "High" in df.columns else "N/A")
    day_low = info.get("dayLow", df["Low"].iloc[-1] if "Low" in df.columns else "N/A")
    div_yield = dividend_yield_from_info(info, current_price)

    # top overview boxes (boxed and shaded)
    topcols = st.columns([2,2,2,2])
    with topcols[0]:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Current Price</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>₹{round(current_price,2)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{change_pct}% since period start</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with topcols[1]:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>P/E Ratio</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{pe}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>52W High: {fiftyTwoHigh}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with topcols[2]:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Market Cap</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{market_cap}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>52W Low: {fiftyTwoLow}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with topcols[3]:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Volume</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{vol}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Dividend Yield: {div_yield}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # compute and cache technicals
    cache_key = f"{st.session_state['last_symbol']}_{st.session_state['last_period']}"
    if cache_key not in st.session_state["tech_cache"]:
        tech = {}
        tech["ema12"] = compute_ema(df["Close"], 12)
        tech["ema26"] = compute_ema(df["Close"], 26)
        tech["rsi14"] = compute_rsi(df["Close"], 14)
        tech["macd"], tech["macd_sig"], tech["macd_hist"] = compute_macd(df["Close"])
        tech["atr14"] = compute_atr(df, 14)
        tech["mfi14"] = compute_mfi(df, 14)
        tech["adx14"] = compute_adx(df, 14)
        tech["roc12"] = compute_roc(df["Close"], 12)
        tech["sma20"] = df["Close"].rolling(20).mean()
        tech["sma50"] = df["Close"].rolling(50).mean()
        tech["sma200"] = df["Close"].rolling(200).mean()
        st.session_state["tech_cache"][cache_key] = tech
    else:
        tech = st.session_state["tech_cache"][cache_key]

    # technical summary / quick interpretation similar to Trendlyne (compact)
    latest_rsi = float(tech["rsi14"].iloc[-1]) if not tech["rsi14"].empty else None
    macd_latest = tech["macd"].iloc[-1] if not tech["macd"].empty else None
    macd_sig = tech["macd_sig"].iloc[-1] if not tech["macd_sig"].empty else None
    ema12_v = tech["ema12"].iloc[-1] if not tech["ema12"].empty else None
    ema26_v = tech["ema26"].iloc[-1] if not tech["ema26"].empty else None

    summary_lines = []
    if latest_rsi is not None:
        if latest_rsi < 30: summary_lines.append("RSI indicates Oversold (possible rebound).")
        elif latest_rsi > 70: summary_lines.append("RSI indicates Overbought (watch for pullback).")
        else: summary_lines.append("RSI in neutral zone.")
    if macd_latest is not None and macd_sig is not None:
        if macd_latest > macd_sig: summary_lines.append("MACD above signal — bullish momentum.")
        else: summary_lines.append("MACD below signal — bearish momentum.")
    if ema12_v and ema26_v:
        summary_lines.append("Short EMA above Long EMA" if ema12_v > ema26_v else "Short EMA below Long EMA")

    st.markdown("<div style='margin-top:12px; margin-bottom:8px' class='small-muted'>Technical Summary: " + " • ".join(summary_lines) + "</div>", unsafe_allow_html=True)

    # upcoming events only (future)
    upcoming = []
    try:
        # earningsTimestampStart or earningsTimestamp (seconds)
        if info.get("earningsTimestampStart"):
            dt = pd.to_datetime(info.get("earningsTimestampStart"), unit="s")
            if dt.date() >= datetime.now().date(): upcoming.append(("Earnings", dt))
        elif info.get("earningsTimestamp"):
            dt = pd.to_datetime(info.get("earningsTimestamp"), unit="s")
            if dt.date() >= datetime.now().date(): upcoming.append(("Earnings", dt))
        if info.get("dividendDate"):
            dt = pd.to_datetime(info.get("dividendDate"), unit="s")
            if dt.date() >= datetime.now().date(): upcoming.append(("Dividend", dt))
    except Exception:
        pass
    if upcoming:
        for ev, dt in upcoming:
            st.markdown(f"<div class='upcoming'>🔔 Upcoming: <b>{ev}</b> on <b>{dt.strftime('%d %b %Y')}</b></div>", unsafe_allow_html=True)

    st.divider()

    # ---------------- Chart area (multi-panel)
    rows = 3 if (show_rsi or show_macd) else 1
    row_heights = [0.55, 0.225, 0.225] if rows==3 else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)

    # main price panel
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines+markers", name="Close"), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["ema12"], name="EMA12", line=dict(color="#00e6ff", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["ema26"], name="EMA26", line=dict(color="#7a00ff", dash="dash")), row=1, col=1)
    # subtle SMA 20
    fig.add_trace(go.Scatter(x=df["Date"], y=tech["sma20"], name="SMA20", line=dict(color="#9fbfcf", dash="dot"), opacity=0.5), row=1, col=1)

    # RSI panel
    if show_rsi:
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["rsi14"], name="RSI(14)", line=dict(color="#ff7f0e")), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1)
        fig.update_yaxes(range=[0,100], row=2, col=1)

    # MACD panel
    if show_macd:
        fig.add_trace(go.Bar(x=df["Date"], y=tech["macd_hist"], name="MACD Hist", marker_color=np.where(tech["macd_hist"]>=0,"#2ca02c","#d62728")), row=rows, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["macd"], name="MACD", line=dict(width=1)), row=rows, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=tech["macd_sig"], name="Signal", line=dict(width=1, dash="dot")), row=rows, col=1)

    fig.update_layout(template="plotly_dark", title=f"IntelliSphere — {symbol.upper()} Technical View ({st.session_state['last_period']})", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # technical indicator grid (boxed)
    if show_full_tech:
        st.subheader("Technical Indicators")
        a,b,c,d = st.columns(4)
        a.markdown("<div class='metric-box'><div class='metric-title'>RSI (14)</div><div class='metric-value'>" + (f"{round(tech['rsi14'].iloc[-1],2)}" if not tech['rsi14'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)
        b.markdown("<div class='metric-box'><div class='metric-title'>MFI (14)</div><div class='metric-value'>" + (f"{round(tech['mfi14'].iloc[-1],2)}" if not tech['mfi14'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)
        c.markdown("<div class='metric-box'><div class='metric-title'>ADX (14)</div><div class='metric-value'>" + (f"{round(tech['adx14'].iloc[-1],2)}" if not tech['adx14'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)
        d.markdown("<div class='metric-box'><div class='metric-title'>ATR (14)</div><div class='metric-value'>" + (f"{round(tech['atr14'].iloc[-1],4)}" if not tech['atr14'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)

        e,f,g,h = st.columns(4)
        e.markdown("<div class='metric-box'><div class='metric-title'>MACD</div><div class='metric-value'>" + (f"{round(tech['macd'].iloc[-1],4)}" if not tech['macd'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)
        f.markdown("<div class='metric-box'><div class='metric-title'>MACD Signal</div><div class='metric-value'>" + (f"{round(tech['macd_sig'].iloc[-1],4)}" if not tech['macd_sig'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)
        g.markdown("<div class='metric-box'><div class='metric-title'>ROC (12)</div><div class='metric-value'>" + (f"{round(tech['roc12'].iloc[-1],2)}" if not tech['roc12'].empty else "N/A") + "</div></div>", unsafe_allow_html=True)
        g.markdown("", unsafe_allow_html=True)
        st.markdown("**Quick Interpretation:** " + " • ".join(summary_lines), unsafe_allow_html=True)

# -------------------- Trends / Research / Skills / News (safe) --------------------
def render_trends():
    st.header("💻 Tech & Startup Trends")
    lang = st.text_input("Language / Topic (e.g., python):", "python")
    if st.button("Fetch Trending Repos"):
        try:
            repos = fetch_github_trending(lang)
            if not repos:
                st.warning("No trending repositories found now.")
            for r in repos[:10]:
                st.markdown(f"**[{r['name']}]({'https://github.com/'+r['name']})** — {r.get('stars','0')}")
                st.caption(r.get('description',''))
                st.divider()
        except Exception as e:
            st.error("Could not fetch GitHub trending. " + str(e))
    st.divider()
    st.subheader("Startup News (sentiment)")
    try:
        startup_news = get_news("startup OR funding OR venture capital India", max_items=6)
        startup_sent = analyze_headlines_sentiment(startup_news)
        for n in startup_sent:
            st.markdown(f"**{n['title']}**")
            if n.get('link'): st.markdown(f"[Read more]({n['link']})")
            st.caption(f"Sentiment: {n['sentiment']['label']} ({n['sentiment']['score']:.2f})")
            st.divider()
    except Exception:
        st.warning("Startup news temporarily unavailable.")

def render_research():
    st.header("📚 Research & Education")
    topic = st.text_input("Search topic:", "machine learning")
    if st.button("Fetch Papers"):
        try:
            papers = fetch_arxiv_papers(topic, max_results=5)
            if not papers: st.warning("No papers found.")
            for p in papers:
                st.subheader(p['title'])
                st.caption(", ".join(p.get('authors',[])))
                st.write(p.get('summary','')[:700] + "...")
                st.markdown(f"[Read full paper]({p['link']})")
                st.divider()
        except Exception as e:
            st.error("Could not fetch arXiv results. " + str(e))
    st.subheader("Recommended Courses")
    try:
        recs = recommend_learning_resources(topic)
        for c in recs.get("courses", []):
            st.markdown(f"- [{c}]({c})")
    except Exception:
        st.warning("Course recommendations currently unavailable.")

def render_skills():
    st.header("🔍 Skill & Job Trends")
    skills = st.text_input("Enter skills (comma-separated):", "Python, Java, SQL")
    if st.button("Analyze"):
        keys = [k.strip() for k in skills.split(",") if k.strip()]
        if not keys:
            st.warning("Enter at least one skill.")
            return
        try:
            trends = get_trends_keywords(keys)
            if not trends:
                st.warning("Google Trends currently unavailable.")
                return
            df = pd.DataFrame([{"Skill": k, "Change (%)": trends[k]["pct_change"]} for k in keys if k in trends])
            fig = go.Figure([go.Bar(x=df['Skill'], y=df['Change (%)'])])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Could not fetch trends. " + str(e))

def render_news():
    st.header("📰 News & Sentiment")
    query = st.text_input("Enter topic / company:", "Indian Stock Market")
    num_items = st.slider("Number of articles:", 3, 15, 6)
    if st.button("Get News"):
        try:
            articles = get_news(query, max_items=num_items)
            sentiments = analyze_headlines_sentiment(articles)
            if not sentiments:
                st.warning("No articles found.")
            else:
                for art in sentiments:
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.markdown(f"**{art['title']}**")
                        if art.get('link'): st.markdown(f"[Read more]({art['link']})")
                        st.caption(f"{art['published'].strftime('%b %d, %Y %H:%M')}")
                    with col2:
                        lab = art['sentiment']['label']; sc = art['sentiment']['score']
                        badge = "🟢" if "POS" in lab else "🔴" if "NEG" in lab else "⚪"
                        st.markdown(f"**{badge} {lab}**\n\n{sc:.2f}")
                    st.divider()
        except Exception as e:
            st.error("Could not fetch news. " + str(e))

def render_feedback():
    st.header("💬 Feedback")
    name = st.text_input("Name")
    rating = st.slider("Rate IntelliSphere (1-5):",1,5,4)
    comments = st.text_area("Comments")
    if st.button("Submit Feedback"):
        try:
            new_entry = {"Name":name,"Rating":rating,"Comments":comments,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            try:
                df = pd.read_csv("feedback.csv")
            except FileNotFoundError:
                df = pd.DataFrame(columns=["Name","Rating","Comments","Timestamp"])
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv("feedback.csv", index=False)
            st.success("Thanks — feedback submitted!")
        except Exception as e:
            st.error("Could not save feedback. " + str(e))

# -------------------- Main router --------------------
def render_dashboard():
    navbar()
    page = st.session_state.get("nav","home")
    if page == "home":
        st.title("IntelliSphere: AI-Powered Insight Platform")
        st.write("Welcome — use the Stocks tab for a deep technical view. UI is styled to be compact, readable and exam-worthy.")
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
        st.write("Unknown page — showing home.")
        st.title("IntelliSphere")

if __name__ == "__main__":
    render_dashboard()
