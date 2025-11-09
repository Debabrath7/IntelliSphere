# ==============================================
# IntelliSphere Frontend UI (Dark Neon Revamp)
# Author: Debabrath
# ==============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import backend_modules as bm
from backend_modules import (
    fetch_github_trending,
    fetch_arxiv_papers,
    get_trends_keywords,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide")

# -----------------------------------------------------
# GLOBAL STYLE
# -----------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #060c1f, #0b163a, #07132c, #091128);
    background-size: 400% 400%;
    animation: gradientBG 22s ease infinite;
    color: #eafcff;
    font-family: 'Inter', sans-serif;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
h1, h2, h3 {
    color: #00e6ff !important;
    text-shadow: 0 0 12px rgba(0, 230, 255, 0.7);
}
div.stButton > button {
    background: linear-gradient(90deg, #00e6ff, #7a00ff);
    color: white;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 230, 255, 0.6);
    transform: scale(1.05);
}
.navbar {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 24px;
}
.navbar button {
    background-color: #121a2b;
    border: 1px solid #00e6ff;
    color: #00e6ff;
    border-radius: 999px;
    padding: 8px 18px;
    cursor: pointer;
    font-weight: 500;
    transition: 0.25s;
}
.navbar button:hover {
    background-color: #00e6ff;
    color: #020a13;
    transform: scale(1.05);
}
.navbar button.active {
    background-color: #00e6ff;
    color: #020a13;
    box-shadow: 0 0 15px rgba(0, 230, 255, 0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# NAVIGATION STATE
# -----------------------------------------------------
if "nav" not in st.session_state:
    st.session_state["nav"] = "home"

def set_nav(page):
    st.session_state["nav"] = page

# -----------------------------------------------------
# NAVBAR
# -----------------------------------------------------
def navbar():
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    buttons = [
        ("Home", "home"),
        ("Stocks", "stock"),
        ("Trends", "trends"),
        ("Research", "research"),
        ("Skills", "skills"),
        ("News", "news"),
        ("Feedback", "feedback")
    ]
    cols = st.columns(len(buttons))
    for i, (label, key) in enumerate(buttons):
        if cols[i].button(label, key=f"btn_{key}"):
            set_nav(key)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def humanize(num):
    try:
        num = float(num)
        if num >= 1e12: return f"{num/1e12:.2f} Tn"
        if num >= 1e7:  return f"{num/1e7:.2f} Cr"
        if num >= 1e5:  return f"{num/1e5:.2f} L"
        if num >= 1e3:  return f"{num/1e3:.2f} K"
        return str(round(num, 2))
    except: return "N/A"

def safe_dividend_format(dy):
    try:
        if dy is None or dy == 0: return "N/A"
        return f"{round(dy*100, 2)}%" if dy < 1 else f"{round(dy, 2)}%"
    except:
        return "N/A"

def compute_ema(series, span): return series.ewm(span=span, adjust=False).mean()
def compute_rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    ma_up, ma_down = up.ewm(alpha=1/period, adjust=False).mean(), down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

# -----------------------------------------------------
# SECTIONS
# -----------------------------------------------------
def render_home():
    st.title("IntelliSphere: AI-Powered Insight Platform")
    st.markdown("""
        Welcome to **IntelliSphere**, your unified AI dashboard for real-time stock insights,
        tech trends, research breakthroughs, and sentiment analysis — all in one powerful, neon-inspired hub.
    """)
    st.image("https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.success("All systems operational!")

# -----------------------------------------------------
# STOCKS MODULE
# -----------------------------------------------------
def render_stock():
    st.header("Stock Insights")

    symbol = st.text_input("Enter company symbol:", "TCS")
    period = st.selectbox("Select time range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
    chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True)
    show_ema = st.checkbox("Show EMA (12 & 26)", value=True)
    show_rsi = st.checkbox("Show RSI (14)", value=False)

    if st.button("Fetch Stock Data"):
        ticker_raw = symbol.strip().upper()
        tickers = [ticker_raw, ticker_raw + ".NS"]
        df, info = None, None

        for t in tickers:
            try:
                df = bm.get_stock_data(t, period=period)
                if df is not None and not df.empty:
                    info = bm.yf.Ticker(t).info
                    ticker = t
                    break
            except:
                continue

        if df is None or df.empty:
            st.error("⚠️ No data found for this symbol.")
            return

        # Format DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index().rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])

        # Metrics
        latest, first = df["Close"].iloc[-1], df["Close"].iloc[0]
        change = round((latest - first) / first * 100, 2)
        pe, mc, vol = info.get("trailingPE", "N/A"), humanize(info.get("marketCap", 0)), humanize(info.get("volume", 0))
        div_yield = safe_dividend_format(info.get("dividendYield"))

        # Insight Card
        st.success(f"📈 **{ticker_raw}** moved **{change}%** over selected period. Current price ₹{round(latest,2)}")

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E Ratio", pe)
        c2.metric("Market Cap", mc)
        c3.metric("Volume", vol)

        c4, c5, c6 = st.columns(3)
        c4.metric("52W High", info.get("fiftyTwoWeekHigh", "N/A"))
        c5.metric("52W Low", info.get("fiftyTwoWeekLow", "N/A"))
        c6.metric("Dividend Yield", div_yield)

        # Chart with optional RSI
        fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25] if show_rsi else [1])

        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines+markers", name="Close"), row=1, col=1)
            if show_ema:
                fig.add_trace(go.Scatter(x=df["Date"], y=compute_ema(df["Close"], 12), name="EMA 12", line=dict(dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["Date"], y=compute_ema(df["Close"], 26), name="EMA 26", line=dict(dash="dash")), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candlestick"), row=1, col=1)

        if show_rsi:
            rsi = compute_rsi(df["Close"])
            fig.add_trace(go.Scatter(x=df["Date"], y=rsi, mode="lines", name="RSI(14)", line=dict(color="#ff7f0e")), row=2, col=1)
            fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
            fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)

        fig.update_layout(template="plotly_dark", title=f"{ticker_raw} ({period})", xaxis_title="Date", yaxis_title="Price (₹)", hovermode="x unified", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        # Company events
        try:
            events = []
            if info.get("earningsTimestamp"):
                events.append(f"📢 Earnings on **{pd.to_datetime(info['earningsTimestamp'], unit='s').strftime('%d %b %Y')}**")
            if info.get("dividendDate"):
                events.append(f"💰 Dividend payout expected **{pd.to_datetime(info['dividendDate'], unit='s').strftime('%d %b %Y')}**")
            if events: st.info("\n\n".join(events))
        except: st.caption("📅 No upcoming events found.")

# -----------------------------------------------------
# TRENDS MODULE
# -----------------------------------------------------
def render_trends():
    st.header("Tech & Startup Trends")
    lang = st.text_input("Enter a programming language or topic:", "python")
    if st.button("Fetch GitHub Trends"):
        repos = fetch_github_trending(lang)
        if not repos: st.warning("No trending repositories right now.")
        for r in repos[:8]:
            st.markdown(f"**[{r['name']}]({'https://github.com/' + r['name']})** {r['stars']}")
            st.caption(r['description'] or "_No description available_")
            st.divider()

# -----------------------------------------------------
# RESEARCH MODULE
# -----------------------------------------------------
def render_research():
    st.header("Research & Learning")
    topic = st.text_input("Enter research topic:", "machine learning")
    if st.button("Find Research Papers"):
        papers = fetch_arxiv_papers(topic, max_results=5)
        for p in papers:
            st.subheader(p["title"])
            st.caption(", ".join(p["authors"]))
            st.write(p["summary"][:600] + "...")
            st.markdown(f"[🔗 Read Full Paper]({p['link']})")
            st.divider()
    st.subheader("🎓 Recommended Courses")
    for link in recommend_learning_resources(topic)["courses"]:
        st.markdown(f"- [{link}]({link})")

# -----------------------------------------------------
# SKILLS MODULE
# -----------------------------------------------------
def render_skills():
    st.header("Skill & Job Trends")
    skills = st.text_input("Enter skills (comma-separated):", "Python, SQL, ML")
    if st.button("Analyze Popularity"):
        keys = [k.strip() for k in skills.split(",")]
        trends = get_trends_keywords(keys)
        df = pd.DataFrame([{"Skill": k, "Change (%)": v["pct_change"]} for k, v in trends.items()])
        fig = px.bar(df, x="Skill", y="Change (%)", color="Skill", title="Skill Popularity (Last 3 Months)")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# NEWS MODULE
# -----------------------------------------------------
def render_news():
    st.header("News & Sentiment")
    query = st.text_input("Enter topic:", "Indian Stock Market")
    if st.button("Get Latest News"):
        articles = get_news(query, max_items=6)
        sentiments = analyze_headlines_sentiment(articles)
        for art in sentiments:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{art['title']}**")
                st.caption(f"{art['published'].strftime('%b %d, %Y %H:%M')}")
            with col2:
                label = art['sentiment']['label']
                score = art['sentiment']['score']
                color = "🟢" if "POS" in label else "🔴" if "NEG" in label else "⚪"
                st.markdown(f"**{color} {label} ({score:.2f})**")
            st.divider()

# -----------------------------------------------------
# FEEDBACK MODULE
# -----------------------------------------------------
def render_feedback():
    st.header("Feedback")
    name = st.text_input("Your Name")
    rating = st.slider("Rate IntelliSphere (1-5):", 1, 5, 4)
    comments = st.text_area("Your Feedback")
    if st.button("Submit"):
        entry = {"Name": name, "Rating": rating, "Comments": comments, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            df = pd.read_csv("feedback.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Name", "Rating", "Comments", "Timestamp"])
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv("feedback.csv", index=False)
        st.success("Feedback submitted successfully!")

# -----------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------
def render_dashboard():
    navbar()
    page = st.session_state["nav"]

    if page == "home": render_home()
    elif page == "stock": render_stock()
    elif page == "trends": render_trends()
    elif page == "research": render_research()
    elif page == "skills": render_skills()
    elif page == "news": render_news()
    elif page == "feedback": render_feedback()
