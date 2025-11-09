# ==============================================
# IntelliSphere Frontend UI (Dark Neon Revamp)
# Author: Debabrath
# ==============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from backend_modules import (
    fetch_github_trending,
    fetch_arxiv_papers,
    get_trends_keywords,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)
import backend_modules as bm

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide")

# -----------------------------------------------------
# STYLING (Neon Gradient + Animated Background)
# -----------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #050c1f, #091533, #0e1c43, #091128);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
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
    text-shadow: 0 0 10px rgba(0, 230, 255, 0.7);
}
div.stButton > button {
    background: linear-gradient(90deg, #00e6ff, #7a00ff);
    color: white;
    border-radius: 10px;
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
# NAVBAR COMPONENT
# -----------------------------------------------------
def navbar():
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    nav_buttons = [
        ("🏠 Home", "home"),
        ("💹 Stocks", "stock"),
        ("💻 Trends", "trends"),
        ("📚 Research", "research"),
        ("🔍 Skills", "skills"),
        ("📰 News", "news"),
        ("💬 Feedback", "feedback")
    ]
    cols = st.columns(len(nav_buttons))
    for i, (label, key) in enumerate(nav_buttons):
        is_active = st.session_state["nav"] == key
        if cols[i].button(label, key=f"btn_{key}"):
            set_nav(key)
        if is_active:
            cols[i].markdown(
                f"<style>div[data-testid='stButton'] button#btn_{key}{{background-color:#00e6ff;color:#020a13;}}</style>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# SECTIONS
# -----------------------------------------------------
def render_home():
    st.title("🌐 IntelliSphere: AI-Powered Insight Platform")
    st.markdown("""
        Welcome to **IntelliSphere**, your unified AI dashboard for real-time stock insights,
        startup trends, research breakthroughs, and market sentiment —  
        wrapped in a stunning Dark Neon theme.
    """)
    st.image("https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.success("✅ All systems operational!")

def render_stock():
    st.header("💹 Stock Insights")

    symbol = st.text_input("Enter company symbol:", "TCS")
    period = st.selectbox(
        "Select time range:",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2
    )

    chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True)

    if st.button("Fetch Stock Data"):
        ticker_raw = symbol.strip().upper()

        # Try both plain & .NS tickers
        possible_tickers = [ticker_raw, ticker_raw + ".NS"]
        df, info, ticker = None, None, None

        for t in possible_tickers:
            try:
                df = bm.get_stock_data(t, period=period)
                if df is not None and not df.empty:
                    info = bm.yf.Ticker(t).info
                    ticker = t
                    break
            except Exception:
                continue

        if df is None or df.empty:
            st.error("⚠️ No data found for this company.")
            return

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index().rename(columns={"Datetime": "Date", "date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])

        # -------- Metric Calculations --------
        latest_price = round(df["Close"].iloc[-1], 2)
        first_price = round(df["Close"].iloc[0], 2)
        change = round(((latest_price - first_price) / first_price) * 100, 2)

        # Helper to make big numbers readable
        def humanize(num):
            try:
                num = float(num)
                if num >= 1e12: return f"{num/1e12:.2f} Tn"
                if num >= 1e7:  return f"{num/1e7:.2f} Cr"
                if num >= 1e5:  return f"{num/1e5:.2f} L"
                if num >= 1e3:  return f"{num/1e3:.2f} K"
                return str(round(num, 2))
            except: return "N/A"

        pe_ratio = info.get("trailingPE", "N/A")
        market_cap = humanize(info.get("marketCap", 0))
        volume = humanize(info.get("volume", 0))
        high_52w = info.get("fiftyTwoWeekHigh", "N/A")
        low_52w = info.get("fiftyTwoWeekLow", "N/A")
        day_high = info.get("dayHigh", "N/A")
        day_low = info.get("dayLow", "N/A")
        dividend_yield = (
            str(round(info.get("dividendYield", 0) * 100, 2)) + "%"
            if info.get("dividendYield") else "N/A"
        )

        # -------- Insight Card --------
        insight_msg = (
            f"📈 **{ticker_raw}** has moved **{change}%** "
            f"over the selected period. Current price: ₹ {latest_price}."
        )
        st.success(insight_msg)

        # -------- Metrics Grid --------
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E Ratio", pe_ratio)
        c2.metric("Market Cap", market_cap)
        c3.metric("Volume", volume)

        c4, c5, c6 = st.columns(3)
        c4.metric("52 W High", high_52w)
        c5.metric("52 W Low", low_52w)
        c6.metric("Dividend Yield", dividend_yield)

        c7, c8, c9 = st.columns(3)
        c7.metric("Today's High", day_high)
        c8.metric("Today's Low", day_low)
        c9.metric("Price Change %", f"{change}%")

        # -------- Chart Section --------
        if chart_type == "Line":
            fig = px.line(
                df, x="Date", y="Close",
                title=f"{ticker_raw} Price Movement ({period})",
                markers=True
            )
            fig.update_traces(line_shape="spline")
        else:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(
                x=df["Date"],
                open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"]
            )])
            fig.update_layout(title=f"{ticker_raw} Candlestick Chart ({period})")

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date", yaxis_title="Price (₹)",
            title_x=0.5, margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------- Company Events --------
        event_info = []
        try:
            earn_ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            if earn_ts:
                date_obj = pd.to_datetime(earn_ts, unit="s")
                event_info.append(f"📢 Earnings announcement on **{date_obj.strftime('%d %b %Y')}**")

            div_ts = info.get("dividendDate")
            if div_ts:
                date_obj = pd.to_datetime(div_ts, unit="s")
                event_info.append(f"💰 Dividend payout expected around **{date_obj.strftime('%d %b %Y')}**")
        except Exception:
            pass

        if event_info:
            st.info("\n\n".join(event_info))
        else:
            st.caption("📅 No upcoming results or dividends reported.")

def render_trends():
    st.header("💻 Tech & Startup Trends")
    lang = st.text_input("Enter a programming language or topic:", "python")
    if st.button("Fetch GitHub Trends"):
        repos = fetch_github_trending(lang)
        if not repos:
            st.warning("No trending repositories right now.")
        for r in repos[:8]:
            st.markdown(f"**[{r['name']}]({'https://github.com/' + r['name']})**  ⭐ {r['stars']}")
            st.caption(r['description'] or "_No description_")
            st.divider()

def render_research():
    st.header("📚 Research & Learning")
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
    recs = recommend_learning_resources(topic)
    for link in recs["courses"]:
        st.markdown(f"- [{link}]({link})")

def render_skills():
    st.header("🔍 Skill & Job Trends")
    skills = st.text_input("Enter skills (comma-separated):", "Python, Java, SQL")
    if st.button("Analyze Popularity"):
        keys = [k.strip() for k in skills.split(",")]
        trends = get_trends_keywords(keys)
        df = pd.DataFrame([{"Skill": k, "Change (%)": v["pct_change"]} for k, v in trends.items()])
        fig = px.bar(df, x="Skill", y="Change (%)", color="Skill", title="Skill Popularity (3 Months)")
        st.plotly_chart(fig, use_container_width=True)

def render_news():
    st.header("📰 News & Sentiment")
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

def render_feedback():
    st.header("💬 Feedback")
    name = st.text_input("Your Name")
    rating = st.slider("Rate IntelliSphere (1-5):", 1, 5, 4)
    comments = st.text_area("Your Feedback")
    if st.button("Submit"):
        new_entry = {"Name": name, "Rating": rating, "Comments": comments, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            df = pd.read_csv("feedback.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Name", "Rating", "Comments", "Timestamp"])
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv("feedback.csv", index=False)
        st.success("✅ Thank you for your feedback!")

# -----------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------
def render_dashboard():
    navbar()
    page = st.session_state["nav"]

    if page == "home":
        render_home()
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
