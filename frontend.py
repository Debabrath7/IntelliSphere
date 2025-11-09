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
    period = st.selectbox("Select time range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
    if st.button("Fetch Stock Data"):
        ticker = symbol.strip().upper()
        if "." not in ticker:
            ticker += ".NS"
        df = bm.get_stock_data(ticker, period=period)
        if df is None:
            st.error("⚠️ No data available for this stock.")
            return
        info = bm.yf.Ticker(ticker).info
        latest, first = round(df["Close"].iloc[-1], 2), round(df["Close"].iloc[0], 2)
        change = round(((latest - first) / first) * 100, 2)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{symbol.upper()}", f"₹{latest}", f"{change}%")
        c2.metric("52W High", info.get("fiftyTwoWeekHigh", "N/A"))
        c3.metric("52W Low", info.get("fiftyTwoWeekLow", "N/A"))
        fig = px.line(df, x="Date", y="Close", title=f"{symbol.upper()} Price Movement", markers=True)
        fig.update_traces(line_shape="spline")
        fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

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
