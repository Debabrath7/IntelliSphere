# ==============================================
# IntelliSphere Frontend UI (Dark Neon Edition)
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
import backend_modules as bm  # use optimized functions directly

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------
# NAV ITEMS
# -----------------------------------------------------
NAV_ITEMS = [
    ("Home", "home"),
    ("Stock Insights", "stock"),
    ("Tech & Startup Trends", "trends"),
    ("Research & Education", "research"),
    ("Skill & Job Trends", "skills"),
    ("News & Sentiment", "news"),
    ("Feedback", "feedback"),
]

# Initialize session state
if "nav" not in st.session_state:
    st.session_state["nav"] = "home"

# -----------------------------------------------------
# CUSTOM STYLING (Dark Neon Theme)
# -----------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a0f29 0%, #04060d 90%);
        color: #eafcff;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #00e6ff !important;
        text-shadow: 0 0 8px rgba(0, 230, 255, 0.5);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00e6ff, #7a00ff);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 12px rgba(0, 230, 255, 0.6);
        transform: scale(1.02);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #121a2b !important;
        color: #eafcff !important;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] { color: #00e6d4; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #9feafc; }
    footer { visibility: hidden; }
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00e6ff, transparent);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------
# NAVIGATION BAR (TOP)
# -----------------------------------------------------
def navbar():
    st.markdown(
        """
        <div style='display:flex;justify-content:center;gap:10px;margin-bottom:20px;flex-wrap:wrap;'>
            <a href='/?page=home' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>🏠 Home</button></a>
            <a href='/?page=stock' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>💹 Stocks</button></a>
            <a href='/?page=trends' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>💻 Trends</button></a>
            <a href='/?page=research' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>📚 Research</button></a>
            <a href='/?page=skills' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>🔍 Skills</button></a>
            <a href='/?page=news' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>📰 News</button></a>
            <a href='/?page=feedback' target='_self'><button style='padding:8px 18px;border-radius:999px;background:#121a2b;color:#00e6ff;border:1px solid #00e6ff;'>💬 Feedback</button></a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------
# SECTIONS
# -----------------------------------------------------
def render_home():
    st.title("🌐 IntelliSphere: AI-Powered Insight Platform")
    st.write("""
        Welcome to **IntelliSphere**, your all-in-one AI dashboard for 
        real-time stock analysis, tech trends, research, and market sentiment — 
        crafted in a sleek Dark Neon theme.  
    """)
    st.image(
        "https://miro.medium.com/v2/resize:fit:1100/format:webp/1*0ZrJ6x8r9KxZxB1ITlmN_Q.png",
        use_container_width=True
    )
    st.success("✅ All systems operational!")

def render_stock():
    st.header("💹 Stock Insights")
    user_input = st.text_input("Enter Company Name or Symbol:", "TCS")
    period = st.selectbox("Select Time Range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)

    if st.button("Get Stock Data"):
        ticker = user_input.strip().upper()
        if "." not in ticker:
            ticker += ".NS"
        df = bm.get_stock_data(ticker, period=period)
        if df is None:
            st.error("⚠️ No data available. Try a different stock.")
        else:
            info = bm.yf.Ticker(ticker).info
            latest = round(df["Close"].iloc[-1], 2)
            first = round(df["Close"].iloc[0], 2)
            change = round(((latest - first) / first) * 100, 2)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{user_input.upper()}", f"₹{latest}", f"{change}%")
            c2.metric("52W High", info.get("fiftyTwoWeekHigh", "N/A"))
            c3.metric("52W Low", info.get("fiftyTwoWeekLow", "N/A"))
            fig = px.line(df, x="Date", y="Close", title=f"{user_input.upper()} Price Trend ({period})", markers=True)
            fig.update_traces(line_shape="spline")
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

def render_trends():
    st.header("💻 Tech & Startup Trends")
    lang = st.text_input("Enter language (e.g., Python, Java):", "python")
    if st.button("Fetch Trending Repositories"):
        repos = fetch_github_trending(lang)
        if repos:
            for r in repos[:10]:
                st.markdown(f"### [{r['name']}]({'https://github.com/' + r['name']}) ⭐ {r['stars']}")
                st.caption(r['description'] or "_No description available_")
                st.divider()
        else:
            st.warning("No trending repositories found right now.")
    st.subheader("🚀 Startup News")
    startup_news = get_news("startup OR funding OR venture capital India", max_items=6)
    startup_sent = analyze_headlines_sentiment(startup_news)
    for n in startup_sent:
        st.markdown(f"**{n['title']}**")
        if n['link']:
            st.markdown(f"[Read more]({n['link']})")
        st.caption(f"Sentiment: {n['sentiment']['label']} ({n['sentiment']['score']:.2f})")
        st.divider()

def render_research():
    st.header("📚 Research & Education")
    topic = st.text_input("Enter topic:", "machine learning")
    if st.button("Fetch Papers"):
        papers = fetch_arxiv_papers(topic, max_results=5)
        for p in papers:
            st.subheader(p["title"])
            st.caption(", ".join(p["authors"]))
            st.write(p["summary"])
            st.markdown(f"[Read Full Paper]({p['link']})")
            st.divider()
    st.subheader("🎓 Recommended Courses")
    recs = recommend_learning_resources(topic)
    for link in recs["courses"]:
        st.markdown(f"- [{link}]({link})")

def render_skills():
    st.header("🔍 Skill & Job Trends")
    skills = st.text_input("Enter skills (comma-separated):", "Python, Java, SQL")
    if st.button("Analyze"):
        keys = [k.strip() for k in skills.split(",")]
        trends = get_trends_keywords(keys)
        if trends:
            df = pd.DataFrame([{"Skill": k, "Change (%)": v["pct_change"]} for k, v in trends.items()])
            fig = px.bar(df, x="Skill", y="Change (%)", color="Skill", title="Skill Popularity (Last 3 Months)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Google Trends temporarily unavailable.")

def render_news():
    st.header("📰 News & Sentiment")
    query = st.text_input("Enter topic, company, or keyword:", "Indian Stock Market")
    num_items = st.slider("Number of articles:", 3, 15, 8)
    if st.button("Get News"):
        with st.spinner("Fetching latest headlines..."):
            articles = get_news(query, max_items=num_items)
            sentiments = analyze_headlines_sentiment(articles)
            if not sentiments:
                st.warning("⚠️ No recent articles found.")
            else:
                for art in sentiments:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {art['title']}")
                        if art["link"]:
                            st.markdown(f"[🔗 Read full article]({art['link']})")
                        st.caption(f"🕒 {art['published'].strftime('%b %d, %Y %H:%M')}")
                    with col2:
                        sentiment = art["sentiment"]["label"]
                        score = art["sentiment"]["score"]
                        color = "🟢" if "POS" in sentiment else "🔴" if "NEG" in sentiment else "⚪"
                        st.markdown(f"**{color} {sentiment} ({score:.2f})**")
                    st.divider()

def render_feedback():
    st.header("💬 Feedback")
    name = st.text_input("Your Name")
    rating = st.slider("Rate IntelliSphere (1-5):", 1, 5, 4)
    comments = st.text_area("Your suggestions or thoughts:")
    if st.button("Submit Feedback"):
        new_entry = {
            "Name": name,
            "Rating": rating,
            "Comments": comments,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            df = pd.read_csv("feedback.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Name", "Rating", "Comments", "Timestamp"])
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv("feedback.csv", index=False)
        st.success("✅ Thank you for your feedback!")

# -----------------------------------------------------
# MAIN RENDER FUNCTION
# -----------------------------------------------------
def render_dashboard():
    query_params = st.query_params
    page = query_params.get("page", ["home"])[0]
    navbar()

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
    else:
        render_home()
