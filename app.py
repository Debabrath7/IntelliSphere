# ==============================================
# IntelliSphere: AI-Powered Insight Dashboard 🌍
# Author: Debabrath
# ==============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ✅ Import all backend modules
from backend_modules import (
    stock_summary,
    fetch_github_trending,
    fetch_arxiv_papers,
    get_trends_keywords,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)

# -----------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | AI-Powered Insights",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {background-color: #0e1117; color: white;}
        .stTextInput>div>div>input {background-color: #1e2229; color: white;}
        .stTextArea>div>div>textarea {background-color: #1e2229; color: white;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio(
    "Choose a Section",
    [
        "🏠 Home",
        "💹 Stock Insights",
        "💻 Tech & Startup Trends",
        "📚 Research & Education",
        "🔍 Skill & Job Trends",
        "📰 News & Sentiment",
        "💬 Feedback"
    ]
)

# -----------------------------------------------------
# 🏠 HOME TAB
# -----------------------------------------------------
if section == "🏠 Home":
    st.title("🌐 IntelliSphere: AI-Powered Insight Platform")
    st.write("""
        **Your all-in-one AI dashboard** for real-time stock analysis, 
        tech trends, research insights, skill forecasts, and news sentiment.

        Designed and developed by **Debabrath** 👨‍💻  
        Built with **Python**, **Streamlit**, **Hugging Face**, **Yahoo Finance**, and **ArXiv APIs**.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*0ZrJ6x8r9KxZxB1ITlmN_Q.png", use_container_width=True)
    st.success("✅ All backend modules verified and ready!")

# -----------------------------------------------------
# 💹 STOCK INSIGHTS
# -----------------------------------------------------
elif section == "💹 Stock Insights":
    st.header("📈 Stock Market Insights")
    ticker = st.text_input("Enter stock ticker (e.g., TCS.NS, INFY.NS):", "TCS.NS")
    if st.button("Get Stock Data"):
        s = stock_summary(ticker)
        if s:
            st.metric(label=f"{ticker} Latest Close", value=f"₹{s['latest_close']:.2f}")
            st.metric(label="6-Month Change (%)", value=f"{s['pct_change_6mo']:.2f}%")
            fig = px.line(s["recent"], x="Date", y="Close", title=f"{ticker} - Last 30 Days Performance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ No data found for that ticker.")

# -----------------------------------------------------
# 💻 TECH & STARTUP TRENDS
# -----------------------------------------------------
elif section == "💻 Tech & Startup Trends":
    st.header("💻 GitHub Trending Repositories")
    lang = st.text_input("Enter programming language (e.g., python, javascript):", "python")
    if st.button("Show Trending Repositories"):
        repos = fetch_github_trending(lang)
        if repos:
            df = pd.DataFrame(repos)
            st.dataframe(df[["name", "stars", "description"]])
        else:
            st.warning("⚠️ No trending repositories found right now. Try again later.")

# -----------------------------------------------------
# 📚 RESEARCH & EDUCATION
# -----------------------------------------------------
elif section == "📚 Research & Education":
    st.header("📚 Latest Research Papers & Learning Resources")
    topic = st.text_input("Enter research topic:", "machine learning")
    if st.button("Search Research Papers"):
        papers = fetch_arxiv_papers(topic, max_results=5)
        for p in papers:
            st.subheader(p["title"])
            st.caption(", ".join(p["authors"]))
            st.write(p["summary"])
            st.markdown(f"[Read Full Paper ➜]({p['link']})")
            st.markdown("---")

        st.subheader("🎓 Recommended Courses & Resources")
        recs = recommend_learning_resources(topic)
        if recs["courses"]:
            for link in recs["courses"]:
                st.markdown(f"- [{link}]({link})")
        else:
            st.info("No specific course recommendations available for this topic.")

# -----------------------------------------------------
# 🔍 SKILL & JOB TRENDS
# -----------------------------------------------------
elif section == "🔍 Skill & Job Trends":
    st.header("🔍 Skill Popularity Trends (Google Trends)")
    keywords = st.text_input("Enter skills (comma-separated):", "Python, Java")
    if st.button("Analyze Trends"):
        keys = [k.strip() for k in keywords.split(",")]
        trends = get_trends_keywords(keys)
        if trends:
            df = pd.DataFrame([
                {"Skill": k, "Change (%)": v["pct_change"]}
                for k, v in trends.items()
            ])
            fig = px.bar(df, x="Skill", y="Change (%)", color="Skill", title="Skill Popularity Change (3 Months)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Google Trends temporarily unavailable (rate limit).")

# -----------------------------------------------------
# 📰 NEWS & SENTIMENT
# -----------------------------------------------------
elif section == "📰 News & Sentiment":
    st.header("📰 News Headlines & Sentiment Analysis")
    query = st.text_input("Enter topic or company name:", "Tata Motors")
    if st.button("Analyze News Sentiment"):
        news = get_news(query, max_items=5)
        news_sent = analyze_headlines_sentiment(news)
        for n in news_sent:
            sentiment_label = n['sentiment']['label']
            score = n['sentiment']['score']
            st.subheader(n["title"])
            if sentiment_label.upper() in ["POSITIVE", "LABEL_2"]:
                st.success(f"Sentiment: POSITIVE ({score:.2f})")
            elif sentiment_label.upper() in ["NEGATIVE", "LABEL_0"]:
                st.error(f"Sentiment: NEGATIVE ({score:.2f})")
            else:
                st.info(f"Sentiment: NEUTRAL ({score:.2f})")

            if n["link"]:
                st.markdown(f"[🔗 Read more]({n['link']})")
            st.markdown("---")

# -----------------------------------------------------
# 💬 FEEDBACK TAB
# -----------------------------------------------------
elif section == "💬 Feedback":
    st.header("💬 Share Your Feedback")
    name = st.text_input("Your Name")
    rating = st.slider("Rate your experience (1-5):", 1, 5, 4)
    comments = st.text_area("Any suggestions or comments?")
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
# FOOTER
# -----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("Developed by **Debabrath** | Final Year B.Tech Project")
