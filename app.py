# ==============================================
# IntelliSphere: AI-Powered Insight Dashboard 🌍
# Author: Debabrath
# ==============================================

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime
import requests

# ✅ Import backend helpers
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

# -----------------------------------------------------
# CUSTOM STYLING
# -----------------------------------------------------
st.markdown("""
    <style>
        .main {background-color: #0e1117; color: white;}
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #1e2229; color: white; border-radius: 8px;
        }
        div[data-testid="stMetricValue"] {color: #08e0d1;}
        div[data-testid="stMetricLabel"] {color: #c7c7c7;}
        h1, h2, h3, h4 {color: #00c2ff;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
st.sidebar.title("📊 IntelliSphere Menu")
section = st.sidebar.radio(
    "Select a Module",
    [
        "🏠 Home",
        "💹 Stock Insights",
        "🤖 AI Company Insights",
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
        company insights, tech trends, research papers, job skill forecasts, 
        and market sentiment — powered by data & intelligence.  

        Built with ❤️ by **Debabrath (B.Tech Final Year)** 
    """)
    st.image(
        "https://miro.medium.com/v2/resize:fit:1100/format:webp/1*0ZrJ6x8r9KxZxB1ITlmN_Q.png",
        use_container_width=True
    )
    st.success("✅ All systems operational!")

# -----------------------------------------------------
# 💹 STOCK INSIGHTS
# -----------------------------------------------------
elif section == "💹 Stock Insights":
    st.header("Stock Market Insights")

    user_input = st.text_input("Enter Company Name or Symbol:", "TCS")
    period = st.selectbox("Select Time Range:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

    if st.button("Get Stock Data"):
        ticker = user_input.strip().upper()
        if "." not in ticker:
            ticker += ".NS"

        stock = yf.Ticker(ticker)
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

        if not df.empty:
            info = stock.info
            latest_price = round(df["Close"].iloc[-1], 2)
            change = round(((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100, 2)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{ticker}", f"₹{latest_price}", f"{change}%")
            c2.metric("P/E Ratio", info.get("trailingPE", "N/A"))
            c3.metric("Market Cap", f"{info.get('marketCap', 'N/A'):,}" if info.get("marketCap") else "N/A")

            c4, c5, c6 = st.columns(3)
            c4.metric("52W High", info.get("fiftyTwoWeekHigh", "N/A"))
            c5.metric("52W Low", info.get("fiftyTwoWeekLow", "N/A"))
            c6.metric("Volume", info.get("volume", "N/A"))

            df_recent = df.reset_index()
            fig = px.line(df_recent, x="Date", y="Close", title=f"{ticker} Price Trend ({period})", markers=True)
            fig.update_layout(template="plotly_dark", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ Could not fetch data. Try again later.")

# -----------------------------------------------------
# 🤖 AI COMPANY INSIGHTS
# -----------------------------------------------------
elif section == "🤖 AI Company Insights":
    st.header("AI Company Insights Dashboard")
    company = st.text_input("Enter company name:", "Infosys")

    if st.button("Analyze Company"):
        ticker = company.upper().strip()
        if "." not in ticker:
            ticker += ".NS"
        stock = yf.Ticker(ticker)
        info = stock.info

        st.subheader(f"{info.get('longName', company)} Overview")
        st.write(info.get("longBusinessSummary", "No detailed description available."))

        st.divider()
        st.subheader("Company Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sector", info.get("sector", "N/A"))
        col2.metric("Industry", info.get("industry", "N/A"))
        col3.metric("Employees", info.get("fullTimeEmployees", "N/A"))

        col4, col5, col6 = st.columns(3)
        col4.metric("P/E Ratio", info.get("trailingPE", "N/A"))
        col5.metric("Profit Margin", info.get("profitMargins", "N/A"))
        col6.metric("Return on Equity", info.get("returnOnEquity", "N/A"))

        if info.get("website"):
            st.markdown(f" [Official Website]({info['website']})")

# -----------------------------------------------------
# 💻 TECH & STARTUP TRENDS
# -----------------------------------------------------
elif section == "💻 Tech & Startup Trends":
    st.header("Global Tech & Startup Trends")
    lang = st.text_input("Enter language (e.g., Python, Java):", "python")
    sort_by = st.radio("Sort repositories by:", ["Stars (High → Low)", "Name (A → Z)"], horizontal=True)

    if st.button("Fetch Trending Repositories"):
        repos = fetch_github_trending(lang)
        if repos:
            for r in repos[:10]:
                st.markdown(f"### [{r['name']}]({'https://github.com/' + r['name']}) ⭐ {r['stars']}")
                st.caption(r['description'] or "_No description available_")
                st.divider()
        else:
            st.warning("No trending repositories found right now.")

    st.subheader("Latest Startup News")
    startup_news = get_news("startup OR funding OR venture capital India", max_items=6)
    startup_sent = analyze_headlines_sentiment(startup_news)
    for n in startup_sent:
        st.markdown(f"**{n['title']}**")
        if n['link']:
            st.markdown(f"[Read more]({n['link']})")
        st.caption(f"Sentiment: {n['sentiment']['label']} ({n['sentiment']['score']:.2f})")
        st.divider()

# -----------------------------------------------------
# 📚 RESEARCH & EDUCATION
# -----------------------------------------------------
elif section == "📚 Research & Education":
    st.header("Research & Learning Explorer")
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

# -----------------------------------------------------
# 🔍 SKILL & JOB TRENDS
# -----------------------------------------------------
elif section == "🔍 Skill & Job Trends":
    st.header("Skill Popularity Trends")
    skills = st.text_input("Enter skills (comma-separated):", "Python, Java, SQL")
    if st.button("Analyze"):
        keys = [k.strip() for k in skills.split(",")]
        trends = get_trends_keywords(keys)
        if trends:
            df = pd.DataFrame([
                {"Skill": k, "Change (%)": v["pct_change"]} for k, v in trends.items()
            ])
            fig = px.bar(df, x="Skill", y="Change (%)", color="Skill", title="Skill Popularity (Last 3 Months)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Google Trends temporarily unavailable.")

# -----------------------------------------------------
# 📰 NEWS & SENTIMENT
# -----------------------------------------------------
elif section == "📰 News & Sentiment":
    st.header("Live News & Market Sentiment")
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
                        st.markdown(f"### 🗞️ {art['title']}")
                        if art["link"]:
                            st.markdown(f"[🔗 Read full article]({art['link']})")
                        st.caption(f"🕒 {art['published'].strftime('%b %d, %Y %H:%M')}")
                    with col2:
                        sentiment = art["sentiment"]["label"]
                        score = art["sentiment"]["score"]
                        color = "🟢" if "POS" in sentiment else "🔴" if "NEG" in sentiment else "⚪"
                        st.markdown(f"**{color} {sentiment} ({score:.2f})**")
                    st.divider()

# -----------------------------------------------------
# 💬 FEEDBACK
# -----------------------------------------------------
elif section == "💬 Feedback":
    st.header("Share Your Feedback")
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
# FOOTER
# -----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info(f"Developed by **Debabrath** | Last Updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
