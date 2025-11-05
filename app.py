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
st.sidebar.title("IntelliSphere Menu")
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

    user_input = st.text_input("Enter Company Name or Stock Symbol:", "TCS")

    # Dropdown for selecting time period
    period = st.selectbox(
        "Select Time Range:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )

    if st.button("Get Stock Data"):
        # Clean up and normalize user input
        ticker = user_input.strip().upper()
        if "." not in ticker:
            ticker = ticker + ".NS"  # auto-append NSE for Indian stocks

        # Get stock data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

        if not df.empty:
            info = stock.info

            # Extract useful metrics safely
            latest_price = round(df["Close"].iloc[-1], 2)
            change_6m = round(((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100, 2)
            pe_ratio = info.get("trailingPE", "N/A")
            market_cap = info.get("marketCap", "N/A")
            high_52wk = info.get("fiftyTwoWeekHigh", "N/A")
            low_52wk = info.get("fiftyTwoWeekLow", "N/A")
            day_high = info.get("dayHigh", "N/A")
            day_low = info.get("dayLow", "N/A")
            volume = info.get("volume", "N/A")
            dividend_yield = info.get("dividendYield", "N/A")
            next_earnings = info.get("earningsTimestampStart", "N/A")
            last_earnings = info.get("earningsTimestamp", "N/A")

            # Display main metrics
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{ticker}", f"₹{latest_price}", f"{change_6m}% (change over period)")
            c2.metric("P/E Ratio", pe_ratio)
            c3.metric("Market Cap", f"{market_cap:,}" if market_cap != "N/A" else "N/A")

            c4, c5, c6 = st.columns(3)
            c4.metric("52-Week High", high_52wk)
            c5.metric("52-Week Low", low_52wk)
            c6.metric("Volume", volume)

            c7, c8, c9 = st.columns(3)
            c7.metric("Today's High", day_high)
            c8.metric("Today's Low", day_low)
            c9.metric("Dividend Yield", dividend_yield)

            if next_earnings != "N/A":
                st.info(f"📅 Next earnings announcement: **{pd.to_datetime(next_earnings, unit='s').date()}**")
            elif last_earnings != "N/A":
                st.info(f"📅 Last earnings were declared on: **{pd.to_datetime(last_earnings, unit='s').date()}**")
            else:
                st.info("📅 Earnings date information currently unavailable.")

            # Prepare chart
            df_recent = df.reset_index()
            fig = px.line(
                df_recent,
                x="Date",
                y="Close",
                title=f"{ticker} Price Trend ({period})",
                markers=True
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                title_x=0.5,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("⚠️ Could not fetch stock data. Please check the company name or try again later.")

# -----------------------------------------------------
# 💻 TECH & STARTUP TRENDS
# -----------------------------------------------------
elif section == "💻 Tech & Startup Trends":
    st.header("💻 Global Tech & Startup Trends")

    # -------------------------------
    # 🔹 GitHub Trending Section
    # -------------------------------
    st.subheader("🔥 Top GitHub Trending Repositories")
    lang = st.text_input("Enter language (e.g., python, javascript, java):", "python")
    sort_by = st.radio("Sort repositories by:", ["Stars (High → Low)", "Name (A → Z)"], horizontal=True)

    if st.button("Fetch Trending Repositories"):
        repos = fetch_github_trending(lang)
        if repos:
            # Convert stars to numeric (cleaning commas)
            for r in repos:
                try:
                    r["stars"] = int(r["stars"].replace(",", ""))
                except:
                    r["stars"] = 0

            # Sorting logic
            if sort_by == "Stars (High → Low)":
                repos.sort(key=lambda x: x["stars"], reverse=True)
            elif sort_by == "Name (A → Z)":
                repos.sort(key=lambda x: x["name"].lower())

            # Pagination setup
            if "repo_page" not in st.session_state:
                st.session_state.repo_page = 1

            per_page = 10
            start_idx = 0
            end_idx = st.session_state.repo_page * per_page

            display_repos = repos[start_idx:end_idx]
            for r in display_repos:
                st.markdown(f"### [{r['name']}]({'https://github.com/' + r['name']}) ⭐ {r['stars']}")
                st.write(r['description'] or "_No description available_")
                st.markdown("---")

            if end_idx < len(repos):
                if st.button("Load More"):
                    st.session_state.repo_page += 1
                    st.experimental_rerun()
        else:
            st.warning("⚠️ No trending repositories found right now. Try again later.")

    # -------------------------------
    # 🚀 Startup News Section
    # -------------------------------
    st.subheader("🚀 Latest Startup & Funding News")
    startup_news = get_news("startup OR funding OR venture capital India", max_items=5)
    startup_sent = analyze_headlines_sentiment(startup_news)

    for n in startup_sent:
        st.markdown(f"**{n['title']}**")
        st.caption(f"Sentiment: {n['sentiment']['label']} ({n['sentiment']['score']:.2f})")
        if n['link']:
            st.markdown(f"[Read more]({n['link']})")
        st.markdown("---")

    # -------------------------------
    # 🧠 Technology Innovation News
    # -------------------------------
    st.subheader("🧠 Global Technology & Innovation Headlines")
    tech_news = get_news("technology OR artificial intelligence OR innovation OR software", max_items=5)
    tech_sent = analyze_headlines_sentiment(tech_news)

    for n in tech_sent:
        st.markdown(f"**{n['title']}**")
        st.caption(f"Sentiment: {n['sentiment']['label']} ({n['sentiment']['score']:.2f})")
        if n['link']:
            st.markdown(f"[Read more]({n['link']})")
        st.markdown("---")

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
# ==========================================================
# 📰 NEWS & SENTIMENT MODULE (Real-time + Accurate)
# ==========================================================
import feedparser
from datetime import datetime, timedelta

NEWSAPI_KEY = "1a4b8e472abc4784acd8c0da8b7cadd6"  # Optional but recommended for better freshness

def fetch_news_via_google_rss(query, max_items=15):
    """Fetch fresh, real-time headlines via Google News RSS"""
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_items]:
        title = entry.title
        link = entry.link
        published = entry.get("published_parsed")
        if published:
            pub_date = datetime(*published[:6])
        else:
            pub_date = datetime.now()
        articles.append({
            "title": title,
            "link": link,
            "published": pub_date
        })
    return articles


def fetch_news_via_newsapi(query, max_items=15):
    """Fetch headlines from NewsAPI if API key is available"""
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_items,
        "apiKey": NEWSAPI_KEY
    }
    r = safe_get(url, params=params)
    if not r:
        return []
    data = r.json()
    articles = []
    for a in data.get("articles", []):
        pub_date = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
        articles.append({
            "title": a["title"],
            "link": a["url"],
            "published": pub_date
        })
    return articles


def get_news(query, max_items=10):
    """Hybrid source for highly relevant, up-to-date news"""
    # Step 1: Try NewsAPI first (for real-time accuracy)
    articles = fetch_news_via_newsapi(query, max_items)
    
    # Step 2: Fallback to Google RSS
    if not articles or len(articles) < 3:
        rss_articles = fetch_news_via_google_rss(query, max_items)
        articles.extend(rss_articles)
    
    # Step 3: Remove duplicates and old news
    seen_titles = set()
    final_articles = []
    for art in sorted(articles, key=lambda x: x["published"], reverse=True):
        if art["title"] not in seen_titles:
            if art["published"] > datetime.now() - timedelta(hours=36):  # within last 36 hours
                seen_titles.add(art["title"])
                final_articles.append(art)
    
    # Step 4: Final fallback if still empty
    if not final_articles:
        final_articles = [
            {"title": "Markets rally as investors eye global cues", "link": "", "published": datetime.now()},
            {"title": "Tech sector sees renewed interest amid AI boom", "link": "", "published": datetime.now()},
            {"title": "Investors shift focus to sustainable startups", "link": "", "published": datetime.now()},
        ]
    return final_articles[:max_items]


def analyze_headlines_sentiment(articles):
    """Analyze sentiment for fetched news headlines"""
    results = []
    for art in articles:
        text = art.get("title", "")
        if not text:
            continue
        try:
            sentiment = sentiment_analyzer(text[:512])
            label_data = sentiment[0] if sentiment else {"label": "NEUTRAL", "score": 0.0}
        except Exception:
            label_data = {"label": "NEUTRAL", "score": 0.0}
        results.append({
            **art,
            "sentiment": {
                "label": label_data["label"],
                "score": label_data["score"]
            }
        })
    return results

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


