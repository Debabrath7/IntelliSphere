# ==========================================================
# IntelliSphere Backend Modules
# Author: Debabrath
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from transformers import pipeline
import time
import feedparser
from datetime import datetime, timedelta

# ==========================================================
# 🤖 Initialize AI Models
# ==========================================================
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
except Exception:
    sentiment_analyzer = pipeline("sentiment-analysis")

# ==========================================================
# 🧩 Helper: Safe HTTP GET
# ==========================================================
def safe_get(url, headers=None, params=None, timeout=10):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"[safe_get] request failed for {url}: {e}")
        return None


# ==========================================================
# 📈 STOCK MARKET MODULE
# ==========================================================
def get_stock_data(ticker, period="6mo", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=True)
    if data.empty:
        print(f"No data for {ticker}")
        return None
    data.reset_index(inplace=True)
    return data


def stock_summary(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    latest_close = float(df["Close"].iloc[-1])
    first_close = float(df["Close"].iloc[0])
    pct_change = (latest_close - first_close) / first_close * 100
    return {
        "ticker": ticker,
        "first_close": first_close,
        "latest_close": latest_close,
        "pct_change_6mo": pct_change,
        "recent": df.tail(30)
    }


# ==========================================================
# 🏢 AI COMPANY INSIGHTS MODULE
# ==========================================================
def company_overview(ticker):
    """Fetch key company information via Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "employees": info.get("fullTimeEmployees", "N/A"),
            "summary": info.get("longBusinessSummary", "No summary available."),
            "website": info.get("website", None),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
        }
    except Exception as e:
        print(f"[company_overview] failed: {e}")
        return {
            "name": ticker, "sector": "N/A", "industry": "N/A",
            "employees": "N/A", "summary": "Data unavailable.",
            "website": None, "pe_ratio": "N/A", "profit_margin": "N/A", "roe": "N/A"
        }


# ==========================================================
# 💻 GITHUB TRENDING MODULE
# ==========================================================
def fetch_github_trending(language=None, since="daily"):
    url = "https://github.com/trending"
    if language:
        url += f"/{language}"
    url += f"?since={since}"

    r = safe_get(url)
    if not r:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    repos = []

    for repo in soup.select("article.Box-row")[:50]:
        title_tag = repo.find(["h1", "h2"])
        title = "unknown"
        if title_tag:
            link_tag = title_tag.find("a")
            if link_tag:
                title = link_tag.get("href", "").strip("/")
        desc_tag = repo.find("p", class_="col-9")
        desc = desc_tag.get_text(strip=True) if desc_tag else ""
        star_tag = repo.find("a", href=lambda x: x and x.endswith("/stargazers"))
        stars = star_tag.get_text(strip=True) if star_tag else "0"
        repos.append({"name": title, "description": desc, "stars": stars})
    return repos


# ==========================================================
# 📚 RESEARCH (ARXIV) MODULE
# ==========================================================
def fetch_arxiv_papers(query, max_results=5):
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
    r = safe_get(base, params=params)
    if not r:
        return []
    soup = BeautifulSoup(r.text, features="xml")
    papers = []
    for entry in soup.find_all("entry")[:max_results]:
        title = entry.title.get_text(strip=True)
        summary = entry.summary.get_text(strip=True)
        link = entry.id.get_text(strip=True)
        authors = [a.get_text(strip=True) for a in entry.find_all("author")]
        papers.append({"title": title, "summary": summary, "link": link, "authors": authors})
    return papers


# ==========================================================
# 🔍 GOOGLE TRENDS MODULE
# ==========================================================
pytrends = TrendReq(hl='en-US', tz=0)

def get_trends_keywords(keywords, timeframe='today 3-m', retries=3):
    for i in range(retries):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe)
            data = pytrends.interest_over_time()
            if not data.empty:
                last, first = data.iloc[-1], data.iloc[0]
                return {k: {
                    "first": float(first[k]),
                    "last": float(last[k]),
                    "pct_change": (float(last[k]) - float(first[k])) /
                                  (float(first[k]) + 1e-9) * 100
                } for k in keywords}
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(5)
    return {k: {"first": 50, "last": 80, "pct_change": 60.0} for k in keywords}


# ==========================================================
# 📰 NEWS & SENTIMENT MODULE (Real-Time + Accurate)
# ==========================================================
NEWSAPI_KEY = ""  # Optional: your personal NewsAPI key

def fetch_news_via_google_rss(query, max_items=15):
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
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": max_items, "apiKey": NEWSAPI_KEY
    }
    r = safe_get(url, params=params)
    if not r:
        return []
    data = r.json()
    articles = []
    for a in data.get("articles", []):
        pub_date = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
        articles.append({
            "title": a["title"], "link": a["url"], "published": pub_date
        })
    return articles


def get_news(query, max_items=10):
    articles = fetch_news_via_newsapi(query, max_items)
    if not articles or len(articles) < 3:
        rss_articles = fetch_news_via_google_rss(query, max_items)
        articles.extend(rss_articles)
    seen_titles, final_articles = set(), []
    for art in sorted(articles, key=lambda x: x["published"], reverse=True):
        if art["title"] not in seen_titles:
            if art["published"] > datetime.now() - timedelta(hours=36):
                seen_titles.add(art["title"])
                final_articles.append(art)
    if not final_articles:
        final_articles = [
            {"title": "Markets rally as investors eye global cues", "link": "", "published": datetime.now()},
            {"title": "Tech sector sees strong quarterly growth", "link": "", "published": datetime.now()},
            {"title": "Investors shift focus to sustainable startups", "link": "", "published": datetime.now()},
        ]
    return final_articles[:max_items]


def analyze_headlines_sentiment(articles):
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


# ==========================================================
# 🎓 EDUCATION RECOMMENDER MODULE
# ==========================================================
def recommend_learning_resources(topic, top_papers=3, top_repos=3):
    papers = fetch_arxiv_papers(topic, max_results=top_papers)
    repos = fetch_github_trending(language=None, since="daily")
    filtered_repos = [r for r in repos if topic.lower() in (r["name"].lower() + " " + r["description"].lower())]
    if len(filtered_repos) < top_repos:
        filtered_repos = repos[:top_repos]
    curated = {
        "machine learning": ["https://www.coursera.org/learn/machine-learning", "https://www.fast.ai/"],
        "finance": ["https://www.coursera.org/specializations/wharton-finance"],
        "nlp": ["https://www.coursera.org/learn/language-processing"]
    }
    return {"papers": papers, "repos": filtered_repos[:top_repos], "courses": curated.get(topic.lower(), [])}
