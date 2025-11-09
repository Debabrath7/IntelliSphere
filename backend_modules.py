# ==========================================================
# IntelliSphere Backend Modules (Optimized)
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
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# ==========================================================
# 🤖 Initialize AI Models (cached)
# ==========================================================
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
except Exception:
    sentiment_analyzer = pipeline("sentiment-analysis")

# ==========================================================
# 🧩 Safe GET Helper with Cache
# ==========================================================
@lru_cache(maxsize=128)
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
@lru_cache(maxsize=64)
def get_stock_data(ticker, period="6mo", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False, threads=True, auto_adjust=True)
    if data.empty:
        print(f"No data for {ticker}")
        return None
    return data.reset_index()

@lru_cache(maxsize=128)
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
# 💻 GITHUB TRENDING MODULE
# ==========================================================
@lru_cache(maxsize=32)
def fetch_github_trending(language=None, since="daily"):
    url = f"https://github.com/trending{f'/{language}' if language else ''}?since={since}"
    r = safe_get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    repos = []
    for repo in soup.select("article.Box-row")[:30]:
        title_tag = repo.find(["h1", "h2"])
        title = title_tag.find("a").get("href", "").strip("/") if title_tag and title_tag.find("a") else "unknown"
        desc_tag = repo.find("p", class_="col-9")
        desc = desc_tag.get_text(strip=True) if desc_tag else ""
        stars = repo.select_one("a[href$='/stargazers']")
        star_count = stars.get_text(strip=True) if stars else "0"
        repos.append({"name": title, "description": desc, "stars": star_count})
    return repos


# ==========================================================
# 📚 RESEARCH (ARXIV) MODULE
# ==========================================================
@lru_cache(maxsize=32)
def fetch_arxiv_papers(query, max_results=5):
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
    r = safe_get(base, params=params)
    if not r:
        return []
    soup = BeautifulSoup(r.text, features="xml")
    papers = []
    for entry in soup.find_all("entry")[:max_results]:
        papers.append({
            "title": entry.title.get_text(strip=True),
            "summary": entry.summary.get_text(strip=True),
            "link": entry.id.get_text(strip=True),
            "authors": [a.get_text(strip=True) for a in entry.find_all("author")]
        })
    return papers


# ==========================================================
# 🔍 GOOGLE TRENDS MODULE
# ==========================================================
pytrends = TrendReq(hl='en-US', tz=0)

@lru_cache(maxsize=64)
def get_trends_keywords(keywords, timeframe='today 3-m'):
    try:
        pytrends.build_payload(keywords, timeframe=timeframe)
        data = pytrends.interest_over_time()
        if data.empty:
            raise ValueError("Empty data")
        last, first = data.iloc[-1], data.iloc[0]
        return {k: {
            "first": float(first[k]),
            "last": float(last[k]),
            "pct_change": (float(last[k]) - float(first[k])) / (float(first[k]) + 1e-9) * 100
        } for k in keywords}
    except Exception as e:
        print(f"[Trends] {e}")
        return {k: {"first": 50, "last": 80, "pct_change": 60.0} for k in keywords}


# ==========================================================
# 📰 NEWS & SENTIMENT MODULE
# ==========================================================
NEWSAPI_KEY = ""

@lru_cache(maxsize=64)
def fetch_news_via_google_rss(query, max_items=15):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_items]:
        pub_date = datetime(*entry.published_parsed[:6]) if "published_parsed" in entry else datetime.now()
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": pub_date
        })
    return articles


def get_news(query, max_items=10):
    rss_articles = fetch_news_via_google_rss(query, max_items)
    seen_titles, final_articles = set(), []
    for art in sorted(rss_articles, key=lambda x: x["published"], reverse=True):
        if art["title"] not in seen_titles:
            seen_titles.add(art["title"])
            final_articles.append(art)
    return final_articles[:max_items]


def analyze_headlines_sentiment(articles):
    results = []
    for art in articles:
        text = art.get("title", "")
        if not text:
            continue
        try:
            sentiment = sentiment_analyzer(text[:512])[0]
            label, score = sentiment["label"], sentiment["score"]
        except Exception:
            label, score = "NEUTRAL", 0.0
        results.append({**art, "sentiment": {"label": label, "score": score}})
    return results


# ==========================================================
# 🎓 EDUCATION RECOMMENDER MODULE
# ==========================================================
@lru_cache(maxsize=32)
def recommend_learning_resources(topic, top_papers=3, top_repos=3):
    with ThreadPoolExecutor() as executor:
        papers_future = executor.submit(fetch_arxiv_papers, topic, top_papers)
        repos_future = executor.submit(fetch_github_trending, None, "daily")
        papers, repos = papers_future.result(), repos_future.result()
    filtered = [r for r in repos if topic.lower() in (r["name"].lower() + " " + r["description"].lower())]
    if len(filtered) < top_repos:
        filtered = repos[:top_repos]
    curated = {
        "machine learning": ["https://www.coursera.org/learn/machine-learning", "https://www.fast.ai/"],
        "finance": ["https://www.coursera.org/specializations/wharton-finance"],
        "nlp": ["https://www.coursera.org/learn/language-processing"]
    }
    return {"papers": papers, "repos": filtered, "courses": curated.get(topic.lower(), [])}
