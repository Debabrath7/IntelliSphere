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

# Initialize sentiment analyzer (Hugging Face)
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Helper: Safe GET with error handling
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

    for repo in soup.select("article.Box-row")[:20]:
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
    # fallback data for demo
    return {k: {"first": 50, "last": 80, "pct_change": 60.0} for k in keywords}


# ==========================================================
# 📰 NEWS & SENTIMENT MODULE (multi-source)
# ==========================================================
def fetch_news_via_google_news_rss(query, max_items=10):
    url = f"https://news.google.com/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    r = safe_get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    articles = []
    for a in soup.select("article")[:max_items]:
        title_tag = a.find("a", {"class": "DY5T1d"})
        if title_tag:
            title = title_tag.get_text(strip=True)
            href = title_tag.get("href")
            if href and href.startswith("./"):
                href = "https://news.google.com" + href[1:]
            articles.append({"title": title, "link": href})
    return articles


def fetch_news_via_yahoo_finance(max_items=10):
    yahoo_url = "https://finance.yahoo.com/rss/topstories"
    r = safe_get(yahoo_url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "xml")
    articles = []
    for item in soup.find_all("item")[:max_items]:
        title = item.title.get_text(strip=True)
        link = item.link.get_text(strip=True)
        articles.append({"title": title, "link": link})
    return articles


def get_news(query, max_items=10):
    articles = fetch_news_via_google_news_rss(query + " India", max_items=max_items)
    if not articles:
        print("Google News empty, switching to Yahoo Finance RSS...")
        articles = fetch_news_via_yahoo_finance(max_items=max_items)
    if not articles:
        print("All sources unavailable; using static fallback headlines.")
        articles = [
            {"title": "Indian markets end higher on positive global cues", "link": ""},
            {"title": "Technology sector shows strong quarterly growth", "link": ""},
            {"title": "Rising EV adoption boosts automobile industry outlook", "link": ""}
        ]
    return articles


def analyze_headlines_sentiment(articles):
    results = []
    for art in articles:
        text = art.get("title", "")
        if not text:
            continue
        try:
            output = sentiment_analyzer(text[:512])
            if output and len(output) > 0:
                label_data = output[0]
                label = label_data.get("label", "NEUTRAL")
                score = label_data.get("score", 0.0)
            else:
                label, score = "NEUTRAL", 0.0
        except Exception:
            label, score = "NEUTRAL", 0.0
        results.append({**art, "sentiment": {"label": label, "score": score}})
    return results


# ==========================================================
# 🎓 EDUCATION RECOMMENDER
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
