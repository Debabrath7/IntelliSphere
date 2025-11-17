# ==========================================================
# backend_modules.py — Data helpers & safe wrappers
# Author: Debabrath (refactor)
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Optional: transformers for sentiment (kept but optional)
try:
    from transformers import pipeline
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception:
            sentiment_analyzer = None
except Exception:
    sentiment_analyzer = None

# -------------------------
# Utilities
# -------------------------
def clean_ticker(ticker: str) -> str:
    """Normalize tickers: remove .NS/.BO and whitespace, keep upper-case."""
    if not isinstance(ticker, str):
        return ticker
    s = ticker.strip()
    # Accept inputs like "541143" (BSE numeric) and return as-is
    # Remove common suffixes
    for suf in [".NS", ":NS", ".BO", ":BO", " NS", " BO"]:
        if s.upper().endswith(suf):
            s = s[: -len(suf)]
    return s.strip().upper()

def safe_get(url, headers=None, params=None, timeout=10):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None

# -------------------------
# Stock data fetchers
# -------------------------
def _normalize_df(df):
    """Return df with Date/Datetime normalization and expected columns."""
    if df is None:
        return None
    df = df.copy()
    # reset index
    if not isinstance(df, pd.DataFrame):
        return None
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
    # coerce to datetime
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass
    # ensure OHLCV exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

def get_stock_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    Download stock data using yfinance.
    ticker: e.g. 'RELIANCE.NS' or '541143.BO' or 'RELIANCE'
    period: '1d','5d','1mo','3mo','6mo','1y','2y','5y'
    interval: '1m','2m','5m','15m','30m','60m','1d'
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True, auto_adjust=False)
    except Exception:
        # retry once with small delay
        try:
            time.sleep(0.8)
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True, auto_adjust=False)
        except Exception:
            return None
    if df is None or df.empty:
        return None
    df = df.reset_index()
    return _normalize_df(df)

def get_intraday_data(ticker: str, interval: str = "1m", period: str = "1d"):
    return get_stock_data(ticker, period=period, interval=interval)

def get_today_high_low(ticker: str, prefer_intraday=True):
    """
    Returns dict: {'high':float,'low':float,'timestamp':datetime} or None
    """
    if prefer_intraday:
        df = get_intraday_data(ticker, interval="1m", period="1d")
        if df is not None and not df.empty:
            df2 = df.dropna(subset=["Close"])
            if not df2.empty:
                return {"high": float(df2["High"].max()), "low": float(df2["Low"].min()), "timestamp": df2["Date"].iloc[-1]}
    # fallback to daily
    df = get_stock_data(ticker, period="5d", interval="1d")
    if df is not None and not df.empty:
        row = df.iloc[-1]
        return {"high": float(row["High"]), "low": float(row["Low"]), "timestamp": row["Date"]}
    return None

# -------------------------
# stock summary (safe)
# -------------------------
@lru_cache(maxsize=128)
def stock_summary(ticker: str, period: str = "6mo", interval: str = "1d"):
    df = get_stock_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return None
    try:
        first = float(df["Close"].dropna().iloc[0])
        last = float(df["Close"].dropna().iloc[-1])
        pct = (last - first) / (first + 1e-9) * 100
        return {"ticker": ticker, "first_close": first, "latest_close": last, "pct_change": pct, "recent": df.tail(30).to_dict(orient="records")}
    except Exception:
        return None

# -------------------------
# News & sentiment helpers (kept minimal)
# -------------------------
def fetch_news_via_google_rss(query: str, max_items: int = 10):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_items]:
        try:
            pub = datetime(*entry.published_parsed[:6])
        except Exception:
            pub = datetime.now()
        articles.append({"title": entry.title, "link": entry.link, "published": pub})
    return articles

def analyze_headlines_sentiment(articles):
    if sentiment_analyzer is None:
        return [{**a, "sentiment": None} for a in articles]
    out = []
    for a in articles:
        txt = a.get("title", "")[:512]
        try:
            s = sentiment_analyzer(txt)[0]
            out.append({**a, "sentiment": {"label": s.get("label"), "score": float(s.get("score", 0.0))}})
        except Exception:
            out.append({**a, "sentiment": None})
    return out

# -------------------------
# Misc: Github / ArXiv simple wrappers (kept as-is)
# -------------------------
def fetch_github_trending(language=None, since="daily"):
    url = f"https://github.com/trending{f'/{language}' if language else ''}?since={since}"
    r = safe_get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    repos = []
    for repo in soup.select("article.Box-row")[:20]:
        title_tag = repo.find(["h1", "h2"])
        if not title_tag:
            continue
        link_tag = title_tag.find("a")
        name = link_tag.get("href","").strip("/") if link_tag else "unknown"
        desc = (repo.find("p") or type("x",(),{"get_text":lambda *_:""})) .get_text(strip=True)
        repos.append({"name": name, "description": desc})
    return repos

def fetch_arxiv_papers(query, max_results=5):
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    r = safe_get(base, params=params)
    if not r:
        return []
    soup = BeautifulSoup(r.text, features="xml")
    papers=[]
    for entry in soup.find_all("entry")[:max_results]:
        papers.append({"title": entry.title.get_text(strip=True), "summary": entry.summary.get_text(strip=True), "link": entry.id.get_text(strip=True)})
    return papers
