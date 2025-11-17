# ==========================================================
# backend_modules.py 
# Author: Debabrath
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import time
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache

# Optional sentiment (unchanged)
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
    if not isinstance(ticker, str):
        return ticker
    s = ticker.strip().upper()
    for suf in [".NS", ".BO", ":NS", ":BO"]:
        if s.endswith(suf):
            s = s.replace(suf, "")
    return s.strip()

def resolve_ticker_candidates(symbol: str):
    """Return candidates: NSE (.NS), BSE (.BO), numeric BSE, raw."""
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        return [f"{s}.BO", s]
    return [f"{s}.NS", f"{s}.BO", s]

def _normalize_df(df):
    """Return DataFrame with Date,Open,High,Low,Close,Volume or None."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.copy()
    # Normalize date column
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index":"Date"})
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass
    # Ensure OHLCV
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    # Keep only relevant columns
    out = df.loc[:, ["Date","Open","High","Low","Close","Volume"]]
    # drop rows with invalid dates or all NaNs in price
    out = out.dropna(subset=["Date"])
    if out.empty:
        return None
    return out

# -------------------------
# yfinance download with retry
# -------------------------
def _yf_download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return None
        return df.reset_index()
    except Exception:
        try:
            time.sleep(0.8)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
            if df is None or df.empty:
                return None
            return df.reset_index()
        except Exception:
            return None

# -------------------------
# Yahoo CSV fallback
# -------------------------
_PERIOD_DAYS = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 90,
    "6mo": 182,
    "1y": 365,
    "2y": 365*2,
    "5y": 365*5,
    "10y": 365*10
}

def _period_to_days(period):
    # accept e.g. '6mo', '1y', '1d', or int days
    if isinstance(period, int):
        return period
    p = str(period).lower()
    if p in _PERIOD_DAYS:
        return _PERIOD_DAYS[p]
    # try parse like '6mo' or '3m'
    if p.endswith("mo") or p.endswith("m"):
        try:
            return int(p.rstrip("mo").rstrip("m")) * 30
        except:
            pass
    if p.endswith("y"):
        try:
            return int(p.rstrip("y")) * 365
        except:
            pass
    # default 365
    return 365

def _yahoo_csv_download(ticker, period="6mo", interval="1d"):
    """
    Download historical CSV directly from Yahoo Finance:
    query1.finance.yahoo.com/v7/finance/download/{ticker}
    Params: period1, period2 (unix), interval
    """
    try:
        # compute timestamps
        days = _period_to_days(period)
        end = int(datetime.utcnow().timestamp())
        start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        params = {
            "period1": start,
            "period2": end,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true"
        }
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        s = r.text
        if not s or "404 Not Found" in s:
            return None
        df = pd.read_csv(io.StringIO(s), parse_dates=["Date"])
        # Some CSVs contain 'null' or nonstandard; ensure columns
        if df.empty:
            return None
        # rename/ensure columns expected by _normalize_df
        cols_lower = [c.lower() for c in df.columns]
        # Standard mapping check
        expected = ["Date","Open","High","Low","Close","Adj Close","Volume"]
        # If 'Adj Close' present keep Close as Close
        if "adj close" in cols_lower and "close" in cols_lower:
            # columns likely standard
            pass
        # Ensure numeric
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan
        return df.reset_index(drop=True)
    except Exception:
        return None

# -------------------------
# get_best_ticker: try yfinance then yahoo csv
# -------------------------
def get_best_ticker(symbol, period="6mo", interval="1d"):
    """
    Try all candidate tickers for symbol and return (working_ticker, normalized_df) or (None,None).
    """
    candidates = resolve_ticker_candidates(symbol)
    for t in candidates:
        # 1) try yfinance
        df = _yf_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None:
            return t, df
        # 2) try yahoo csv fallback
        df2 = _yahoo_csv_download(t, period, interval)
        df2 = _normalize_df(df2)
        if df2 is not None:
            return t, df2
    return None, None

# -------------------------
# Public fetch functions (used by frontend)
# -------------------------
def get_stock_data(symbol: str, period="6mo", interval="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df

def get_intraday_data(symbol: str, interval="1m", period="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df

def get_today_high_low(symbol: str):
    t, df = get_best_ticker(symbol, "1d", "1m")
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["Close"])
    if d.empty:
        return None
    return {"high": float(d["High"].max()), "low": float(d["Low"].min()), "timestamp": d["Date"].iloc[-1]}

@lru_cache(maxsize=128)
def stock_summary(symbol: str, period="6mo", interval="1d"):
    t, df = get_best_ticker(symbol, period, interval)
    if df is None:
        return None
    try:
        first = float(df["Close"].iloc[0])
        last = float(df["Close"].iloc[-1])
        return {"ticker": t, "first_close": first, "latest_close": last, "pct_change": (last-first)/(first+1e-9)*100, "recent": df.tail(30).to_dict("records")}
    except Exception:
        return None

# -------------------------
# News & sentiment (unchanged)
# -------------------------
import feedparser
def fetch_news_via_google_rss(query: str, max_items: int = 10):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_items]:
        try:
            pub = datetime(*entry.published_parsed[:6])
        except Exception:
            pub = datetime.utcnow()
        articles.append({"title": entry.title, "link": entry.link, "published": pub})
    return articles

def analyze_headlines_sentiment(articles):
    if sentiment_analyzer is None:
        return [{**a, "sentiment": None} for a in articles]
    out = []
    for a in articles:
        try:
            s = sentiment_analyzer(a["title"][:512])[0]
            out.append({**a, "sentiment": {"label": s.get("label"), "score": float(s.get("score", 0.0))}})
        except Exception:
            out.append({**a, "sentiment": None})
    return out

# -------------------------
# lightweight helpers for trending / arxiv (unchanged)
# -------------------------
from bs4 import BeautifulSoup
def fetch_github_trending(language=None, since="daily"):
    url = f"https://github.com/trending{f'/{language}' if language else ''}?since={since}"
    r = requests.get(url, timeout=8)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    repos = []
    for repo in soup.select("article.Box-row")[:20]:
        title_tag = repo.find(["h1","h2"])
        if not title_tag:
            continue
        link = title_tag.find("a").get("href","").strip("/")
        desc_tag = repo.find("p")
        desc = desc_tag.get_text(strip=True) if desc_tag else ""
        repos.append({"name": link, "description": desc})
    return repos

def fetch_arxiv_papers(query, max_results=5):
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    r = requests.get(base, params=params, timeout=8)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, features="xml")
    papers=[]
    for entry in soup.find_all("entry")[:max_results]:
        papers.append({"title": entry.title.get_text(strip=True), "summary": entry.summary.get_text(strip=True), "link": entry.id.get_text(strip=True)})
    return papers
