# ==========================================================
# backend_modules.py 
# Author: Debabrath 
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from functools import lru_cache

# -------- Sentiment (optional) ----------
try:
    from transformers import pipeline
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except:
        sentiment_analyzer = pipeline("sentiment-analysis")
except:
    sentiment_analyzer = None


# ==========================================================
# CLEAN / NORMALIZE TICKER
# ==========================================================
def clean_ticker(ticker: str) -> str:
    """Normalize tickers: remove .NS/.BO and whitespace, keep upper-case."""
    if not isinstance(ticker, str):
        return ticker
    s = ticker.strip().upper()

    # remove suffixes
    for suf in [".NS", ".BO", ":NS", ":BO"]:
        if s.endswith(suf):
            s = s.replace(suf, "")
    return s.strip()


# ==========================================================
# GENERATE MULTIPLE POSSIBLE TICKER OPTIONS
# ==========================================================
def resolve_ticker_candidates(symbol: str):
    """
    Converts simple inputs into multiple valid possible NSE/BSE tickers.
    Example:
        "BDL" → ["BDL.NS", "BDL.BO", "BDL"]
        "TCS" → ["TCS.NS", "TCS.BO", "TCS"]
        "541143" → ["541143.BO", "541143"]
    """
    symbol = clean_ticker(symbol)

    candidates = []

    # If user enters numeric → BSE code (e.g., "541143")
    if symbol.isdigit():
        candidates = [f"{symbol}.BO", symbol]
        return candidates

    # Otherwise treat as NSE/BSE symbol
    candidates = [
        f"{symbol}.NS",
        f"{symbol}.BO",
        f"{symbol}"       # last fallback
    ]
    return candidates


# ==========================================================
# NORMALIZE DOWNLOADED DF
# ==========================================================
def _normalize_df(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()

    # Ensure Date column
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "Date"})

    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except:
        pass

    # Ensure OHLCV columns
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[["Date","Open","High","Low","Close","Volume"]]


# ==========================================================
# yfinance wrapper with retry
# ==========================================================
def _yf_download(ticker, period, interval):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if df is None or df.empty:
            return None
        return df.reset_index()
    except:
        # Retry once
        try:
            time.sleep(0.8)
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=True,
            )
            if df is None or df.empty:
                return None
            return df.reset_index()
        except:
            return None


# ==========================================================
# FIND BEST AVAILABLE TICKER
# ==========================================================
def get_best_ticker(symbol, period="1d", interval="1m"):
    """
    Tries all possible tickers (NS/BO/raw).
    Returns the FIRST working ticker + dataframe.
    """
    for t in resolve_ticker_candidates(symbol):
        df = _yf_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None and not df.empty:
            return t, df
    return None, None


# ==========================================================
# MAIN STOCK FETCH FUNCTIONS
# ==========================================================
def get_stock_data(symbol: str, period="6mo", interval="1d"):
    """
    Auto-resolves best working ticker from NSE/BSE.
    """
    ticker, df = get_best_ticker(symbol, period, interval)
    return df


def get_intraday_data(symbol: str, interval="1m", period="1d"):
    """
    Intraday version.
    """
    ticker, df = get_best_ticker(symbol, period, interval)
    return df


# ==========================================================
# TODAY'S HIGH / LOW
# ==========================================================
def get_today_high_low(symbol: str):
    ticker, df = get_best_ticker(symbol, "1d", "1m")
    if df is None or df.empty:
        return None
    df2 = df.dropna(subset=["Close"])
    if df2.empty:
        return None
    return {
        "high": float(df2["High"].max()),
        "low": float(df2["Low"].min()),
        "timestamp": df2["Date"].iloc[-1]
    }


# ==========================================================
# STOCK SUMMARY
# ==========================================================
@lru_cache(maxsize=128)
def stock_summary(symbol: str, period="6mo", interval="1d"):
    ticker, df = get_best_ticker(symbol, period, interval)
    if df is None or df.empty:
        return None
    try:
        first = float(df["Close"].iloc[0])
        last = float(df["Close"].iloc[-1])
        pct = (last-first)/(first+1e-9)*100
        return {
            "ticker": ticker,
            "first_close": first,
            "latest_close": last,
            "pct_change": pct,
            "recent": df.tail(30).to_dict("records")
        }
    except:
        return None


# ==========================================================
# NEWS + SENTIMENT (unchanged)
# ==========================================================
def fetch_news_via_google_rss(query, max_items=15):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    out=[]
    for e in feed.entries[:max_items]:
        try:
            pub = datetime(*e.published_parsed[:6])
        except:
            pub = datetime.now()
        out.append({"title": e.title, "link": e.link, "published": pub})
    return out

def analyze_headlines_sentiment(articles):
    if sentiment_analyzer is None:
        return [{**a,"sentiment": None} for a in articles]
    res=[]
    for a in articles:
        try:
            s = sentiment_analyzer(a["title"][:512])[0]
            res.append({**a,"sentiment":{"label": s["label"], "score": float(s["score"])}})
        except:
            res.append({**a,"sentiment": None})
    return res


# ==========================================================
# GITHUB / ARXIV (unchanged)
# ==========================================================
def fetch_github_trending(lang=None):
    url = f"https://github.com/trending/{lang}" if lang else "https://github.com/trending"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    out=[]
    for row in soup.select("article.Box-row")[:20]:
        a = row.find("a")
        if not a: continue
        name = a.get("href","").strip("/")
        desc_tag = row.find("p")
        desc = desc_tag.get_text(strip=True) if desc_tag else ""
        out.append({"name":name,"description":desc})
    return out

def fetch_arxiv_papers(q, max_results=5):
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{q}", "start":0,"max_results":max_results}
    r = requests.get(base, params=params)
    if r.status_code !=200:
        return []
    soup = BeautifulSoup(r.text,"xml")
    out=[]
    for e in soup.find_all("entry")[:max_results]:
        out.append({
            "title": e.title.get_text(strip=True),
            "summary": e.summary.get_text(strip=True),
            "link": e.id.get_text(strip=True),
        })
    return out
