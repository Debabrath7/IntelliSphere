# ==============================================================
# backend_modules.py
# Multi-source stock fetcher (NSE/BSE) with DEMO fallback
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import time
import io
import json
import math
import hashlib
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
from bs4 import BeautifulSoup
import feedparser

# --------------------------------------------------------------
# OPTIONAL: Sentiment Pipeline
# --------------------------------------------------------------

try:
    from transformers import pipeline
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
    except:
        sentiment_analyzer = pipeline("sentiment-analysis")
except:
    sentiment_analyzer = None


# --------------------------------------------------------------
# DEMO — Updated Baseline Prices (Realistic Approx from 2025)
# You provided: SBIN 971, INFY 1501, ITC 407, LT 4011
# I fetched: TCS 3106, RELIANCE 1519
# --------------------------------------------------------------

DEMO_BASELINE = {
    "RELIANCE": 1519,
    "TCS": 3106,
    "INFY": 1501,
    "HDFCBANK": 1530,     # approx live
    "ICICIBANK": 1060,    # approx live
    "SBIN": 971,          # YOU GAVE
    "KOTAKBANK": 1765,    # approx
    "ASIANPAINT": 2920,   # approx
    "ITC": 407,           # YOU GAVE
    "LT": 4011,           # YOU GAVE
    "BHARTIARTL": 1210,   # approx
    "ULTRACEMCO": 9810,   # approx
    "WIPRO": 475,         # approx
    "HCLTECH": 1650,      # approx
    "MARUTI": 12100,      # approx
    "TECHM": 1330,        # approx
    "HINDUNILVR": 2550,   # approx
    "AXISBANK": 1240,     # approx
    "BAJAJFINSV": 16500,  # approx
    "POWERGRID": 326      # approx
}


# --------------------------------------------------------------
# TICKER CLEANING
# --------------------------------------------------------------

def clean_ticker(t):
    if not t:
        return ""
    t = t.strip().upper()
    for s in [".NS", ".BO", ":NS", ":BO"]:
        t = t.replace(s, "")
    return t


def resolve_ticker_candidates(symbol):
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        return [f"{s}.BO", s]  # BSE numeric
    return [f"{s}.NS", f"{s}.BO", s]


# --------------------------------------------------------------
# NORMALIZE DATAFRAME
# --------------------------------------------------------------

def _normalize_df(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()

    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Date"])

    return df if not df.empty else None


# --------------------------------------------------------------
# YFINANCE (LIVE)
# --------------------------------------------------------------

def _yf_download(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            threads=True,
            auto_adjust=False
        )
        if df is None or df.empty:
            return None
        return df.reset_index()
    except:
        return None


# --------------------------------------------------------------
# YAHOO CSV (FALLBACK)
# --------------------------------------------------------------

def _period_days(period):
    mp = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    if period in mp:
        return mp[period]
    return 365


def _yahoo_csv_download(ticker, period="6mo", interval="1d"):
    try:
        days = _period_days(period)
        end = int(time.time())
        start = end - days * 86400

        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            "period1": start,
            "period2": end,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true"
        }

        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None

        df = pd.read_csv(io.StringIO(r.text))
        return df
    except:
        return None


# --------------------------------------------------------------
# MONEYCONTROL SCRAPER (BEST-EFFORT)
# --------------------------------------------------------------

def _moneycontrol_autosuggest(q):
    try:
        url = "https://www.moneycontrol.com/mccode/common/autosuggest.php"
        r = requests.get(url, params={"query": q}, timeout=8)
        if r.status_code != 200:
            return []
        txt = r.text.strip()
        try:
            return json.loads(txt)
        except:
            return []
    except:
        return []


def _moneycontrol_historical(code, period="6mo"):
    if not code:
        return None

    try_urls = [
        f"https://priceapi.moneycontrol.com/pricefeed/bse/equitycash/{code}",
        f"https://priceapi.moneycontrol.com/pricefeed/historical/{code}?period={period}",
    ]

    for url in try_urls:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue

            # Try JSON first
            try:
                j = r.json()
                if isinstance(j, dict):
                    for v in j.values():
                        if isinstance(v, list) and v:
                            df = pd.DataFrame(v)
                            df.rename(columns={col: "Date" for col in df.columns if col.lower() in ("date", "dt")}, inplace=True)
                            df.rename(columns={col: col.title() for col in df.columns}, inplace=True)
                            return _normalize_df(df)
            except:
                pass

            # Try CSV/HTML tables fallback
            try:
                df = pd.read_html(r.text)[0]
                return _normalize_df(df)
            except:
                pass

        except:
            continue

    return None


def resolve_moneycontrol_code(symbol):
    q = clean_ticker(symbol)
    if not q:
        return None

    res = _moneycontrol_autosuggest(q)
    for item in res:
        if isinstance(item, dict):
            code = item.get("sc_id") or item.get("scid") or item.get("id") or item.get("sCode")
            if code:
                return str(code)

    return None


# --------------------------------------------------------------
# DEMO MODE (LAST RESORT)
# --------------------------------------------------------------

def _seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def _build_demo_daily(symbol, years=5):
    symbol = clean_ticker(symbol)
    base = DEMO_BASELINE.get(symbol, 500)

    seed = _seed(symbol)
    rng = np.random.RandomState(seed)

    days = years * 250
    prices = [base * 0.8]

    for _ in range(days - 1):
        shock = rng.normal(0, 0.012)
        prices.append(max(10, prices[-1] * (1 + shock)))

    factor = base / prices[-1]
    prices = np.array(prices) * factor

    dates = pd.bdate_range(end=datetime.today(), periods=days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": prices * 0.998,
        "High": prices * 1.004,
        "Low": prices * 0.996,
        "Close": prices,
        "Volume": rng.randint(100000, 5000000, len(prices))
    })

    return _normalize_df(df)


def _demo(symbol, period):
    df = _build_demo_daily(symbol)
    days = _period_days(period)
    return df.tail(days)


# --------------------------------------------------------------
# MASTER RESOLUTION PIPELINE
# --------------------------------------------------------------

def get_best_ticker(symbol, period="6mo", interval="1d"):

    # 1) Live yfinance
    for t in resolve_ticker_candidates(symbol):
        df = _yf_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None:
            return t, df

    # 2) Yahoo CSV
    for t in resolve_ticker_candidates(symbol):
        df = _yahoo_csv_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None:
            return t, df

    # 3) MoneyControl
    code = resolve_moneycontrol_code(symbol)
    if code:
        df = _moneycontrol_historical(code, period)
        df = _normalize_df(df)
        if df is not None:
            return code, df

    # 4) Demo synthetic
    return "DEMO:" + clean_ticker(symbol), _demo(symbol, period)


# --------------------------------------------------------------
# PUBLIC FRONTEND API CALLS
# --------------------------------------------------------------

def get_stock_data(symbol, period="6mo", interval="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df


def get_intraday_data(symbol, interval="1m", period="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df


def get_today_high_low(symbol):
    _, df = get_best_ticker(symbol, "1d", "1m")
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["Close"])
    if d.empty:
        return None
    return {
        "high": float(d["High"].max()),
        "low": float(d["Low"].min()),
        "timestamp": d["Date"].iloc[-1]
    }


@lru_cache(maxsize=64)
def stock_summary(symbol, period="6mo"):
    t, df = get_best_ticker(symbol, period, "1d")
    if df is None:
        return None
    try:
        first = df["Close"].dropna().iloc[0]
        last = df["Close"].dropna().iloc[-1]
        pct = (last - first) / first * 100
        return {
            "ticker": t,
            "first_close": first,
            "latest_close": last,
            "pct_change": pct
        }
    except:
        return None


# --------------------------------------------------------------
# NEWS / SENTIMENT
# --------------------------------------------------------------

def fetch_news_via_google_rss(query, max_items=10):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:max_items]:
        try:
            pub = datetime(*e.published_parsed[:6])
        except:
            pub = datetime.utcnow()
        out.append({
            "title": e.title,
            "link": e.link,
            "published": pub
        })
    return out


def analyze_headlines_sentiment(articles):
    if sentiment_analyzer is None:
        return [{**a, "sentiment": None} for a in articles]

    out = []
    for a in articles:
        try:
            s = sentiment_analyzer(a["title"][:400])[0]
            out.append({**a, "sentiment": s})
        except:
            out.append({**a, "sentiment": None})
    return out


# --------------------------------------------------------------
# GITHUB / ARXIV HELPERS
# --------------------------------------------------------------

def fetch_github_trending(language=None, since="daily"):
    try:
        url = "https://github.com/trending/" + (language or "")
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for repo in soup.select("article.Box-row")[:15]:
            title = repo.find("h2")
            desc = repo.find("p")
            out.append({
                "name": title.get_text(strip=True) if title else "",
                "description": desc.get_text(strip=True) if desc else ""
            })
        return out
    except:
        return []


def fetch_arxiv_papers(query, max_results=5):
    try:
        base = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
        r = requests.get(base, params=params, timeout=10)
        soup = BeautifulSoup(r.text, "xml")
        out = []
        for entry in soup.find_all("entry")[:max_results]:
            out.append({
                "title": entry.title.text.strip(),
                "summary": entry.summary.text.strip(),
                "link": entry.id.text.strip()
            })
        return out
    except:
        return []
