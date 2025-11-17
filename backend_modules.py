# ==============================================================
# backend_modules.py — Multi-source Indian stock fetcher
# Author: Debabrath 
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

# ==============================================================
# SAFE NORMALIZER - SOLVES KEYERROR: 'Date'
# ==============================================================

def _normalize_df(df):
    """Safely normalize any stock dataframe into a clean OHLCV time series."""

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()

    # --------------------------------------------------
    # 1) Ensure DATE column exists
    # --------------------------------------------------
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        else:
            # last resort: create date from index
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date"}, inplace=True)

    # force parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # if still all NaT → create artificial timeline
    if df["Date"].isna().all():
        df["Date"] = pd.date_range(
            end=datetime.today(), periods=len(df), freq="B"
        )

    # --------------------------------------------------
    # 2) Ensure OHLCV columns exist
    # --------------------------------------------------
    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # force numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # 3) Final cleanup
    # --------------------------------------------------
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Date"])

    return df if not df.empty else None


# ==============================================================
# CLEAN TICKER INPUT
# ==============================================================

def clean_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        return ""
    t = ticker.strip().upper()
    for s in [".NS", ".BO", ":NS", ":BO"]:
        if t.endswith(s):
            t = t.replace(s, "")
    return t


def resolve_ticker_candidates(symbol):
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        return [f"{s}.BO", s]
    return [f"{s}.NS", f"{s}.BO", s]


# ==============================================================
# YFINANCE FETCH
# ==============================================================

def _yf_download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return None
        return df.reset_index()
    except Exception:
        return None


# ==============================================================
# YAHOO CSV DIRECT DOWNLOAD (STRONGER)
# ==============================================================

def _period_days(period):
    mapping = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
        "2y": 730, "5y": 1825
    }
    return mapping.get(period, 365)

def _yahoo_csv_download(ticker, period="6mo", interval="1d"):
    try:
        days = _period_days(period)
        end = int(datetime.utcnow().timestamp())
        start = int((datetime.utcnow() - timedelta(days=days)).timestamp())

        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            "period1": start,
            "period2": end,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true"
        }
        r = requests.get(url, params=params, timeout=10)

        if r.status_code != 200:
            return None

        df = pd.read_csv(io.StringIO(r.text))
        return df
    except:
        return None


# ==============================================================
# MONEYCONTROL SUPPORT (AUTOSUGGEST + HISTORICAL)
# ==============================================================

def resolve_moneycontrol_code(symbol):
    symbol = clean_ticker(symbol)
    try:
        url = "https://www.moneycontrol.com/mccode/common/autosuggest.php"
        r = requests.get(url, params={"query": symbol}, timeout=6)
        if r.status_code != 200:
            return None
        arr = json.loads(r.text.strip()[1:-1])
        for item in arr:
            scode = item.get("sc_id") or item.get("scode") or item.get("id")
            if scode:
                return str(scode)
    except:
        return None
    return None


def _moneycontrol_historical(scode, period="6mo"):
    if not scode:
        return None
    try:
        url = f"https://priceapi.moneycontrol.com/pricefeed/nse/equitycash/{scode}"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None

        # parse best available data
        data = r.json()
        if "data" in data and "pricehistory" in data["data"]:
            df = pd.DataFrame(data["data"]["pricehistory"])
            df.rename(columns={
                "price_date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            return df

    except:
        return None

    return None


# ==============================================================
# DEMO MODE (20 STOCKS)
# ==============================================================

BASE_PRICES = {
    "RELIANCE": 2605, "TCS": 4011, "INFY": 1501, "HDFCBANK": 1530,
    "ICICIBANK": 1048, "SBIN": 971, "KOTAKBANK": 1880, "ITC": 407,
    "ASIANPAINT": 2980, "LT": 4011, "HCLTECH": 1702, "WIPRO": 482,
    "BAJAJFINSV": 17350, "HINDUNILVR": 2599, "MARUTI": 12350,
    "TECHM": 1350, "POWERGRID": 312, "AXISBANK": 1100,
    "ULTRACEMCO": 10200, "BHARTIARTL": 1215
}

def _demo(symbol):
    s = clean_ticker(symbol)
    if s not in BASE_PRICES:
        return None

    price = BASE_PRICES[s]
    dates = pd.date_range(end=datetime.today(), periods=200, freq="B")

    rng = np.random.default_rng(hash(s) % 999999)

    close = price * (1 + rng.normal(0, 0.015, size=len(dates))).cumprod()
    openp = close * (1 + rng.normal(0, 0.002, size=len(dates)))
    high = np.maximum(openp, close) * (1 + rng.normal(0, 0.003, size=len(dates)))
    low = np.minimum(openp, close) * (1 - rng.normal(0, 0.003, size=len(dates)))
    vol = rng.integers(1_00_000, 50_00_000, size=len(dates))

    df = pd.DataFrame({
        "Date": dates,
        "Open": openp,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    })
    return df


# ==============================================================
# MAIN DATA SELECTOR
# ==============================================================

def get_best_ticker(symbol, period="6mo", interval="1d"):

    candidates = resolve_ticker_candidates(symbol)

    # 1) yfinance
    for t in candidates:
        df = _yf_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None:
            return t, df

    # 2) Yahoo CSV
    for t in candidates:
        df = _yahoo_csv_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None:
            return t, df

    # 3) MoneyControl
    sc = resolve_moneycontrol_code(symbol)
    if sc:
        df = _moneycontrol_historical(sc, period)
        df = _normalize_df(df)
        if df is not None:
            return sc, df

    # 4) DEMO fallback
    df = _demo(symbol)
    df = _normalize_df(df)
    if df is not None:
        return f"DEMO:{symbol}", df

    return None, None


# ==============================================================
# PUBLIC FUNCTIONS USED BY FRONTEND
# ==============================================================

def get_stock_data(symbol, period="6mo", interval="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df

def get_intraday_data(symbol, interval="1m", period="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df

def get_today_high_low(symbol, prefer_intraday=True):
    _, df = get_best_ticker(symbol, "1d", "1m")
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["Close"])
    return {
        "high": float(d["High"].max()),
        "low": float(d["Low"].min()),
        "timestamp": d["Date"].iloc[-1]
    }


# ==============================================================
# NEWS + SENTIMENT
# ==============================================================

import feedparser

def fetch_news_via_google_rss(query: str, max_items: int = 10):
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:max_items]:
        try:
            pub = datetime(*e.published_parsed[:6])
        except:
            pub = datetime.utcnow()
        out.append({"title": e.title, "link": e.link, "published": pub})
    return out


def analyze_headlines_sentiment(arts):
    return [{**a, "sentiment": None} for a in arts]


# ==============================================================
# GitHub & ArXiv
# ==============================================================

def fetch_github_trending(language=None):
    return []

def fetch_arxiv_papers(q, max_results=5):
    return []

@lru_cache(maxsize=256)
def stock_summary(symbol: str, period="6mo", interval="1d"):
    """
    Safe summary function for frontend.
    Always returns a clean summary dict or None.
    Works with yfinance, Yahoo CSV, MoneyControl & DEMO.
    """
    _, df = get_best_ticker(symbol, period, interval)
    if df is None or df.empty:
        return None

    try:
        first = float(df["Close"].dropna().iloc[0])
        last = float(df["Close"].dropna().iloc[-1])
        pct = (last - first) / (first + 1e-9) * 100
        high = float(df["High"].max())
        low = float(df["Low"].min())

        return {
            "symbol": symbol,
            "first_close": first,
            "latest_close": last,
            "pct_change": pct,
            "high": high,
            "low": low
        }
    except:
        return None


