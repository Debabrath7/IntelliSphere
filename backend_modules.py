# ==============================================================
# backend_modules.py
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

# Optional sentiment pipeline - best-effort (non-fatal)
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
# DEMO BASE PRICES (editable)
# -------------------------
BASE_PRICES = {
    "RELIANCE": 2605, "TCS": 3106, "INFY": 1501, "HDFCBANK": 1530,
    "ICICIBANK": 1048, "SBIN": 971, "KOTAKBANK": 1880, "ITC": 407,
    "ASIANPAINT": 2980, "LT": 4011, "HCLTECH": 1702, "WIPRO": 482,
    "BAJAJFINSV": 17350, "HINDUNILVR": 2599, "MARUTI": 12350,
    "TECHM": 1350, "POWERGRID": 326, "AXISBANK": 1240,
    "ULTRACEMCO": 10200, "BHARTIARTL": 1215
}

# -------------------------
# Utilities
# -------------------------
def clean_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        return ""
    s = ticker.strip().upper()
    for suf in [".NS", ".BO", ":NS", ":BO", " NS", " BO"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()

def resolve_ticker_candidates(symbol: str):
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        # numeric BSE code
        return [f"{s}.BO", s]
    return [f"{s}.NS", f"{s}.BO", s]

def safe_get(url, params=None, headers=None, timeout=8):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r
        return None
    except Exception:
        return None

# -------------------------
# Robust normalizer (no KeyError / TypeError)
# -------------------------
def _normalize_df(df):
    """
    Input: a pandas.DataFrame (or similar)
    Output: cleaned DataFrame with columns: Date, Open, High, Low, Close, Volume (or None)
    This function will return None if it cannot produce a valid timeseries.
    """
    # reject non-dataframes early
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        return None
    if df.empty:
        return None

    df = df.copy()

    # If columns are nested dicts / lists, drop them
    for col in df.columns:
        # If the column contains dict/list values, replace whole column with NaN to avoid to_numeric errors
        try:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = np.nan
        except Exception:
            # in case apply fails for weird types
            df[col] = np.nan

    # Try to locate Date-like column
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        else:
            # Some APIs return timestamps in first column or index; attempt to find a date-like column
            found = False
            for cand in ["date", "dt", "timestamp", "time"]:
                if cand in [c.lower() for c in df.columns]:
                    real = [c for c in df.columns if c.lower() == cand][0]
                    df.rename(columns={real: "Date"}, inplace=True)
                    found = True
                    break
            if not found:
                # fallback: reset index to Date
                df.reset_index(inplace=True)
                if "index" in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)

    # Final attempt: coerce to datetime
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        # create artificial date index if conversion fails
        df["Date"] = pd.NaT

    # If all NaT, create business-day range
    if "Date" in df.columns and df["Date"].isna().all():
        try:
            df["Date"] = pd.bdate_range(end=datetime.utcnow().date(), periods=len(df))
        except Exception:
            df["Date"] = pd.date_range(end=datetime.utcnow().date(), periods=len(df))

    # Ensure OHLCV exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # Convert numeric columns safely (skip if content weird)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        try:
            # if column contains strings like '--' or empty, to_numeric handles them with errors='coerce'
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = np.nan

    # drop rows with invalid Date
    if "Date" in df.columns:
        df = df.dropna(subset=["Date"])
    else:
        return None

    # reorder and ensure columns present
    try:
        out = df.loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        # last resort - create empty structure
        out = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    # If no valid numeric rows, return None
    if out.empty:
        return None

    # Reset index and ensure sorted by date ascending
    out = out.sort_values("Date").reset_index(drop=True)
    return out

# -------------------------
# yfinance wrapper
# -------------------------
def _yf_download(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return None
        return df.reset_index()
    except Exception:
        # retry once
        try:
            time.sleep(0.6)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
            if df is None or df.empty:
                return None
            return df.reset_index()
        except Exception:
            return None

# -------------------------
# Yahoo CSV fallback
# -------------------------
_PERIOD_DAYS = {"1d":1,"5d":5,"1mo":30,"3mo":90,"6mo":182,"1y":365,"2y":365*2,"5y":365*5}
def _period_to_days(period):
    if isinstance(period,int):
        return period
    p = str(period).lower()
    if p in _PERIOD_DAYS:
        return _PERIOD_DAYS[p]
    if p.endswith("mo") or p.endswith("m"):
        try: return int(p.rstrip("mo").rstrip("m"))*30
        except: pass
    if p.endswith("y"):
        try: return int(p.rstrip("y"))*365
        except: pass
    return 365

def _yahoo_csv_download(ticker, period="6mo", interval="1d"):
    try:
        days = _period_to_days(period)
        end = int(datetime.utcnow().timestamp())
        start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        params = {"period1": start, "period2": end, "interval": interval, "events":"history","includeAdjustedClose":"true"}
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        text = r.text
        if not text or "404 Not Found" in text:
            return None
        try:
            df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
            return df
        except Exception:
            return None
    except Exception:
        return None

# -------------------------
# MoneyControl helpers (best-effort)
# -------------------------
def _moneycontrol_autosuggest(q: str):
    try:
        url = "https://www.moneycontrol.com/mccode/common/autosuggest.php"
        r = requests.get(url, params={"query": q}, timeout=8)
        if r.status_code != 200:
            return []
        text = r.text.strip()
        # sometimes JSON wrapped in callback
        try:
            data = json.loads(text)
        except Exception:
            try:
                start = text.find("(")
                end = text.rfind(")")
                if start!=-1 and end!=-1:
                    data = json.loads(text[start+1:end])
                else:
                    data = []
            except Exception:
                data = []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return []
    except Exception:
        return []

def _moneycontrol_historical_by_code(scode: str, period="6mo", interval="1d"):
    try:
        if not scode:
            return None
        candidates = [
            f"https://priceapi.moneycontrol.com/pricefeed/mcfeed/stock/{scode}",
            f"https://priceapi.moneycontrol.com/pricefeed/historical/{scode}?Period={period}"
        ]
        for url in candidates:
            try:
                r = requests.get(url, timeout=8)
                if r.status_code != 200:
                    continue
                # try JSON parse
                try:
                    data = r.json()
                    # search for list-of-dicts with date-like keys
                    def find_list(d):
                        if isinstance(d, list) and d and isinstance(d[0], dict):
                            keys = set(k.lower() for k in d[0].keys())
                            if any(k in keys for k in ("date","dt","timestamp")):
                                return d
                        if isinstance(d, dict):
                            for v in d.values():
                                res = find_list(v)
                                if res:
                                    return res
                        return None
                    arr = find_list(data)
                    if arr:
                        df = pd.DataFrame(arr)
                        # map common names
                        mapping={}
                        for col in df.columns:
                            low = col.lower()
                            if "open" in low: mapping[col]="Open"
                            if "high" in low: mapping[col]="High"
                            if "low" in low: mapping[col]="Low"
                            if "close" in low or "ltp" in low: mapping[col]="Close"
                            if "volume" in low: mapping[col]="Volume"
                            if "date" in low or "dt" in low or "timestamp" in low: mapping[col]="Date"
                        df.rename(columns=mapping, inplace=True)
                        return _normalize_df(df)
                except Exception:
                    pass
                # try CSV/html table
                text = r.text
                if "<table" in text.lower():
                    try:
                        df = pd.read_html(text)[0]
                        return _normalize_df(df)
                    except Exception:
                        pass
                # try CSV style
                if "\n" in text and "," in text and "Date" in text[:200]:
                    try:
                        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
                        return _normalize_df(df)
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        pass
    return None

def resolve_moneycontrol_code(symbol: str):
    s = clean_ticker(symbol)
    if not s:
        return None
    # autosuggest
    try:
        results = _moneycontrol_autosuggest(s)
        for it in results:
            if isinstance(it, dict):
                code = it.get("scode") or it.get("scid") or it.get("id") or it.get("sCode")
                if code:
                    return str(code)
                # fallback: try url based id
                url = it.get("url") or it.get("link")
                if isinstance(url, str) and url.strip():
                    return url.rstrip("/").split("/")[-1]
    except Exception:
        pass
    return None

# -------------------------
# Demo synthetic generator
# -------------------------
def _seed_from_string(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31)

def _generate_price_series(seed:int, start_price:float, days:int, mu=0.0002, sigma=0.02):
    rng = np.random.RandomState(seed)
    prices = [start_price]
    for _ in range(days-1):
        shock = rng.normal(loc=mu, scale=sigma)
        prices.append(max(0.01, prices[-1] * (1 + shock)))
    return np.array(prices)

def _build_demo_daily_df(symbol: str, years: int = 5):
    s = clean_ticker(symbol)
    seed = _seed_from_string(s)
    days = years * 252
    baseline = BASE_PRICES.get(s, 500.0)
    start_price = baseline * 0.75
    closes = _generate_price_series(seed+1, start_price, days, mu=0.0003, sigma=0.015)
    # scale so last matches baseline
    if baseline:
        factor = baseline / float(closes[-1])
        closes = closes * factor
    dates = pd.bdate_range(end=datetime.utcnow().date(), periods=days)
    rng = np.random.RandomState(seed+2)
    opens = closes * (1 + rng.normal(0, 0.002, size=len(closes)))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.004, size=len(closes))))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.004, size=len(closes))))
    volumes = (rng.normal(loc=2e6, scale=6e5, size=len(closes))).astype(int)
    volumes = np.where(volumes < 1000, 1000, volumes)
    df = pd.DataFrame({"Date": dates, "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes})
    return _normalize_df(df)

# -------------------------
# Master resolution pipeline
# -------------------------
def get_best_ticker(symbol: str, period="6mo", interval="1d"):
    """
    Returns (ticker_string, DataFrame) chosen from sources.
    Tries: yfinance -> Yahoo CSV -> MoneyControl -> DEMO
    """
    if not symbol:
        return None, None
    candidates = resolve_ticker_candidates(symbol)

    # 1) yfinance (fast & primary)
    for t in candidates:
        try:
            df = _yf_download(t, period, interval)
            df = _normalize_df(df)
            if df is not None:
                return t, df
        except Exception:
            continue

    # 2) yahoo csv fallback
    for t in candidates:
        try:
            df = _yahoo_csv_download(t, period, interval)
            df = _normalize_df(df)
            if df is not None:
                return t, df
        except Exception:
            continue

    # 3) moneycontrol by resolved code
    try:
        mc = resolve_moneycontrol_code(symbol)
        if mc:
            df = _moneycontrol_historical_by_code(mc, period, interval)
            df = _normalize_df(df)
            if df is not None:
                return mc, df
    except Exception:
        pass

    # 4) Demo fallback
    try:
        demo = _build_demo_daily_df(symbol, years=5)
        if demo is not None:
            return f"DEMO:{clean_ticker(symbol)}", demo
    except Exception:
        pass

    return None, None

# -------------------------
# Public frontend API
# -------------------------
def get_stock_data(symbol: str, period="6mo", interval="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df

def get_intraday_data(symbol: str, interval="1m", period="1d"):
    _, df = get_best_ticker(symbol, period, interval)
    return df

def get_today_high_low(symbol: str, prefer_intraday=True):
    t, df = get_best_ticker(symbol, "1d", "1m" if prefer_intraday else "1d")
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["Close"])
    if d.empty:
        return None
    return {"high": float(d["High"].max()), "low": float(d["Low"].min()), "timestamp": d["Date"].iloc[-1]}

@lru_cache(maxsize=256)
def stock_summary(symbol: str, period="6mo", interval="1d"):
    t, df = get_best_ticker(symbol, period, interval)
    if df is None or df.empty:
        return None
    try:
        first = float(df["Close"].dropna().iloc[0])
        last = float(df["Close"].dropna().iloc[-1])
        pct = (last - first) / (first + 1e-9) * 100
        high = float(df["High"].max())
        low = float(df["Low"].min())
        return {"ticker": t, "first_close": first, "latest_close": last, "pct_change": pct, "high": high, "low": low}
    except Exception:
        return None

# -------------------------
# News & sentiment (safe)
# -------------------------
import feedparser
def fetch_news_via_google_rss(query: str, max_items: int = 10):
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_items]:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pub_date = datetime.utcnow()
            articles.append({"title": entry.title, "link": entry.link, "published": pub_date})
        return articles
    except Exception:
        return []

def analyze_headlines_sentiment(articles):
    if not sentiment_analyzer:
        return [{**a, "sentiment": None} for a in articles]
    out = []
    for a in articles:
        try:
            s = sentiment_analyzer(a.get("title","")[:512])[0]
            out.append({**a, "sentiment": {"label": s.get("label"), "score": float(s.get("score", 0.0))}})
        except Exception:
            out.append({**a, "sentiment": None})
    return out

# -------------------------
# Lightweight github / arxiv helpers
# -------------------------
def fetch_github_trending(language=None, since="daily"):
    try:
        url = f"https://github.com/trending/{language}" if language else "https://github.com/trending"
        r = safe_get(url, timeout=8)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        repos = []
        for repo in soup.select("article.Box-row")[:20]:
            title_tag = repo.find(["h1","h2"])
            link_tag = title_tag.find("a") if title_tag else None
            name = link_tag.get("href","").strip("/") if link_tag else "unknown"
            desc_tag = repo.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            repos.append({"name": name, "description": desc})
        return repos
    except Exception:
        return []

def fetch_arxiv_papers(query, max_results=5):
    try:
        base = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
        r = requests.get(base, params=params, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "xml")
        papers = []
        for entry in soup.find_all("entry")[:max_results]:
            papers.append({"title": entry.title.get_text(strip=True), "summary": entry.summary.get_text(strip=True), "link": entry.id.get_text(strip=True)})
        return papers
    except Exception:
        return []

# -------------------------
# End of file
# -------------------------
