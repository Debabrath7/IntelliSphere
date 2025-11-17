# backend_modules.py
# Robust, multi-source stock data fetcher for Indian stocks
# Tries: yfinance -> Yahoo CSV -> MoneyControl (autosuggest + price endpoints)
# Exposes: get_stock_data, get_intraday_data, get_today_high_low, stock_summary, etc.
# Author: Debabrath

import warnings
warnings.filterwarnings("ignore")

import time
import io
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
from bs4 import BeautifulSoup

# Optional sentiment model (best-effort)
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
    """Normalize tickers: uppercase, remove common suffixes"""
    if not isinstance(ticker, str):
        return ticker
    s = ticker.strip().upper()
    for suf in [".NS", ".BO", ":NS", ":BO"]:
        if s.endswith(suf):
            s = s.replace(suf, "")
    return s.strip()

def resolve_ticker_candidates(symbol: str):
    """
    Return candidate tickers that we will try with yfinance/Yahoo CSV.
    We still rely on MoneyControl for robust mapping if these fail.
    """
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        return [f"{s}.BO", s]
    # try NSE, BSE, raw
    return [f"{s}.NS", f"{s}.BO", s]

# -------------------------
# Normalize DF
# -------------------------
def _normalize_df(df):
    """Return DataFrame with Date,Open,High,Low,Close,Volume or None."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.copy()
    # Ensure Date column
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass
    # Ensure OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    # Keep only the required columns
    out = df.loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Date"])
    if out.empty:
        return None
    return out

# -------------------------
# yfinance wrapper with retry
# -------------------------
def _yf_download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return None
        return df.reset_index()
    except Exception:
        try:
            time.sleep(0.6)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
            if df is None or df.empty:
                return None
            return df.reset_index()
        except Exception:
            return None

# -------------------------
# Yahoo CSV fallback (direct HTTP)
# -------------------------
_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 365*2, "5y": 365*5, "10y": 365*10
}
def _period_to_days(period):
    if isinstance(period, int):
        return period
    p = str(period).lower()
    if p in _PERIOD_DAYS:
        return _PERIOD_DAYS[p]
    if p.endswith("mo") or p.endswith("m"):
        try: return int(p.rstrip("mo").rstrip("m")) * 30
        except: pass
    if p.endswith("y"):
        try: return int(p.rstrip("y")) * 365
        except: pass
    return 365

def _yahoo_csv_download(ticker, period="6mo", interval="1d"):
    try:
        days = _period_to_days(period)
        end = int(datetime.utcnow().timestamp())
        start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        params = {"period1": start, "period2": end, "interval": interval, "events": "history", "includeAdjustedClose": "true"}
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        text = r.text
        if not text or "404 Not Found" in text:
            return None
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
        if df.empty:
            return None
        # Ensure numeric OHLCV
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan
        return df.reset_index(drop=True)
    except Exception:
        return None

# -------------------------
# MoneyControl: autosuggest + historical fetch
# -------------------------
def _moneycontrol_autosuggest(query: str):
    """
    Use MoneyControl autosuggest endpoint to find matching entries.
    Returns list of result dicts (may include 'scode' or 'sc_id' or 'url' slugs).
    """
    try:
        q = str(query).strip()
        if not q:
            return []
        url = "https://www.moneycontrol.com/mccode/common/autosuggest.php"
        r = requests.get(url, params={"query": q}, timeout=8)
        if r.status_code != 200:
            return []
        text = r.text.strip()
        # Response may be JSON or JSONP. Try to extract JSON.
        try:
            data = json.loads(text)
        except Exception:
            # strip callback(...) if present
            try:
                start = text.find("(")
                end = text.rfind(")")
                if start != -1 and end != -1:
                    body = text[start+1:end]
                    data = json.loads(body)
                else:
                    data = []
            except Exception:
                data = []
        # Data may be string-keyed dict or list
        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            items = []
        return items
    except Exception:
        return []

def _moneycontrol_historical_by_code(scode: str, period="6mo", interval="1d"):
    """
    Attempt to retrieve historical OHLC data for a moneycontrol code.
    Tries known MoneyControl endpoints and simple HTML table parsing as a fallback.
    Returns normalized DataFrame or None.
    """
    try:
        if not scode:
            return None
        # Try common MoneyControl endpoints known to work in many cases.
        #  - get_price endpoint (returns CSV / json in several cases)
        #  - pricefeed endpoints
        period_str = period
        candidates = [
            f"https://www.moneycontrol.com/stocks/histories/get_price/{scode}?period={period_str}",
            f"https://priceapi.moneycontrol.com/pricefeed/mcfeed/stock/{scode}",
            f"https://priceapi.moneycontrol.com/pricefeed/historical/{scode}?Period={period_str}"
        ]
        for url in candidates:
            try:
                r = requests.get(url, timeout=8)
                if r.status_code != 200:
                    continue
                text = r.text
                # If JSON response with time-series
                try:
                    data = r.json()
                    # Try to parse common structures
                    # pricefeed/mcfeed/stock/{scode} returns nested dict with 'data' etc.
                    if isinstance(data, dict):
                        # Common keys with historical series might be under 'data' or 'series' or 'history'
                        # Try to extract arrays for dates/prices
                        # Heuristics: look for arrays of dicts with date/open/high/low/close/volume
                        def extract_from_dict(d):
                            if not isinstance(d, dict):
                                return None
                            # search recursively for list of dicts with 'date' and 'close' or 'closePrice'
                            for k, v in d.items():
                                if isinstance(v, list) and v and isinstance(v[0], dict):
                                    if any(key.lower() in ("close","closeprice","close_price","closeValue") for key in v[0].keys()):
                                        return v
                                elif isinstance(v, dict):
                                    res = extract_from_dict(v)
                                    if res:
                                        return res
                            return None
                        arr = extract_from_dict(data)
                        if arr:
                            # convert to DataFrame carefully
                            df = pd.DataFrame(arr)
                            # try to rename typical fields
                            for cand in ["date", "Date", "dt", "timestamp"]:
                                if cand in df.columns:
                                    df = df.rename(columns={cand: "Date"})
                                    break
                            # Normalize OHLCV columns
                            mapping = {}
                            for col in df.columns:
                                low = col.lower()
                                if "open" in low: mapping[col] = "Open"
                                if "high" in low: mapping[col] = "High"
                                if "low" in low: mapping[col] = "Low"
                                if "close" in low: mapping[col] = "Close"
                                if "volume" in low: mapping[col] = "Volume"
                            df = df.rename(columns=mapping)
                            if "Date" in df.columns:
                                # coerce numeric date if needed
                                try:
                                    df["Date"] = pd.to_datetime(df["Date"], unit='s')
                                except Exception:
                                    try:
                                        df["Date"] = pd.to_datetime(df["Date"])
                                    except:
                                        pass
                            df_norm = _normalize_df(df)
                            if df_norm is not None:
                                return df_norm
                    # if not JSON with series, continue to HTML parsing
                except Exception:
                    pass

                # If text contains CSV-like lines, attempt CSV parse
                if "\n" in text and "," in text and "Date" in text[:200]:
                    try:
                        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
                        df_norm = _normalize_df(df)
                        if df_norm is not None:
                            return df_norm
                    except Exception:
                        pass

                # Fallback: parse HTML tables on page
                try:
                    soup = BeautifulSoup(text, "html.parser")
                    table = soup.find("table")
                    if table:
                        df_list = pd.read_html(str(table))
                        if df_list:
                            df = df_list[0]
                            # try to find Date/Close etc.
                            df_norm = _normalize_df(df)
                            if df_norm is not None:
                                return df_norm
                except Exception:
                    pass

            except Exception:
                continue
    except Exception:
        pass
    return None

# -------------------------
# Build MoneyControl master list (cached)
# -------------------------
@lru_cache(maxsize=1)
def build_moneycontrol_master():
    """
    Attempt to create a mapping of many Indian stocks to MoneyControl codes.
    Strategy:
      1. Try priceapi index endpoints (if available)
      2. Use autosuggest for frequent symbols (on-demand)
      3. If available, use a public symbol listing endpoint
    Returns: dict with keys: symbol -> best_match_code (string)
    """
    master = {}
    # Try a few known MoneyControl index endpoints that sometimes include listings
    endpoints = [
        "https://priceapi.moneycontrol.com/pricefeed/bse/equitycash",
        "https://priceapi.moneycontrol.com/pricefeed/nse/equitycash",
        "https://priceapi.moneycontrol.com/pricefeed/stocklist"  # best-effort
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            # data is probably dict containing many entries; try to extract mapping
            # Search recursively for dicts that have 'code' or 'sc_id' and 'symbol' or 'name'
            def walk(d):
                if isinstance(d, dict):
                    keys = set(d.keys())
                    if ("scode" in keys or "scid" in keys or "sc_id" in keys or "code" in keys) and ("symbol" in keys or "name" in keys or "company" in keys or "scrip" in keys):
                        # extract possible keys
                        code = d.get("scode") or d.get("scid") or d.get("sc_id") or d.get("code")
                        symbol = d.get("symbol") or d.get("name") or d.get("company") or d.get("scrip")
                        try:
                            if code and symbol:
                                master[str(symbol).upper()] = str(code)
                                master[str(code).upper()] = str(code)
                        except:
                            pass
                    for v in d.values():
                        walk(v)
                elif isinstance(d, list):
                    for item in d:
                        walk(item)
            walk(data)
        except Exception:
            pass

    # If master still empty, seed with common tickers via autosuggest for popular names
    popular = ["RELIANCE","TCS","INFY","HDFCBANK","SBIN","ICICIBANK","AXISBANK","LT","BHARTIARTL","ITC","HINDUNILVR","MARUTI","BAJAJ-AUTO","ONGC","BPCL","IOC","NTPC","ADANIENT","ADANIPORTS","BDL"]
    for sym in popular:
        try:
            results = _moneycontrol_autosuggest(sym)
            for it in results:
                if isinstance(it, dict):
                    # try many possible keys
                    code = it.get("scode") or it.get("scid") or it.get("id") or it.get("sCode")
                    name = it.get("name") or it.get("label") or it.get("value") or it.get("symbol")
                    url = it.get("url") or it.get("link")
                    if not code and isinstance(url, str):
                        # sometimes url ends with code or slug
                        code = url.rstrip("/").split("/")[-1]
                    if code and name:
                        master[str(name).upper()] = str(code)
                        master[str(code).upper()] = str(code)
        except Exception:
            pass

    return master

# -------------------------
# Resolve to MoneyControl code (best-effort)
# -------------------------
def resolve_moneycontrol_code(symbol: str):
    """
    Returns a MoneyControl code (string) for given symbol/name/BSE-code, or None.
    Uses master mapping first, then autosuggest as fallback.
    """
    if not symbol:
        return None
    s = clean_ticker(symbol)
    master = build_moneycontrol_master()
    # direct lookup
    if s in master:
        return master[s]
    # try more aggressive matching (name contains, etc.)
    for k, v in master.items():
        if k and s and (s in k or k in s):
            return v
    # fallback to autosuggest
    try:
        results = _moneycontrol_autosuggest(s)
        for it in results:
            if isinstance(it, dict):
                code = it.get("scode") or it.get("scid") or it.get("id") or it.get("sCode")
                url = it.get("url") or it.get("link")
                name = it.get("name") or it.get("label") or it.get("value")
                if not code and isinstance(url, str):
                    code = url.rstrip("/").split("/")[-1]
                if code:
                    # update master for future
                    try:
                        build_moneycontrol_master.cache_clear()
                    except:
                        pass
                    return str(code)
    except Exception:
        pass
    return None

# -------------------------
# get_best_ticker: tries all sources
# -------------------------
def get_best_ticker(symbol, period="6mo", interval="1d"):
    """
    Tries (in order):
      1) yfinance attempts with candidate tickers
      2) Yahoo CSV direct download
      3) MoneyControl lookup & fetch
    Returns (source_identifier, normalized_dataframe) or (None,None)
    """
    if not symbol:
        return None, None

    candidates = resolve_ticker_candidates(symbol)
    # 1) yfinance
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
            df2 = _yahoo_csv_download(t, period, interval)
            df2 = _normalize_df(df2)
            if df2 is not None:
                return t, df2
        except Exception:
            continue

    # 3) MoneyControl fallback
    try:
        mc_code = resolve_moneycontrol_code(symbol)
        if mc_code:
            dfmc = _moneycontrol_historical_by_code(mc_code, period=period, interval=interval)
            if dfmc is not None:
                return mc_code, dfmc
    except Exception:
        pass

    return None, None

# -------------------------
# Public API functions used by frontend
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
    df2 = df.dropna(subset=["Close"])
    if df2.empty:
        return None
    return {"high": float(df2["High"].max()), "low": float(df2["Low"].min()), "timestamp": df2["Date"].iloc[-1]}

@lru_cache(maxsize=128)
def stock_summary(symbol: str, period="6mo", interval="1d"):
    t, df = get_best_ticker(symbol, period, interval)
    if df is None or df.empty:
        return None
    try:
        first = float(df["Close"].dropna().iloc[0])
        last = float(df["Close"].dropna().iloc[-1])
        pct = (last - first) / (first + 1e-9) * 100
        return {"ticker": t, "first_close": first, "latest_close": last, "pct_change": pct, "recent": df.tail(30).to_dict(orient="records")}
    except Exception:
        return None

# -------------------------
# News & sentiment helpers
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
        txt = a.get("title","")[:512]
        try:
            s = sentiment_analyzer(txt)[0]
            out.append({**a, "sentiment": {"label": s.get("label"), "score": float(s.get("score",0.0))}})
        except Exception:
            out.append({**a, "sentiment": None})
    return out

# -------------------------
# GitHub / ArXiv helpers (lightweight)
# -------------------------
def fetch_github_trending(language=None, since="daily"):
    try:
        url = f"https://github.com/trending/{language}" if language else "https://github.com/trending"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        repos = []
        for repo in soup.select("article.Box-row")[:20]:
            title_tag = repo.find(["h1","h2"])
            if not title_tag:
                continue
            link_tag = title_tag.find("a")
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
        papers=[]
        for entry in soup.find_all("entry")[:max_results]:
            papers.append({"title": entry.title.get_text(strip=True), "summary": entry.summary.get_text(strip=True), "link": entry.id.get_text(strip=True)})
        return papers
    except Exception:
        return []

# ========================= END FILE =========================
