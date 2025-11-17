# backend_modules.py
# Robust multi-source fetcher: yfinance -> Yahoo CSV -> Moneycontrol autosuggest fallback
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

# Optional: transformers sentiment (try best-effort)
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
# Basic utilities
# -------------------------
def clean_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        return ticker
    s = ticker.strip().upper()
    # remove common suffixes if accidentally passed
    for suf in [".NS", ".BO", ":NS", ":BO"]:
        if s.endswith(suf):
            s = s.replace(suf, "")
    return s.strip()

def resolve_ticker_candidates(symbol: str):
    """Return candidate tickers for yfinance / yahoo fallback purposes."""
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        return [f"{s}.BO", s]
    return [f"{s}.NS", f"{s}.BO", s]

# -------------------------
# Normalizer
# -------------------------
def _normalize_df(df):
    """Return DataFrame with Date,Open,High,Low,Close,Volume or None."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.copy()
    # Normalize Date column
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    # ensure columns exist, drop rows with invalid Date
    out = df.loc[:, ["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Date"])
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
    if isinstance(period, int):
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
        params = {"period1": start, "period2": end, "interval": interval, "events": "history", "includeAdjustedClose": "true"}
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        s = r.text
        if not s or "404 Not Found" in s:
            return None
        df = pd.read_csv(io.StringIO(s), parse_dates=["Date"])
        if df.empty:
            return None
        # ensure numeric columns
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan
        return df.reset_index(drop=True)
    except Exception:
        return None

# -------------------------
# Moneycontrol autosuggest + historical fetch
# -------------------------
def _moneycontrol_autosuggest(query: str):
    """
    Try Moneycontrol autosuggest endpoint to find symbol mapping.
    Returns list of candidate dicts with keys like: {'symbol','scode','name','type','url'} depending on response.
    """
    try:
        q = str(query).strip()
        if not q:
            return []
        # Moneycontrol autosuggest (common endpoint)
        autos_url = "https://www.moneycontrol.com/mccode/common/autosuggest.php"
        params = {"query": q}
        r = requests.get(autos_url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        # autosuggest returns JSON or JSONP-like; try parse
        text = r.text.strip()
        try:
            # If pure JSON
            data = json.loads(text)
        except Exception:
            # Sometimes it's wrapped like: callback(<json>);
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
        # normalize to list of results
        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            items = []
        # keep items as-is
        return items
    except Exception:
        return []

def _moneycontrol_historical_by_code(scode: str, period="6mo", interval="1d"):
    """
    Moneycontrol has endpoints to retrieve data for a company's history.
    We'll try known historical endpoints (works for many symbols).
    """
    try:
        # days
        days = _period_to_days(period)
        # Moneycontrol had endpoints like /stocks/histories/get_price/{code}?period={period}
        # Try the common pattern:
        url_candidates = [
            f"https://www.moneycontrol.com/stocks/company_info/hist_stock.php?sc_id={scode}&duration={period}",
            f"https://www.moneycontrol.com/stocks/histories/get_price/{scode}?period={period}",
        ]
        for url in url_candidates:
            try:
                r = requests.get(url, timeout=8)
                if r.status_code != 200:
                    continue
                text = r.text
                # Try to extract CSV-like data inside text (many MC endpoints return CSV or JS arrays)
                # Look for JSON arrays inside response
                if "{" in text and "}" in text:
                    # attempt to find numeric table with dates
                    soup = BeautifulSoup(text, "html.parser")
                    # Sometimes table rows exist
                    table = soup.find("table")
                    if table:
                        df = pd.read_html(str(table))[0]
                        # try to find Date, Open... columns
                        cols = [c for c in df.columns if isinstance(c, str)]
                        # basic cleaning
                        if "Date" in df.columns or any("Date" in str(c) for c in df.columns):
                            # normalize columns if present
                            df_cols = [str(c).strip() for c in df.columns]
                            df.columns = df_cols
                            # ensure required columns
                            for c in ["Open","High","Low","Close","Volume"]:
                                if c not in df.columns:
                                    df[c] = np.nan
                            if "Date" in df.columns:
                                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                            return _normalize_df(df)
                # fallback: if response CSV-like
                if "\n" in text and "," in text:
                    import io
                    try:
                        df = pd.read_csv(io.StringIO(text))
                        return _normalize_df(df)
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        pass
    return None

# -------------------------
# get_best_ticker: tries sources in order
# -------------------------
def get_best_ticker(symbol, period="6mo", interval="1d"):
    # 1) yfinance candidates
    candidates = resolve_ticker_candidates(symbol)
    for t in candidates:
        df = _yf_download(t, period, interval)
        df = _normalize_df(df)
        if df is not None:
            return t, df

    # 2) yahoo csv fallback
    for t in candidates:
        df2 = _yahoo_csv_download(t, period, interval)
        df2 = _normalize_df(df2)
        if df2 is not None:
            return t, df2

    # 3) Moneycontrol dynamic lookup (best cloud fallback)
    # Use autosuggest to get codes
    items = _moneycontrol_autosuggest(symbol)
    # items might be list of dicts with fields; try to find possible scode values
    scodes = []
    for it in items:
        # try several common keys
        if isinstance(it, dict):
            for key in ("scode","sCode","sc_code","scid","scodeid","scodeId","id"):
                if key in it:
                    scodes.append(str(it[key]))
            # some entries have URL slugs
            if "url" in it and isinstance(it["url"], str):
                # try to extract last part
                scodes.append(it["url"].rstrip("/").split("/")[-1])
    # dedupe
    scodes = [s for s in dict.fromkeys(scodes) if s]
    for sc in scodes:
        dfmc = _moneycontrol_historical_by_code(sc, period=period, interval=interval)
        if dfmc is not None:
            return sc, dfmc

    # final fallback: return None
    return None, None

# -------------------------
# Public functions used by frontend
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
def fetch_news_via_google_rss(query, max_items=15):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    out=[]
    for e in feed.entries[:max_items]:
        try:
            pub = datetime(*e.published_parsed[:6])
        except:
            pub = datetime.utcnow()
        out.append({"title": e.title, "link": e.link, "published": pub})
    return out

def analyze_headlines_sentiment(articles):
    if sentiment_analyzer is None:
        return [{**a, "sentiment": None} for a in articles]
    out=[]
    for a in articles:
        try:
            s = sentiment_analyzer(a["title"][:512])[0]
            out.append({**a, "sentiment": {"label": s.get("label"), "score": float(s.get("score",0.0))}})
        except:
            out.append({**a, "sentiment": None})
    return out

# -------------------------
# Lightweight GitHub / ArXiv helpers (unchanged)
# -------------------------
def fetch_github_trending(language=None, since="daily"):
    try:
        url = f"https://github.com/trending/{language}" if language else "https://github.com/trending"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        repos=[]
        for row in soup.select("article.Box-row")[:20]:
            a = row.find("a")
            name = a.get("href","").strip("/") if a else "unknown"
            desc_tag = row.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            repos.append({"name":name,"description":desc})
        return repos
    except:
        return []

def fetch_arxiv_papers(q, max_results=5):
    try:
        base = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{q}", "start": 0, "max_results": max_results}
        r = requests.get(base, params=params, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "xml")
        out=[]
        for entry in soup.find_all("entry")[:max_results]:
            out.append({"title": entry.title.get_text(strip=True), "summary": entry.summary.get_text(strip=True), "link": entry.id.get_text(strip=True)})
        return out
    except:
        return []
