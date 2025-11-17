# backend_modules.py
# Robust multi-source fetcher with embedded DEMO MODE (real-like baselines)
# Tries: yfinance -> Yahoo CSV -> MoneyControl -> Demo synthetic data (fallback)
# Demo mode embedded for 20 Indian stocks with real-like current baseline prices
# Author: Debabrath

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

# Optional sentiment pipeline
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
# DEMO: 20 stocks & realistic baseline prices (approx current market levels)
# -------------------------
DEMO_SYMBOLS = [
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","ASIANPAINT",
    "ITC","LT","BHARTIARTL","ULTRACEMCO","WIPRO","HCLTECH","MARUTI","TECHM",
    "HINDUNILVR","AXISBANK","BAJAJFINSV","POWERGRID"
]

# Updated baseline prices (Nov 2025 approximate)
BASELINE_PRICES = {
    "RELIANCE": 2885.0,
    "TCS": 4120.0,
    "INFY": 1635.0,
    "HDFCBANK": 1592.0,
    "ICICIBANK": 1125.0,
    "SBIN": 791.0,
    "KOTAKBANK": 1878.0,
    "ASIANPAINT": 3305.0,
    "ITC": 453.0,
    "LT": 3950.0,
    "BHARTIARTL": 1285.0,
    "ULTRACEMCO": 9980.0,
    "WIPRO": 448.0,
    "HCLTECH": 1705.0,
    "MARUTI": 121500.0,
    "TECHM": 1510.0,
    "HINDUNILVR": 2520.0,
    "AXISBANK": 1245.0,
    "BAJAJFINSV": 17400.0,
    "POWERGRID": 298.0
}

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
    s = clean_ticker(symbol)
    if not s:
        return []
    if s.isdigit():
        return [f"{s}.BO", s]
    return [f"{s}.NS", f"{s}.BO", s]

def _normalize_df(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.copy()
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
    out = df.loc[:, ["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Date"])
    if out.empty:
        return None
    return out

# -------------------------
# yfinance fallback
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
# Yahoo CSV fallback
# -------------------------
_PERIOD_DAYS = {"1d":1,"5d":5,"1mo":30,"3mo":90,"6mo":182,"1y":365,"2y":365*2,"5y":365*5,"10y":365*10}
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
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
        if df.empty:
            return None
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan
        return df.reset_index(drop=True)
    except Exception:
        return None

# -------------------------
# MoneyControl (best-effort)
# -------------------------
def _moneycontrol_autosuggest(q: str):
    try:
        url = "https://www.moneycontrol.com/mccode/common/autosuggest.php"
        r = requests.get(url, params={"query": q}, timeout=8)
        if r.status_code != 200:
            return []
        text = r.text.strip()
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
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
        return []
    except Exception:
        return []

def _moneycontrol_historical_by_code(scode: str, period="6mo", interval="1d"):
    try:
        if not scode:
            return None
        candidates = [
            f"https://www.moneycontrol.com/stocks/histories/get_price/{scode}?period={period}",
            f"https://priceapi.moneycontrol.com/pricefeed/mcfeed/stock/{scode}",
            f"https://priceapi.moneycontrol.com/pricefeed/historical/{scode}?Period={period}"
        ]
        for url in candidates:
            try:
                r = requests.get(url, timeout=8)
                if r.status_code != 200:
                    continue
                try:
                    data = r.json()
                    def find_list(d):
                        if isinstance(d, list) and d and isinstance(d[0], dict):
                            if any(k.lower() in ("date","dt","timestamp") for k in d[0].keys()):
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
                        for cand in ["date","Date","dt","timestamp"]:
                            if cand in df.columns:
                                df = df.rename(columns={cand:"Date"})
                                break
                        mapping={}
                        for col in df.columns:
                            low = col.lower()
                            if "open" in low: mapping[col]="Open"
                            if "high" in low: mapping[col]="High"
                            if "low" in low: mapping[col]="Low"
                            if "close" in low: mapping[col]="Close"
                            if "volume" in low: mapping[col]="Volume"
                        df = df.rename(columns=mapping)
                        if "Date" in df.columns:
                            try:
                                df["Date"] = pd.to_datetime(df["Date"], unit='s')
                            except:
                                try:
                                    df["Date"] = pd.to_datetime(df["Date"])
                                except:
                                    pass
                        df_norm = _normalize_df(df)
                        if df_norm is not None:
                            return df_norm
                except Exception:
                    pass
                text = r.text
                if "\n" in text and "," in text and "Date" in text[:200]:
                    try:
                        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
                        df_norm = _normalize_df(df)
                        if df_norm is not None:
                            return df_norm
                    except Exception:
                        pass
                try:
                    soup = BeautifulSoup(text, "html.parser")
                    table = soup.find("table")
                    if table:
                        df = pd.read_html(str(table))[0]
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
# DEMO synthetic data generator (REAL-LIKE baseline)
# -------------------------
def _seed_from_string(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31)

def _generate_price_series(seed:int, start_price:float, days:int, mu=0.0002, sigma=0.02):
    rng = np.random.RandomState(seed)
    dt = 1.0
    prices = [start_price]
    for _ in range(days-1):
        shock = rng.normal(loc=mu*dt, scale=sigma*math.sqrt(dt))
        prices.append(max(0.01, prices[-1] * math.exp(shock)))
    return np.array(prices)

def _build_demo_daily_df(symbol: str, years: int = 5):
    symbol = symbol.upper()
    seed = _seed_from_string(symbol)
    days = years * 252
    baseline = BASELINE_PRICES.get(symbol, None)
    if baseline is not None:
        start_price = baseline * 0.65
        mu = 0.00025
        sigma = 0.018
    else:
        start_price = 50 + (seed % 4500) * 0.01
        mu = 0.0003
        sigma = 0.02
    closes = _generate_price_series(seed+1, start_price, days, mu=mu, sigma=sigma)
    if baseline is not None:
        factor = baseline / float(closes[-1])
        closes = closes * factor
    dates = []
    today = datetime.utcnow().date()
    dt = today
    while len(dates) < days:
        if dt.weekday() < 5:
            dates.append(dt)
        dt = dt - timedelta(days=1)
    dates = list(reversed(dates))
    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    rng = np.random.RandomState(seed+2)
    opens = closes * (1 + rng.normal(0, 0.0025, size=len(closes)))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.006, size=len(closes))))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.006, size=len(closes))))
    volumes = (rng.normal(loc=5e6, scale=2e6, size=len(closes))).astype(int)
    volumes = np.where(volumes < 1000, 1000, volumes)
    df["Open"] = opens
    df["High"] = highs
    df["Low"] = lows
    df["Close"] = closes
    df["Volume"] = volumes
    return _normalize_df(df)

def _build_demo_intraday_df(symbol: str):
    symbol = symbol.upper()
    seed = _seed_from_string(symbol) + 1000
    rng = np.random.RandomState(seed)
    minutes = 390
    daily = _build_demo_daily_df(symbol, years=1)
    base = float(daily["Close"].iloc[-1]) if (daily is not None and not daily.empty) else (BASELINE_PRICES.get(symbol, 100.0))
    mu = 0.0
    sigma = 0.0007
    returns = rng.normal(loc=mu, scale=sigma, size=minutes)
    prices = [base]
    for r in returns:
        prices.append(max(0.01, prices[-1] * math.exp(r)))
    prices = prices[1:]
    last_day = datetime.utcnow().date()
    if last_day.weekday() >= 5:
        offset = (last_day.weekday() - 4)
        last_day = last_day - timedelta(days=offset)
    start_dt = datetime.combine(last_day, datetime.min.time()) + timedelta(hours=9, minutes=15)
    times = [start_dt + timedelta(minutes=i) for i in range(minutes)]
    df = pd.DataFrame({"Date": times})
    rng2 = np.random.RandomState(seed+2)
    opens = np.array(prices) * (1 + rng2.normal(0, 0.0005, size=len(prices)))
    highs = np.maximum(opens, prices) * (1 + np.abs(rng2.normal(0, 0.0012, size=len(prices))))
    lows = np.minimum(opens, prices) * (1 - np.abs(rng2.normal(0, 0.0012, size=len(prices))))
    volumes = (rng2.normal(loc=2000, scale=1000, size=len(prices))).astype(int)
    volumes = np.where(volumes < 1, 1, volumes)
    df["Open"] = opens
    df["High"] = highs
    df["Low"] = lows
    df["Close"] = prices
    df["Volume"] = volumes
    return _normalize_df(df)

# -------------------------
# DEMO cache for speed
# -------------------------
_DEMO_DAILY_CACHE = {}
_DEMO_INTRADAY_CACHE = {}

def _get_demo(symbol: str, period="6mo", interval="1d"):
    s = clean_ticker(symbol)
    if interval.endswith("m") or interval in ("1m","5m","15m"):
        if s not in _DEMO_INTRADAY_CACHE:
            _DEMO_INTRADAY_CACHE[s] = _build_demo_intraday_df(s)
        return _DEMO_INTRADAY_CACHE[s]
    if s not in _DEMO_DAILY_CACHE:
        _DEMO_DAILY_CACHE[s] = _build_demo_daily_df(s, years=5)
    df = _DEMO_DAILY_CACHE[s]
    days = _period_to_days(period)
    try:
        last = df.tail(days)
        if last.empty:
            return df
        return last.reset_index(drop=True)
    except Exception:
        return df

# -------------------------
# MoneyControl master resolver (lightweight)
# -------------------------
@lru_cache(maxsize=1)
def build_moneycontrol_master():
    master = {}
    endpoints = [
        "https://priceapi.moneycontrol.com/pricefeed/bse/equitycash",
        "https://priceapi.moneycontrol.com/pricefeed/nse/equitycash",
        "https://priceapi.moneycontrol.com/pricefeed/stocklist"
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=6)
            if r.status_code != 200:
                continue
            data = r.json()
            def walk(d):
                if isinstance(d, dict):
                    keys = set(k.lower() for k in d.keys())
                    if ("scode" in keys or "scid" in keys or "sc_id" in keys or "code" in keys) and ("symbol" in keys or "name" in keys or "company" in keys):
                        code = d.get("scode") or d.get("scid") or d.get("sc_id") or d.get("code")
                        symbol = d.get("symbol") or d.get("name") or d.get("company")
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
            continue
    for sym in DEMO_SYMBOLS:
        master.setdefault(sym.upper(), sym.upper())
    return master

def resolve_moneycontrol_code(symbol: str):
    if not symbol:
        return None
    s = clean_ticker(symbol)
    master = build_moneycontrol_master()
    if s in master:
        return master[s]
    for k,v in master.items():
        if k and s and (s in k or k in s):
            return v
    results = _moneycontrol_autosuggest(s)
    for it in results:
        if isinstance(it, dict):
            code = it.get("scode") or it.get("scid") or it.get("id") or it.get("sCode")
            url = it.get("url") or it.get("link")
            if not code and isinstance(url,str):
                code = url.rstrip("/").split("/")[-1]
            if code:
                return str(code)
    return None

# -------------------------
# Main resolver: try live sources, fallback to demo
# -------------------------
def get_best_ticker(symbol, period="6mo", interval="1d"):
    if not symbol:
        return None, None
    candidates = resolve_ticker_candidates(symbol)
    for t in candidates:
        try:
            df = _yf_download(t, period, interval)
            df = _normalize_df(df)
            if df is not None:
                return t, df
        except Exception:
            continue
    for t in candidates:
        try:
            df2 = _yahoo_csv_download(t, period, interval)
            df2 = _normalize_df(df2)
            if df2 is not None:
                return t, df2
        except Exception:
            continue
    try:
        mc = resolve_moneycontrol_code(symbol)
        if mc:
            dfmc = _moneycontrol_historical_by_code(mc, period=period, interval=interval)
            dfmc = _normalize_df(dfmc)
            if dfmc is not None:
                return mc, dfmc
    except Exception:
        pass
    try:
        demo = _get_demo(symbol, period=period, interval=interval)
        if demo is not None:
            return f"DEMO:{clean_ticker(symbol)}", demo
    except Exception:
        pass
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

def get_today_high_low(symbol: str, prefer_intraday=True):
    t, df = get_best_ticker(symbol, "1d", "1m" if prefer_intraday else "1d")
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["Close"])
    if d.empty:
        return None
    return {"high": float(d["High"].max()), "low": float(d["Low"].min()), "timestamp": d["Date"].iloc[-1]}

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
    out = []
    for entry in feed.entries[:max_items]:
        try:
            pub = datetime(*entry.published_parsed[:6])
        except Exception:
            pub = datetime.utcnow()
        out.append({"title": entry.title, "link": entry.link, "published": pub})
    return out

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
# Lightweight GitHub / ArXiv helpers
# -------------------------
def fetch_github_trending(language=None, since="daily"):
    try:
        url = f"https://github.com/trending/{language}" if language else "https://github.com/trending"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text,"lxml")
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
        out = []
        for entry in soup.find_all("entry")[:max_results]:
            out.append({"title": entry.title.get_text(strip=True), "summary": entry.summary.get_text(strip=True), "link": entry.id.get_text(strip=True)})
        return out
    except Exception:
        return []
