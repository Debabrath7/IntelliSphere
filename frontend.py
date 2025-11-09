# frontend.py
# Dark Neon frontend for IntelliSphere
# Author: Generated for Debabrath

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from backend_modules import (
    get_stock_data,
    get_stock_info if False else None,  # placeholder: info fetched via yfinance in get_stock_data or separate call
    fetch_github_trending,
    fetch_arxiv_papers,
    get_trends_keywords,
    get_news,
    analyze_headlines_sentiment,
    recommend_learning_resources
)
import backend_modules as bm  # use optimized functions directly

# ---------- Page config ----------
st.set_page_config(page_title="IntelliSphere", page_icon="🌐", layout="wide", initial_sidebar_state="collapsed")

# ---------- GLOBALS & UTILS ----------
NAV_ITEMS = [
    ("Home", "home"),
    ("Stock Insights", "stock"),
    ("Tech & Startup Trends", "trends"),
    ("Research & Education", "research"),
    ("Skill & Job Trends", "skills"),
    ("News & Sentiment", "news"),
    ("Feedback", "feedback"),
]

def _nav_button(label, key, active_key):
    """Render a nav button styled for the neon header. Returns True if clicked."""
    is_active = (active_key == key)
    cls = "nav-item active" if is_active else "nav-item"
    html = f"""
    <div class="{cls}" onclick="window._stNav='{key}';window.parent.postMessage({{'streamlit':true}}, '*')">
        {label}
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

def set_nav_state(key):
    st.session_state["nav"] = key

if "nav" not in st.session_state:
    st.session_state["nav"] = "home"

# ---------- STYLES (Dark Neon) ----------
st.markdown(
    """
    <style>
    /* page background */
    .stApp {{
        background: radial-gradient(ellipse at top left, #071021 0%, #05060a 35%, #020203 100%);
        color: #e6f7ff;
        font-family: 'Inter', sans-serif;
    }}

    /* top nav bar */
    .topbar {{
        position: sticky;
        top: 0;
        z-index: 999;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        padding:12px 32px;
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-bottom: 1px solid rgba(255,255,255,0.03);
        backdrop-filter: blur(6px);
    }}
    .brand {{
        display:flex;
        align-items:center;
        gap:12px;
    }}
    .logo {{
        width:44px;height:44px;border-radius:10px;
        background: linear-gradient(135deg,#00e6ff 0%, #7a00ff 100%);
        box-shadow: 0 6px 20px rgba(0, 230, 255, 0.12), 0 1px 0 rgba(255,255,255,0.04) inset;
        display:flex;align-items:center;justify-content:center;
        font-weight:700;color:#021; font-size:18px;
    }}
    .brand h1 {{ margin:0; font-size:18px; color: #a8f0ff; letter-spacing:0.4px; }}
    .nav {{
        display:flex; gap:10px; align-items:center;
    }}
    .nav-item {{
        padding:8px 14px; border-radius:999px; cursor:pointer;
        color: #cbeefc; font-weight:600; font-size:13px;
        transition: all 0.18s ease;
        border: 1px solid transparent;
    }}
    .nav-item:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,230,255,0.06);
    }}
    .nav-item.active {{
        background: linear-gradient(90deg, rgba(0,230,255,0.08), rgba(122,0,255,0.06));
        border: 1px solid rgba(0,230,255,0.12);
        box-shadow: 0 8px 30px rgba(0,230,255,0.06);
        color: #eaffff;
    }}

    /* page sections and cards */
    .section {{
        padding: 28px 40px;
    }}
    .glass-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 30px rgba(3,8,23,0.6);
    }}

    /* small tweaks */
    .metric-custom .stMetricLabel {{ color:#9feafc; }}
    .metric-custom .stMetricValue {{ color:#00e6d4; font-weight:700; }}

    /* mobile responsiveness */
    @media (max-width: 768px) {{
        .topbar {{ padding: 12px; flex-direction:column; gap:8px; align-items:flex-start; }}
        .nav {{ flex-wrap:wrap; gap:6px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- TOP NAVBAR ----------
topbar = st.container()
with topbar:
    # build html container for brand + nav and simple JS to set session nav
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <div class="logo">IS</div>
                <div>
                    <h1>IntelliSphere</h1>
                    <div style="font-size:11px;color:#7fdcff;margin-top:2px">AI-Powered Insight Dashboard</div>
                </div>
            </div>
            <div class="nav" id="nav">
                {"".join([f'<div class="nav-item {"active" if st.session_state["nav"]==key else "" }" onclick="document.cookie = \"_is_nav={key}; path=/\"">{label}</div>' for label,key in NAV_ITEMS])}
            </div>
        </div>
        <script>
        // read cookie nav and push to location hash so Streamlit can see it on re-run
        const setNavCookie = () => {{
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach(it => {{
                it.addEventListener('click', () => {{
                    const text = it.innerText.trim();
                    const keyMap = { {label:key for (label,key) in NAV_ITEMS} };
                    // create a mapping for labels to keys in JS-friendly format
                }});
            }});
        }};
        </script>
        """,
        unsafe_allow_html=True,
    )

# NOTE: We will use simple buttons below to control navigation (safer/reliable with Streamlit)
cols = st.columns([1, 8, 1])
with cols[1]:
    nav_row = st.columns(len(NAV_ITEMS))
    for i, (label, key) in enumerate(NAV_ITEMS):
        clicked = nav_row[i].button(label, key=f"nav_{key}")
        if clicked:
            set_nav_state(key)

# ---------- SECTION RENDERERS ----------
def render_home():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Welcome to IntelliSphere")
    st.markdown(
        """
        **IntelliSphere** is a modular AI dashboard built for fast insights — stocks, trends,
        research, and news — wrapped in a modern Dark Neon theme.
        """
    )
    st.divider()
    c1, c2, c3 = st.columns([3,2,2])
    c1.markdown("### Quick Actions")
    if c1.button("Get Sample Stock (TCS)"):
        st.session_state["nav"] = "stock"
    c1.markdown("- Fast, cached data\n- Neon-styled charts\n- Portfolio-friendly layout")
    c2.metric("Active Modules", "6")
    c3.metric("Last Update", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_stock():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.header("💹 Stock Insights")
    user_input = st.text_input("Company symbol (eg. TCS / TCS.NS):", "TCS")
    period = st.selectbox("Time range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=4)
    if st.button("Fetch Stock"):
        ticker_raw = user_input.strip().upper()
        ticker = ticker_raw if "." in ticker_raw else ticker_raw + ".NS"
        with st.spinner("Fetching data..."):
            df = bm.get_stock_data(ticker, period=period, interval="1h" if period in ["1d","5d"] else "1d")
            try:
                info = bm.yf.Ticker(ticker).info
            except Exception:
                info = {}
        if df is None or df.empty:
            st.error("No data available.")
        else:
            latest = round(float(df["Close"].iloc[-1]),2)
            first = round(float(df["Close"].iloc[0]),2)
            change = round((latest-first)/first*100,2) if first else 0.0
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{ticker_raw}", f"₹{latest}", f"{change}%")
            col2.metric("52W High", info.get("fiftyTwoWeekHigh","N/A"))
            col3.metric("52W Low", info.get("fiftyTwoWeekLow","N/A"))

            # interactive plot
            fig = px.line(df, x="Date", y="Close", title=f"{ticker_raw} price ({period})", markers=True)
            fig.update_traces(line_shape="spline")
            fig.update_layout(template="plotly_dark", margin=dict(t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_trends():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.header("💻 Tech & Startup Trends")
    lang = st.text_input("Language / Topic (eg. python):", "python")
    if st.button("Fetch Trending Repos"):
        with st.spinner("Fetching trending repos..."):
            repos = bm.fetch_github_trending(lang)
        if repos:
            for r in repos[:8]:
                st.markdown(f"**[{r['name']}]**  \n{r['description']}")
                st.caption(f"Stars: {r.get('stars','0')}")
                st.divider()
        else:
            st.warning("No trending repos found.")
    st.divider()
    st.subheader("Startup News (sentiment)")
    try:
        news = bm.get_news("startup OR funding OR venture capital India", max_items=6)
        sentiment = bm.analyze_headlines_sentiment(news)
        for n in sentiment:
            st.markdown(f"**{n['title']}**")
            st.caption(f"Sentiment: {n['sentiment']['label']} ({n['sentiment']['score']:.2f})")
            st.divider()
    except Exception as e:
        st.warning("Could not fetch startup news.")
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_research():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.header("📚 Research & Education")
    topic = st.text_input("Search papers on:", "machine learning")
    if st.button("Fetch Papers"):
        with st.spinner("Fetching papers..."):
            papers = bm.fetch_arxiv_papers(topic, max_results=5)
        for p in papers:
            st.subheader(p["title"])
            st.caption(", ".join(p["authors"]))
            st.write(p["summary"][:800] + "...")
            st.markdown(f"[Read more]({p['link']})")
            st.divider()
    st.divider()
    st.subheader("Recommended Courses")
    recs = bm.recommend_learning_resources(topic)
    for c in recs.get("courses", []):
        st.markdown(f"- [{c}]({c})")
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_skills():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.header("🔍 Skill & Job Trends")
    skills = st.text_input("Enter skills (comma separated):", "Python, SQL, ML")
    if st.button("Analyze"):
        keys = [k.strip() for k in skills.split(",")]
        with st.spinner("Analyzing trends..."):
            trends = bm.get_trends_keywords(tuple(keys) if isinstance(keys, list) else keys)
        if trends:
            df = pd.DataFrame([{"Skill": k, "Change (%)": trends[k]["pct_change"]} for k in keys if k in trends])
            fig = px.bar(df, x="Skill", y="Change (%)", title="Skill Popularity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Trends currently unavailable.")
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_news():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.header("📰 News & Sentiment")
    query = st.text_input("Topic / Company:", "Indian Stock Market")
    n_items = st.slider("Articles to fetch:", 3, 15, 6)
    if st.button("Get Live News"):
        with st.spinner("Fetching news..."):
            articles = bm.get_news(query, max_items=n_items)
            sentiments = bm.analyze_headlines_sentiment(articles)
        if not sentiments:
            st.warning("No articles found.")
        else:
            for art in sentiments:
                left, right = st.columns([5,1])
                with left:
                    st.markdown(f"**{art['title']}**")
                    if art.get("link"):
                        st.markdown(f"[Read more]({art['link']})")
                    st.caption(f"{art['published'].strftime('%b %d, %Y %H:%M')}")
                with right:
                    lab = art['sentiment']['label']
                    sc = art['sentiment']['score']
                    badge = "🟢" if "POS" in lab else "🔴" if "NEG" in lab else "⚪"
                    st.markdown(f"**{badge} {lab}**\n\n{sc:.2f}")
                st.divider()
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_feedback():
    st.markdown("<div class='section'><div class='glass-card'>", unsafe_allow_html=True)
    st.header("💬 Feedback")
    name = st.text_input("Name")
    rating = st.slider("Rate (1-5):", 1, 5, 4)
    comments = st.text_area("Comments")
    if st.button("Submit Feedback"):
        try:
            import pandas as pd, os
            row = {"Name": name, "Rating": rating, "Comments": comments, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            if not os.path.exists("feedback.csv"):
                pd.DataFrame([row]).to_csv("feedback.csv", index=False)
            else:
                df = pd.read_csv("feedback.csv")
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv("feedback.csv", index=False)
            st.success("Thanks — feedback submitted!")
        except Exception as e:
            st.error("Unable to save feedback.")
    st.markdown("</div></div>", unsafe_allow_html=True)


# ---------- MAIN RENDER ----------
def render_dashboard():
    st.experimental_memo.clear()  # clears any leftover stale cache on manual reloads
    nav = st.session_state.get("nav", "home")
    if nav == "home":
        render_home()
    elif nav == "stock":
        render_stock()
    elif nav == "trends":
        render_trends()
    elif nav == "research":
        render_research()
    elif nav == "skills":
        render_skills()
    elif nav == "news":
        render_news()
    elif nav == "feedback":
        render_feedback()
    else:
        render_home()
