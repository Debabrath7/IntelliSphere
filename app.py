# ==============================================
# app.py — IntelliSphere launcher
# Author: Debabrath (refactored)
# ==============================================
import streamlit as st
import sys, os

# allow running when repository root is not pythonpath
sys.path.append(os.path.dirname(__file__) or ".")

from frontend import render_dashboard

st.set_page_config(
    page_title="IntelliSphere | AI-Powered Insights",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

if __name__ == "__main__":
    render_dashboard()
