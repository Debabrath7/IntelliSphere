# ==============================================
# IntelliSphere: AI-Powered Insight Dashboard
# Author: Debabrath
# ==============================================

import streamlit as st
import sys, os

# ✅ Ensure Python can find 'frontend.py' even on Streamlit Cloud or mounted paths
sys.path.append(os.path.dirname(__file__))

from frontend import render_dashboard

# -----------------------------------------------------
# STREAMLIT CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="IntelliSphere | AI-Powered Insights",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------
# MAIN ENTRY
# -----------------------------------------------------
if __name__ == "__main__":
    render_dashboard()
