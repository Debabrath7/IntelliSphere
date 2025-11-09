# ==============================================
# IntelliSphere: AI-Powered Insight Dashboard
# Author: Debabrath
# ==============================================

import streamlit as st
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
