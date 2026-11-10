import streamlit as st
import pandas as pd
from agents import run_customervoice_analysis

st.set_page_config(page_title="CustomerVoice AI", layout="wide")

st.title("🗣️ CustomerVoice — Agentic AI Feedback Analyzer")
st.write("Multi-Agent AI System for feedback categorization, pain-point detection, and business recommendations.")

# Sidebar Controls
st.sidebar.header("Settings")
sample_size = st.sidebar.slider("Number of low-rating reviews to analyze", min_value=5, max_value=50, value=15)

# Load data preview
df = pd.read_csv("data/processed/cleaned_reviews.csv")

st.subheader("Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Cleaned Reviews", len(df))
col2.metric("Avg Rating", round(df["rating"].mean(), 2))
col3.metric("Negative/Neutral Reviews (<=3★)", len(df[df["rating"] <= 3]))

if st.button("🚀 Run Agentic Analysis"):
    with st.spinner("CrewAI Agents are analyzing feedback..."):
        report = run_customervoice_analysis(sample_size=sample_size)
        st.success("Analysis Complete!")
        st.markdown("### Executive Business Report")
        st.markdown(report)