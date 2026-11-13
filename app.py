
from datetime import datetime

import streamlit as st
import pandas as pd

from crew_pipeline import run_customervoice_analysis

from database.mongodb import (
    test_connection,
    get_recent_analysis_runs,
    get_analysis_by_id,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CustomerVoice | AI Intelligence",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
    }

    [data-testid="stSidebar"] {
        background-color: #11182b;
        border-right: 1px solid #25304a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #f1f5ff !important;
    }

    p, label, .stMarkdown {
        color: #c5cee0;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            #172554 0%,
            #312e81 55%,
            #4338ca 100%
        );
        border: 1px solid #414b91;
        margin-bottom: 25px;
    }

    .hero-title {
        color: white;
        font-size: 32px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #dbe4ff;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: #151e33;
        border: 1px solid #283653;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aab8d3;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f5ff;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    div[data-testid="stExpander"] {
        background-color: #131d31;
        border: 1px solid #293956;
        border-radius: 12px;
    }

    .section-note {
        color: #91a0bc;
        font-size: 13px;
    }

    .status-card {
        background-color: #151e33;
        border: 1px solid #283653;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

DATA_PATH = "data/processed/cleaned_reviews.csv"


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def display_agent_outputs(outputs):
    if not outputs:
        st.info("No individual agent outputs available.")
        return

    for output in outputs:
        task_number = output.get("task_number", "")
        agent_name = output.get("agent", "AI Agent")

        with st.expander(
            f"Step {task_number}: {agent_name}"
        ):
            st.markdown(str(output.get("output", "")))


def display_sentiment_results(sentiment_results):
    if not sentiment_results:
        st.info("Sentiment results are not available.")
        return

    sentiment_df = pd.DataFrame(sentiment_results)

    if "sentiment" in sentiment_df.columns:
        sentiment_counts = (
            sentiment_df["sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                sentiment_counts,
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.bar_chart(
                sentiment_counts.set_index("Sentiment")
            )

    st.dataframe(
        sentiment_df,
        use_container_width=True,
        hide_index=True,
    )


def prepare_batch_comparison(data):
    """
    Divide reviews into four reproducible groups.

    These are comparison batches, NOT calendar quarters.
    """

    shuffled = data.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    batch_number = (
        shuffled.index * 4 // len(shuffled)
    ) + 1

    shuffled["Comparison Batch"] = (
        "Batch " + batch_number.astype(str)
    )

    return shuffled


# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = load_dataset()

except FileNotFoundError:
    st.error(
        "Cleaned dataset not found. "
        "Please check data/processed/cleaned_reviews.csv."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🗣️ CustomerVoice")
    st.caption("AI Customer Intelligence Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Run Analysis",
            "Feedback Comparison",
            "Analysis History",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    connection = test_connection()

    if connection["success"]:
        st.success("MongoDB Connected")
    else:
        st.error("MongoDB Disconnected")
        st.caption(connection["message"])

    st.markdown(
        """
        <div class="status-card">
            <small>AI ENGINE</small><br>
            <b>Local Qwen</b><br><br>
            <small>ORCHESTRATION</small><br>
            <b>CrewAI</b><br><br>
            <small>DATABASE</small><br>
            <b>MongoDB Atlas</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("CustomerVoice | Project Dashboard")


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                CustomerVoice Intelligence
            </div>
            <div class="hero-subtitle">
                Turn customer feedback into meaningful business insights
                using AI-powered review analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Dataset Overview")

    total_reviews = len(df)

    average_rating = (
        pd.to_numeric(df["rating"], errors="coerce").mean()
    )

    low_rated = len(
        df[pd.to_numeric(df["rating"], errors="coerce") <= 3]
    )

    high_rated = len(
        df[pd.to_numeric(df["rating"], errors="coerce") >= 4]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        f"{total_reviews:,}",
    )

    col2.metric(
        "Average Rating",
        f"{average_rating:.2f} / 5",
    )

    col3.metric(
        "Low-Rated Reviews",
        f"{low_rated:,}",
    )

    col4.metric(
        "High-Rated Reviews",
        f"{high_rated:,}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Rating Distribution")

        rating_distribution = (
            df["rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_distribution)

    with right:
        st.subheader("Reviews by Product Category")

        if "product_category" in df.columns:

            category_distribution = (
                df["product_category"]
                .value_counts()
                .head(10)
            )

            st.bar_chart(category_distribution)

        else:
            st.info("Product category information unavailable.")

    st.divider()

    st.subheader("About CustomerVoice")

    st.write(
        """
        CustomerVoice analyzes e-commerce customer reviews
        to identify sentiment, recurring feedback categories,
        customer concerns and possible business improvements.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧠")
        st.markdown("**Sentiment Analysis**")
        st.caption(
            "Identifies positive, negative and neutral customer feedback."
        )

    with col2:
        st.markdown("### 📊")
        st.markdown("**Customer Insights**")
        st.caption(
            "Identifies recurring patterns and feedback categories."
        )

    with col3:
        st.markdown("### 💡")
        st.markdown("**Business Recommendations**")
        st.caption(
            "Generates practical recommendations from customer feedback."
        )


# =========================================================
# RUN ANALYSIS PAGE
# =========================================================

elif page == "Run Analysis":

    st.title("Run AI Analysis")

    st.write(
        "Submit a review sample to the CustomerVoice AI pipeline."
    )

    st.info(
        "The workflow uses direct Qwen sentiment analysis "
        "followed by three CrewAI agents."
    )

    st.subheader("Analysis Configuration")

    sample_size = st.slider(
        "Number of reviews to analyze",
        min_value=1,
        max_value=min(50, len(df)),
        value=min(5, len(df)),
    )

    st.caption(
        "Start with a small sample while testing the local AI model."
    )

    if st.button(
        "🚀 Start Analysis",
        type="primary",
        use_container_width=True,
    ):

        if not connection["success"]:
            st.error(
                "MongoDB is disconnected. "
                "Check your database configuration."
            )

        else:

            try:
                with st.spinner(
                    "Qwen and CrewAI are analyzing customer feedback..."
                ):

                    result = run_customervoice_analysis(
                        sample_size=sample_size
                    )

                st.session_state["latest_result"] = result

                st.success(
                    "Analysis completed and saved successfully!"
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.exception(error)

    result = st.session_state.get("latest_result")

    if result:

        st.divider()

        st.subheader("Analysis Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Reviews Analyzed",
            result.get("sample_size", 0),
        )

        col2.metric(
            "Average Rating",
            result.get("average_rating", "N/A"),
        )

        col3.metric(
            "Sentiment Records",
            len(result.get("sentiment_results", [])),
        )

        st.caption(
            f"Analysis ID: {result.get('run_id', 'Unknown')}"
        )

        st.divider()

        st.subheader("Sentiment Analysis")

        display_sentiment_results(
            result.get("sentiment_results", [])
        )

        st.divider()

        st.subheader("Executive Business Report")

        st.markdown(
            result.get(
                "report",
                "No report available.",
            )
        )

        st.divider()

        st.subheader("AI Agent Outputs")

        display_agent_outputs(
            result.get("agent_outputs", [])
        )


# =========================================================
# FEEDBACK COMPARISON PAGE
# =========================================================

elif page == "Feedback Comparison":

    st.title("Feedback Comparison")

    st.write(
        "Compare customer feedback patterns across four "
        "reproducible groups of reviews."
    )

    st.warning(
        "The dataset does not contain review dates. "
        "These are comparison batches, not actual calendar quarters "
        "or chronological trends."
    )

    batch_df = prepare_batch_comparison(df)

    batch_summary = (
        batch_df.groupby("Comparison Batch")
        .agg(
            Total_Reviews=("rating", "count"),
            Average_Rating=("rating", "mean"),
        )
        .reset_index()
    )

    batch_summary["Average_Rating"] = (
        batch_summary["Average_Rating"].round(2)
    )

    if "recommended" in batch_df.columns:

        recommended_values = pd.to_numeric(
            batch_df["recommended"],
            errors="coerce",
        )

        batch_df["recommendation_value"] = recommended_values

        recommendation_summary = (
            batch_df.groupby("Comparison Batch")[
                "recommendation_value"
            ]
            .mean()
            .mul(100)
            .reset_index(name="Recommendation_Percentage")
        )

        batch_summary = batch_summary.merge(
            recommendation_summary,
            on="Comparison Batch",
            how="left",
        )

        batch_summary["Recommendation_Percentage"] = (
            batch_summary["Recommendation_Percentage"].round(1)
        )

    st.subheader("Batch Performance")

    metric_columns = st.columns(4)

    for index, (_, batch) in enumerate(batch_summary.iterrows()):

        metric_columns[index].metric(
            batch["Comparison Batch"],
            f"{batch['Average_Rating']:.2f} / 5",
            f"{int(batch['Total_Reviews']):,} reviews",
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Average Rating Comparison")

        st.bar_chart(
            batch_summary.set_index("Comparison Batch")[
                ["Average_Rating"]
            ]
        )

    with right:

        if "Recommendation_Percentage" in batch_summary.columns:

            st.subheader("Recommendation Comparison")

            st.bar_chart(
                batch_summary.set_index("Comparison Batch")[
                    ["Recommendation_Percentage"]
                ]
            )

        else:
            st.info("Recommendation data is unavailable.")

    st.divider()

    st.subheader("Rating Distribution by Batch")

    rating_batch = (
        batch_df.groupby(
            ["Comparison Batch", "rating"]
        )
        .size()
        .unstack(fill_value=0)
    )

    st.bar_chart(rating_batch)

    st.divider()

    st.subheader("Product Category Comparison")

    if "product_category" in batch_df.columns:

        category_batch = pd.crosstab(
            batch_df["Comparison Batch"],
            batch_df["product_category"],
        )

        st.dataframe(
            category_batch,
            use_container_width=True,
        )

    st.subheader("Detailed Batch Metrics")

    st.dataframe(
        batch_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Batch membership is reproducible using a fixed random seed. "
        "Differences between batches should not be interpreted as "
        "changes over time or as causal effects."
    )


# =========================================================
# ANALYSIS HISTORY PAGE
# =========================================================

elif page == "Analysis History":

    st.title("Analysis History")

    st.write(
        "Review previous CustomerVoice analysis runs saved in MongoDB."
    )

    if not connection["success"]:

        st.error(
            "MongoDB is disconnected. "
            "Analysis history cannot be retrieved."
        )

    else:

        try:

            previous_runs = get_recent_analysis_runs(20)

            if not previous_runs:

                st.info(
                    "No previous analyses found. "
                    "Run your first analysis to see it here."
                )

            else:

                history_data = []

                for run in previous_runs:

                    created_at = run.get("created_at")

                    if created_at:
                        date_display = created_at.strftime(
                            "%d %b %Y, %I:%M %p"
                        )
                    else:
                        date_display = "Unknown"

                    history_data.append({
                        "Analysis ID": run["run_id"][:8],
                        "Date": date_display,
                        "Reviews": run.get("sample_size", 0),
                        "Average Rating": run.get(
                            "average_rating", 0
                        ),
                        "Status": run.get("status", "Completed"),
                    })

                st.dataframe(
                    pd.DataFrame(history_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                st.subheader("View Previous Report")

                selected_run = st.selectbox(
                    "Select an analysis",
                    options=[
                        run["run_id"]
                        for run in previous_runs
                    ],
                    format_func=lambda run_id: run_id[:8],
                )

                if st.button(
                    "View Saved Report",
                    type="primary",
                ):

                    saved_analysis = get_analysis_by_id(
                        selected_run
                    )

                    if saved_analysis:

                        st.success("Analysis retrieved successfully!")

                        st.caption(
                            f"Analysis ID: {saved_analysis['run_id']}"
                        )

                        st.subheader("Executive Business Report")

                        st.markdown(
                            saved_analysis.get(
                                "final_report",
                                "No report available.",
                            )
                        )

                        st.divider()

                        st.subheader("Reviews Analyzed")

                        st.dataframe(
                            pd.DataFrame(
                                saved_analysis.get("reviews", [])
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.subheader("Sentiment Results")

                        display_sentiment_results(
                            saved_analysis.get(
                                "sentiment_results",
                                [],
                            )
                        )

                        st.subheader("AI Agent Outputs")

                        display_agent_outputs(
                            saved_analysis.get(
                                "agent_outputs",
                                [],
                            )
                        )

                    else:
                        st.warning("Analysis not found.")

        except Exception as error:
            st.error(
                f"Could not retrieve analysis history: {error}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#71809d; font-size:12px;">
        CustomerVoice AI · CrewAI · Local Qwen · MongoDB Atlas
        <br>
        Customer Feedback Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)
