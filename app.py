
from datetime import timezone

import streamlit as st
import pandas as pd

from crew_pipeline import run_customervoice_analysis

from database.mongodb import (
    test_connection,
    get_recent_analysis_runs,
    get_analysis_by_id,
)


st.set_page_config(
    page_title="CustomerVoice AI",
    page_icon="🗣️",
    layout="wide",
)


st.title("🗣️ CustomerVoice")
st.subheader("AI-Powered Customer Feedback Intelligence")

st.write(
    "Analyze customer reviews using four specialized AI agents "
    "and store the results in MongoDB."
)


# -----------------------------------
# MongoDB Connection Status
# -----------------------------------

st.sidebar.header("Database Connection")

connection = test_connection()

if connection["success"]:
    st.sidebar.success("MongoDB Connected")
else:
    st.sidebar.error("MongoDB Disconnected")
    st.sidebar.caption(connection["message"])


# -----------------------------------
# Load Dataset
# -----------------------------------

DATA_PATH = "data/processed/cleaned_reviews.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Cleaned dataset not found.")
    st.stop()


# -----------------------------------
# Dataset Overview
# -----------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Reviews",
    f"{len(df):,}",
)

col2.metric(
    "Average Rating",
    f"{df['rating'].mean():.2f} / 5",
)

col3.metric(
    "Low-Rated Reviews",
    f"{len(df[df['rating'] <= 3]):,}",
)


st.divider()


# -----------------------------------
# Run New Analysis
# -----------------------------------

st.header("Run Customer Feedback Analysis")

sample_size = st.slider(
    "Number of reviews to analyze",
    min_value=1,
    max_value=min(50, len(df)),
    value=min(5, len(df)),
)

if st.button("🚀 Run Agentic Analysis", type="primary"):

    if not connection["success"]:
        st.error(
            "MongoDB is not connected. "
            "Please check your database configuration."
        )

    else:
        try:
            with st.spinner(
                "Four AI agents are analyzing customer feedback..."
            ):

                result = run_customervoice_analysis(
                    sample_size=sample_size
                )

            st.success(
                "Analysis completed and saved to MongoDB!"
            )

            st.write("**Analysis ID:**")
            st.code(result["run_id"])

            st.write(
                f"Reviews analyzed: {result['sample_size']}"
            )

            st.write(
                f"Average sample rating: "
                f"{result['average_rating']}"
            )

            st.header("Executive Business Report")
            st.markdown(result["report"])

            st.subheader("Individual Agent Results")

            for output in result["agent_outputs"]:

                with st.expander(
                    f"Agent {output['task_number']}: "
                    f"{output['agent']}"
                ):
                    st.write(output["output"])

        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.exception(error)


st.divider()


# -----------------------------------
# Previous Analysis History
# -----------------------------------

st.header("Analysis History")

if connection["success"]:

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
                    created_at = created_at.strftime(
                        "%d %b %Y, %I:%M %p UTC"
                    )

                history_data.append({
                    "Analysis ID": run["run_id"][:8],
                    "Date": created_at or "Unknown",
                    "Reviews": run.get("sample_size", 0),
                    "Average Rating": run.get(
                        "average_rating", 0
                    ),
                    "Status": run.get("status", "Unknown"),
                })

            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True,
            )

            selected_run = st.selectbox(
                "Select a previous analysis",
                options=[
                    run["run_id"] for run in previous_runs
                ],
                format_func=lambda run_id: next(
                    (
                        f"{run_id[:8]} — "
                        f"{run.get('created_at').strftime('%d %b %Y %H:%M')}"
                        for run in previous_runs
                        if run["run_id"] == run_id
                        and run.get("created_at")
                    ),
                    run_id,
                ),
            )

            if st.button("View Saved Report"):

                saved_analysis = get_analysis_by_id(
                    selected_run
                )

                if saved_analysis:

                    st.success("Saved analysis retrieved!")

                    st.write("**Analysis ID:**")
                    st.code(saved_analysis["run_id"])

                    st.subheader("Executive Business Report")
                    st.markdown(
                        saved_analysis["final_report"]
                    )

                    st.subheader("Reviews Analyzed")

                    st.dataframe(
                        pd.DataFrame(
                            saved_analysis["reviews"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("Individual Agent Results")

                    for output in saved_analysis.get(
                        "agent_outputs", []
                    ):

                        with st.expander(
                            f"Agent {output['task_number']}: "
                            f"{output['agent']}"
                        ):
                            st.write(output["output"])

                else:
                    st.warning("Analysis not found.")

    except Exception as error:
        st.error(
            f"Could not retrieve analysis history: {error}"
        )


st.divider()

st.caption(
    "CustomerVoice | CrewAI + Local Qwen + MongoDB Atlas"
)
