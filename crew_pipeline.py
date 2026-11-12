
from pathlib import Path
from uuid import uuid4

import pandas as pd
from crewai import Crew, Process

from agents.classification_agent import classification_agent
from agents.insights_agent import insights_agent
from agents.recommendation_agent import recommendation_agent

from tasks.classification_task import create_classification_task
from tasks.insights_task import create_insights_task
from tasks.recommendation_task import create_recommendation_task

from sentiment_analysis import analyze_sentiment
from database.mongodb import save_analysis_run


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_reviews.csv"


def run_customervoice_analysis(sample_size=5):

    # 1. Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "product_id",
        "product_category",
        "rating",
        "review_text",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2. Validate and clean
    df = df.dropna(
        subset=["review_text", "rating"]
    ).copy()

    df["review_text"] = (
        df["review_text"].astype(str).str.strip()
    )

    df = df[df["review_text"] != ""]

    if df.empty:
        raise ValueError("No valid reviews available.")

    sample_size = int(sample_size)

    if sample_size < 1:
        raise ValueError("Sample size must be at least 1.")

    sample_size = min(sample_size, len(df))

    # 3. Sample reviews
    selected = df.sample(
        n=sample_size,
        random_state=42,
    ).reset_index(drop=True)

    review_records = []

    for index, row in selected.iterrows():
        review_records.append({
            "review_id": index + 1,
            "product_id": str(row["product_id"]),
            "product_category": str(row["product_category"]),
            "rating": float(row["rating"]),
            "review_text": str(row["review_text"]),
        })

    reviews_text = "\n".join(
        f"Review ID: {r['review_id']}\n"
        f"Product ID: {r['product_id']}\n"
        f"Category: {r['product_category']}\n"
        f"Rating: {r['rating']}\n"
        f"Review: {r['review_text']}\n"
        for r in review_records
    )

    # 4. Direct sentiment analysis (no CrewAI agent)
    print("Running direct Qwen sentiment analysis...")

    sentiment_results = analyze_sentiment(review_records)

    print("Sentiment analysis completed.")

    # 5. Build three CrewAI tasks
    classification_task = create_classification_task(
        classification_agent,
        reviews_text,
    )

    insights_task = create_insights_task(
        insights_agent,
        classification_task,
        sentiment_results,
        len(review_records),
    )

    recommendation_task = create_recommendation_task(
        recommendation_agent,
        classification_task,
        insights_task,
        sentiment_results,
        len(review_records),
    )

    # 6. Run the three agents
    crew = Crew(
        agents=[
            classification_agent,
            insights_agent,
            recommendation_agent,
        ],
        tasks=[
            classification_task,
            insights_task,
            recommendation_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    print("Running three-agent CrewAI workflow...")

    result = crew.kickoff()

    final_report = str(result)

    # 7. Preserve the existing output structure
    task_outputs = [
        {
            "task_number": 1,
            "agent": "Direct LLM Sentiment Analysis",
            "output": str(sentiment_results),
        }
    ]

    for index, task_output in enumerate(
        result.tasks_output,
        start=2,
    ):
        task_outputs.append({
            "task_number": index,
            "agent": str(task_output.agent),
            "output": str(task_output.raw),
        })

    # 8. Calculate statistics using Python
    average_rating = round(
        sum(r["rating"] for r in review_records)
        / len(review_records),
        2,
    )

    sentiment_counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0,
    }

    for item in sentiment_results:
        sentiment_counts[item["sentiment"]] += 1

    # 9. Prepare MongoDB document
    run_id = str(uuid4())

    analysis_data = {
        "run_id": run_id,
        "project": "CustomerVoice",
        "architecture": "3_agents_1_direct_llm",
        "sample_size": len(review_records),
        "average_rating": average_rating,
        "reviews": review_records,
        "sentiment_results": sentiment_results,
        "sentiment_counts": sentiment_counts,
        "agent_outputs": task_outputs,
        "final_report": final_report,
    }

    # 10. Save to MongoDB
    save_analysis_run(analysis_data)

    print("Analysis saved to MongoDB.")

    # 11. Return results to Streamlit
    return {
        "run_id": run_id,
        "sample_size": len(review_records),
        "average_rating": average_rating,
        "report": final_report,
        "agent_outputs": task_outputs,
        "sentiment_results": sentiment_results,
        "sentiment_counts": sentiment_counts,
    }


if __name__ == "__main__":
    output = run_customervoice_analysis(3)

    print("\nCUSTOMERVOICE ANALYSIS COMPLETED")
    print("Run ID:", output["run_id"])
    print("\nSentiment Results:")
    print(output["sentiment_results"])
    print("\nBusiness Report:")
    print(output["report"])
