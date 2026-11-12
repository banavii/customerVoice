
from crewai import Task


def create_insights_task(
    agent,
    classification_task,
    sentiment_results,
    sample_size,
):
    return Task(
        description=f"""
        Analyze the classification results from the
        previous CrewAI task together with these
        direct-LLM sentiment results:

        SENTIMENT RESULTS:
        {sentiment_results}

        NUMBER OF REVIEWS:
        {sample_size}

        Identify:
        1. Main feedback categories.
        2. Negative customer concerns.
        3. Positive customer experiences.
        4. Patterns supported by the sample.
        5. Possible explanations requiring verification.

        Rules:
        - Use only supplied evidence.
        - Do not invent statistics.
        - Do not claim a systemic problem from a small sample.
        - Distinguish observed findings from hypotheses.
        - Mention the sample size and its limitations.
        """,
        expected_output=(
            "A concise evidence-based customer insights "
            "report with positive findings, concerns, "
            "possible explanations and limitations."
        ),
        agent=agent,
        context=[classification_task],
    )
