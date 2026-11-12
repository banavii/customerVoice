
from crewai import Task


def create_recommendation_task(
    agent,
    classification_task,
    insights_task,
    sentiment_results,
    sample_size,
):
    return Task(
        description=f"""
        Prepare practical business recommendations using
        the classification and insights results.

        DIRECT SENTIMENT RESULTS:
        {sentiment_results}

        SAMPLE SIZE:
        {sample_size}

        Include:
        1. Main customer concerns.
        2. Recommended actions.
        3. Responsible business teams.
        4. Priority: High, Medium or Low.
        5. Expected potential benefits.
        6. Findings requiring further investigation.

        Rules:
        - Base recommendations on supplied evidence.
        - Do not invent financial statistics.
        - Do not guarantee business improvements.
        - Avoid broad conclusions from small samples.
        - Prefer low-cost validation before major changes.
        """,
        expected_output=(
            "A concise prioritized business recommendation "
            "report with supporting evidence, responsible "
            "teams and clearly stated limitations."
        ),
        agent=agent,
        context=[
            classification_task,
            insights_task,
        ],
    )
