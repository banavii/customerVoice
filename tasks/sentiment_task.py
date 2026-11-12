
from crewai import Task

def create_sentiment_task(agent, reviews_text):
    return Task(
        description=f"""
        Analyze the sentiment of every customer review below.

        REVIEWS:
        {reviews_text}

        For each review:
        1. Identify sentiment as Positive, Negative or Neutral.
        2. Identify sentiment intensity as Low, Medium or High.
        3. Explain the sentiment briefly.
        4. Preserve the original review ID.

        Important:
        - Analyze the actual review text.
        - Do not assume that a rating always represents sentiment.
        - A review can contain both positive and negative opinions.
        - Identify the dominant sentiment.

        Return a structured list with:
        Review ID, Sentiment, Intensity, Explanation.
        """,
        expected_output=(
            "A structured sentiment analysis for every review "
            "with ID, sentiment, intensity and explanation."
        ),
        agent=agent,
    )
