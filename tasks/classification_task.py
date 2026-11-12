
from crewai import Task

def create_classification_task(agent, reviews_text):
    return Task(
        description=f"""
        Analyze every customer review below.

        REVIEWS:
        {reviews_text}

        Classify each review into exactly ONE category:

        1. Product Quality
        2. Delivery
        3. Pricing
        4. Customer Service
        5. Packaging
        6. Usability and Fit
        7. Returns and Refunds
        8. Other

        Rules:
        - Classify every review separately.
        - Use its review ID in the result.
        - Select the main topic.
        - Do not invent information.

        Return a structured list containing:
        Review ID, Category, Reason.
        """,
        expected_output=(
            "A structured list classifying every review, "
            "including its ID, category and short reason."
        ),
        agent=agent,
    )
