
import json

from agents.llm_config import direct_client, direct_model


def analyze_sentiment(review_records):
    """
    Analyze customer review sentiment using a direct
    Qwen API call without a CrewAI agent.
    """

    results = []

    for review in review_records:
        review_id = review["review_id"]
        review_text = review["review_text"]

        response = direct_client.chat.completions.create(
            model=direct_model,
            temperature=0.1,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer sentiment classifier. "
                        "Return only a valid JSON object with "
                        "two fields: sentiment and intensity. "
                        "sentiment must be Positive, Negative "
                        "or Neutral. intensity must be Low, "
                        "Medium or High. No markdown or explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Classify this customer review:\n"
                        f"{review_text}"
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content or ""

        # Some local models wrap JSON in markdown fences.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid sentiment JSON for review {review_id}: "
                f"{raw[:300]}"
            ) from error

        sentiment = parsed.get("sentiment")
        intensity = parsed.get("intensity")

        if sentiment not in {"Positive", "Negative", "Neutral"}:
            raise ValueError(
                f"Invalid sentiment for review {review_id}: {sentiment}"
            )

        if intensity not in {"Low", "Medium", "High"}:
            raise ValueError(
                f"Invalid intensity for review {review_id}: {intensity}"
            )

        results.append({
            "review_id": review_id,
            "sentiment": sentiment,
            "intensity": intensity,
        })

    return results
