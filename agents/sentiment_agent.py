
from crewai import Agent
from agents.llm_config import llm

sentiment_agent = Agent(
    role="Customer Sentiment Analysis Specialist",
    goal=(
        "Identify the sentiment expressed in each customer review "
        "and explain its intensity."
    ),
    backstory=(
        "You are a sentiment analysis expert specializing in "
        "e-commerce reviews. You distinguish positive, negative "
        "and neutral opinions using the actual review text."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)
