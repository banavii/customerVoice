
from crewai import Agent
from agents.llm_config import review_llm

classification_agent = Agent(
    role="Customer Feedback Classification Specialist",
    goal=(
        "Classify each customer review into the most "
        "relevant e-commerce feedback category."
    ),
    backstory=(
        "You analyze customer reviews accurately and "
        "consistently. You classify feedback using only "
        "the information provided by customers."
    ),
    llm=review_llm,
    verbose=False,
    allow_delegation=False,
)
