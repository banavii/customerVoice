
from crewai import Agent
from agents.llm_config import business_llm

recommendation_agent = Agent(
    role="Business Recommendations Strategist",
    goal=(
        "Turn verified customer feedback findings into "
        "practical, prioritized business recommendations."
    ),
    backstory=(
        "You are an e-commerce business strategist. "
        "You propose realistic improvements while clearly "
        "separating confirmed problems from issues that "
        "require further investigation."
    ),
    llm=business_llm,
    verbose=False,
    allow_delegation=False,
)
