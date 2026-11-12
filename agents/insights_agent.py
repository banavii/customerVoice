
from crewai import Agent
from agents.llm_config import business_llm

insights_agent = Agent(
    role="Customer Insights and Trend Analyst",
    goal=(
        "Identify meaningful customer feedback patterns "
        "from classification and sentiment results."
    ),
    backstory=(
        "You are an experienced customer experience analyst. "
        "You distinguish observed customer feedback from "
        "possible explanations and unsupported assumptions. "
        "You avoid generalizing from small review samples."
    ),
    llm=business_llm,
    verbose=False,
    allow_delegation=False,
)
