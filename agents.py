import os
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

def run_customervoice_analysis(sample_size=15):
    # Load processed dataset
    df = pd.read_csv("data/processed/cleaned_reviews.csv")
    
    # Filter for low ratings (1-3 stars)
    negative_reviews = df[df["rating"] <= 3].head(sample_size)
    reviews_text = "\n".join([
        f"- [ID: {row['product_id']}, Category: {row['product_category']}, Rating: {row['rating']}] {row['review_text']}" 
        for _, row in negative_reviews.iterrows()
    ])

    # Configure LM Studio Local LLM
    qwen_llm = LLM(
        model="openai/qwen2.5",
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define Agents
    sentiment_agent = Agent(
        role="Customer Feedback Categorizer & Sentiment Analyst",
        goal="Categorize raw customer reviews by theme and analyze granular sentiment.",
        backstory="You are an expert customer experience analyst...",
        llm=qwen_llm,
        verbose=True
    )

    trend_agent = Agent(
        role="Emerging Trend & Pain-Point Detector",
        goal="Identify recurring customer complaints and root causes.",
        backstory="You are a senior product researcher...",
        llm=qwen_llm,
        verbose=True
    )

    insights_agent = Agent(
        role="Actionable Business Recommendations Strategist",
        goal="Synthesize customer pain points into actionable steps.",
        backstory="You are a Chief Operations Officer...",
        llm=qwen_llm,
        verbose=True
    )

    # Define Tasks
    task_categorize = Task(
        description=f"Analyze the following list of customer reviews and categorize each review into standard themes:\n\n{reviews_text}",
        expected_output="A structured summary breaking down reviews into categories with key recurring mentions.",
        agent=sentiment_agent
    )

    task_identify_trends = Task(
        description="Review categorized feedback and identify top 3-5 critical pain points and root causes.",
        expected_output="A prioritized list of customer pain points with root-cause analysis.",
        agent=trend_agent
    )

    task_generate_recommendations = Task(
        description="Draft clear, prioritized action steps for Product, QA, and Support teams based on the findings.",
        expected_output="An executive recommendation report outlining immediate fixes and business impact.",
        agent=insights_agent
    )

    # Form & Execute Crew
    crew = Crew(
        agents=[sentiment_agent, trend_agent, insights_agent],
        tasks=[task_categorize, task_identify_trends, task_generate_recommendations],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)

if __name__ == "__main__":
    print(run_customervoice_analysis(10))