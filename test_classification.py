
from crewai import Crew, Process

from agents.classification_agent import classification_agent
from tasks.classification_task import create_classification_task

review = (
    "The dress looks beautiful, but the fabric feels cheap "
    "and the stitching started coming apart after one wash."
)

task = create_classification_task(
    classification_agent,
    review
)

crew = Crew(
    agents=[classification_agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()

print("\n--- CUSTOMERVOICE CLASSIFICATION RESULT ---")
print(result)
