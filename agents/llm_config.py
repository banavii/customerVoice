
import os

from crewai import LLM
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model = os.getenv("LLM_MODEL")
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not all([model, base_url, api_key]):
    raise ValueError(
        "Missing LLM_MODEL, OPENAI_BASE_URL or OPENAI_API_KEY in .env"
    )

# Two configurations of the same installed Qwen model.
review_llm = LLM(
    model=model,
    base_url=base_url,
    api_key=api_key,
    temperature=0.1,
)

business_llm = LLM(
    model=model,
    base_url=base_url,
    api_key=api_key,
    temperature=0.3,
)

# Direct client for sentiment analysis, outside CrewAI.
direct_client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=120.0,
    max_retries=1,
)

# CrewAI uses the "openai/" provider prefix, while
# the direct LM Studio endpoint uses the actual model ID.
direct_model = os.getenv("DIRECT_LLM_MODEL")

if not direct_model:
    direct_model = model.removeprefix("openai/")
