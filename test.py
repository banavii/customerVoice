import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model="qwen2.5", # or any loaded model in LM Studio
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)