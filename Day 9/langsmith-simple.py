import os
from openai import OpenAI
from langsmith import wrappers, traceable

os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-simple-demo"

client = wrappers.wrap_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=""
))

@traceable
def ask_openai(topic: str):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": f"Explain {topic} in simple terms."}]
    )
    return response.choices[0].message.content

@traceable
def ask_claude(openai_output: str):
    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        messages=[{"role": "user", "content": f"Summarize this in one line: {openai_output}"}]
    )
    return response.choices[0].message.content

@traceable
def pipeline(topic: str):
    step1 = ask_openai(topic)
    print(f"OpenAI explanation: {step1}\n")

    step2 = ask_claude(step1)
    print(f"Claude summary: {step2}")

pipeline("black holes")