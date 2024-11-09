print("\n==== CerebrasWithCookie ==== \n")
from main import CerebrasWithCookie
client = CerebrasWithCookie(cookie_path="cookie.json")
response = client.ask(question="Explain quantum computing in simple terms.", sys_prompt="You are a helpful assistant.")
print(response)


import json
from openai_unofficial import OpenAIUnofficial
from rich import print
print("\n==== OPENAI UNOFFICIAL ==== \n")
client = OpenAIUnofficial(base_url="http://localhost:8000/v1")

models = client.list_models()
print("\n==== Available Models ==== \n")
for model in models['data']:
    print(f"- {model['id']}")
# Basic chat completion
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Say hello!"}],
    model="llama3.1-70b"
)
print("\n ==== NON STREAMING ==== \n")
print(response.choices[0].message.content)
print("\n ==== STREAMING ==== \n")
completion_stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a short story in 3 sentences."}],
    model="llama3.1-70b",
    stream=True
)
for chunk in completion_stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end='', flush=True)
print("\n ==== END OF STREAM ==== \n")

from openai import OpenAI
import json
from datetime import datetime

print("\n==== OPENAI ==== \n")

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Your Cerebras FastAPI server URL
    api_key="dummy_key"  # Not used, but required by OpenAI package
)

print("\n ==== NON STREAMING ==== \n")

response = client.chat.completions.create(
    model="llama3.1-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]    ]
)
print(response.choices[0].message.content)

print("\n ==== STREAMING ==== \n")

stream = client.chat.completions.create(
    model="llama3.1-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about technology."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\n ==== END OF STREAM ==== \n")



