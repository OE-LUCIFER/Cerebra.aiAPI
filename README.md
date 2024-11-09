<div align="center">

# 🧠 Cerebra.aiAPI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green)](https://openai.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**An unofficial, feature-rich API for Cerebra.ai with OpenAI compatibility and advanced tools.**

[🚀 Features](#-features) • [🛠️ Installation](#️-installation) • [💻 Usage](#-usage) • [🖥️ Server](#️-server) • [🔌 OpenAI Compatibility](#-openai-compatibility) • [🧰 Tools](#-tools) • [📊 Benchmarks](#-benchmarks) • [🤝 Contributing](#-contributing) • [📄 License](#-license)

</div>

---

## 🚀 Features

- 🔥 Seamless integration with Cerebra.ai models
- 🔄 OpenAI-compatible API endpoints
- 🧠 Access to state-of-the-art language models
- 🛠️ Advanced tools for enhanced capabilities
- 🖥️ Built-in server with FastAPI
- 📊 Detailed usage metrics and quotas
- 🔒 Secure cookie-based authentication
- 📡 Real-time streaming responses

---

## 🛠️ Installation

### Prerequisites

- Python 3.7+- Python 3.7+
- [Cookie-Editor](https://cookie-editor.cgagnier.ca/) extension

### Step-by-step Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OE-LUCIFER/Cerebra.aiAPI.git
   cd Cerebra.aiAPI
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your cookies:**
   - Visit the Cerebra.ai website
   - Use Cookie-Editor to export cookies
   - Save as `cookies.json` in the project root

---

## 💻 Usage

### Quick Start

```python
from cerebra_ai_api import CerebrasWithCookie

client = CerebrasWithCookie(cookie_path='cookies.json')

response = client.ask("Explain the theory of relativity in simple terms.")
print(response)
```

### Advanced Usage

<details>
<summary>Click to expand</summary>

#### Streaming Responses

```python
stream = client.generate_stream(
    ChatRequest(
        messages=[{"role": "user", "content": "Write a haiku about AI."}],
        model="llama3.1-70b",
        stream=True
    )
)

for chunk in stream:
    print(chunk['data'], end='', flush=True)
```

#### Using Different Models

```python
response_8b = client.ask("Summarize the importance of quantum computing.", model="llama3.1-8b")
response_70b = client.ask("Summarize the importance of quantum computing.", model="llama3.1-70b")

print("8B Model:", response_8b)
print("70B Model:", response_70b)
```

#### JSON Responses

```python
json_response = client.ask(
    "List the top 5 programming languages in 2023.",
    json_response=True
)
print(json_response)
```

</details>

---

## 🖥️ Server

### Starting the Server

```python
client = CerebrasWithCookie(cookie_path='cookies.json')
client.start_server(host="0.0.0.0", port=8000)
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/v1/chat/completions` | Chat completions API |
| `/v1/models` | List available models |

### Swagger Documentation

Access the interactive API documentation at `http://localhost:8000/docs` when the server is running.

---

## 🔌 OpenAI Compatibility

Cerebra.aiAPI is designed as a drop-in replacement for the OpenAI Python library.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy_key"
)

response = client.chat.completions.create(
    model="llama3.1-70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the concept of machine learning."}
    ]
)

print(response.choices[0].message.content)
```

---

## 🧰 Tools

Cerebra.aiAPI supports tool calling.


### Example: Web Search Tool

```python
def get_web_info(query: str, max_results: int = 5) -> str:
    results = search(query, num_results=max_results)
    return json.dumps([{"title": r.title, "link": r.url, "snippet": r.description} for r in results])

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_web_info",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results"}
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="llama3.1-70b",
    messages=[{"role": "user", "content": "What are the latest developments in AI?"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.content)
```

<details>
<summary>More Tool Examples</summary>

### Calculator Tool

```python
def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Usage in tools list
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The mathematical expression to evaluate"}
            },
            "required": ["expression"]
        }
    }
}
```

### Weather Information Tool

```python
import requests

def get_weather(city: str) -> str:
    API_KEY = "your_openweathermap_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return f"The current temperature in {city} is {data['main']['temp']}°C with {data['weather'][0]['description']}."

# Usage in tools list
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city"}
            },
            "required": ["city"]
        }
    }
}
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/OE-LUCIFER/Cerebra.aiAPI.git`
3. Create a new **branch**: `git checkout -b feature-name`
4. Make your **changes** and **commit** them: `git commit -m 'Add some feature'`
5. **Push** to the branch: `git push origin feature-name`
6. Submit a **pull request**



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by Team HelpingAI**

⭐️ Star us on GitHub — it motivates us a lot!⭐️ Star us on GitHub — it motivates us a lot!⭐️ Star us on GitHub — it motivates us a lot!⭐️ Star us on GitHub — it motivates us a lot!

[Report Bug](https://github.com/OE-LUCIFER/Cerebra.aiAPI/issues)

</div>

