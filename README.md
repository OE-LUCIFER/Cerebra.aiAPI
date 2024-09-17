# Cerebra.aiAPI 🌟✨

Welcome to the **Cerebra.aiAPI**! 🎉 This is an unofficial API designed for users currently on the waitlist for an official API key. With this API, you can enjoy similar limits and quotas as those provided with an official key, ensuring a smooth transition once you gain access to the official API. 🚀💫

## Prerequisites 🛠️

Before you dive in, make sure you have the following:

1. **Python** installed on your machine. 🐍
2. Installation of Requirements 📦

To install all required packages, you can use:

```
pip install -r requirements.txt
```


3. Install the **Cookie-Editor** extension for either Chrome or Edge:
   - [Chrome Extension](https://chrome.google.com/webstore/detail/cookie-editor/...)
   - [Edge Extension](https://microsoftedge.microsoft.com/addons/detail/cookie-editor/...)

### Exporting Cookies 🍪

Once you have the Cookie-Editor installed, follow these steps to export your cookies:

1. Open the website from which you want to extract cookies. 🌐
2. Click on the Cookie-Editor extension icon. 🔍
3. Click the "Export" button to save your cookies in JSON format. 💾
4. Create a file in your working directory named `cookies.json`. 🗂️
5. Paste the data copied from the Cookie-Editor into `cookies.json` and save it. ✨

## Quota and Usage Limits 📊

The following limits apply for different models when using this API:

### Llama 3.1-8B Model (default):
- **Requests:**
  - Per minute: 30 ⏱️
  - Per hour: 900 ⏳
  - Per day: 14,400 📅
- **Tokens:**
  - Per minute: 60,000 💬
  - Per hour: 1,000,000 🗨️
  - Per day: 1,000,000 📝

### Llama 3.1-70B Model:
- **Requests:**
  - Per minute: 30 ⏱️
  - Per hour: 900 ⏳
  - Per day: 14,400 📅
- **Tokens:**
  - Per minute: 60,000 💬
  - Per hour: 1,000,000 🗨️
  - Per day: 1,000,000 📝

## Example Usage 💻💖

By default, the API uses the Llama 3.1-8B model. You can also switch to other models like Llama 3.1-70B if required. Let’s explore some fun examples! 🎈

### 🌈 Storytelling Example

Imagine you want to ask the API to tell you a heartwarming story. Here’s how you can do it:

```python
# Using the Llama 3.1-70B model
llm_client = CerebrasWithCookie(cookie_path='/your/path/to/cookies.json', model='llama3.1-70b')
response = llm_client.ask("🌟 Can you tell me a magical story about a brave little girl and her talking cat? 🐱✨")
print(response)
```

### 🥳 JSON Response Example

Let’s say you want to generate a list of mythical creatures in JSON format. Here’s how you can request that:

```python
# Using the Llama 3.1-70B model
llm_client = CerebrasWithCookie(cookie_path='/your/path/to/cookies.json', model='llama3.1-70b')
response = llm_client.ask("""
🦄 Generate a list of 5 mythical creatures in the following JSON format:
'''json
{
  "creatures": [
    {
      "name": "string",
      "type": "string",
      "habitat": "string"
    },
  ]
}
'''
Where:
- `"name"` is the name of the creature.
- `"type"` is the type of creature (e.g., dragon, fairy).
- `"habitat"` is where the creature is commonly found.
""", json_response=True)
print(response)
```

### 🎉 Fun Facts Example

Want to learn something new? Ask the API for fun facts! Here’s how:

```python
# Using the Llama 3.1-8B model# Using the Llama 3.1-8B model
llm_client = CerebrasWithCookie(cookie_path='/your/path/to/cookies.json', model='llama3.1-8b')
response = llm_client.ask("🤔 What are some fascinating facts about space? 🌌✨")
print(response)
```


## Contributing 🤝💕

We welcome contributions! If you have suggestions or improvements, feel free to open an issue or submit a pull request. Your input means the world to us! 🌍❤️

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for using **Cerebra.aiAPI**! We hope you find it helpful and enjoyable! If you have any questions, feel free to reach out. Happy coding! 🎊💖
