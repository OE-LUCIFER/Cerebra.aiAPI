import requests
import json
import re
from cerebras.cloud.sdk import Cerebras
from typing import Optional
from fake_useragent import UserAgent

class CerebrasWithCookie:
    def __init__(self, cookie_path: str, model: str = 'llama3.1-8b'):
        """
        Initializes the Cerebras client with the provided cookie.

        :param cookie_path: Path to the cookie JSON file.
        :param model: Model name to use. Defaults to 'llama3.1-8b'.
        """
        self.api_key = self.get_demo_api_key(cookie_path)
        self.client = Cerebras(api_key=self.api_key)
        self.model = model

    @staticmethod
    def extract_query(text: str) -> str:
        """
        Extracts the first code block from the given text.

        :param text: Input text containing code blocks.
        :return: Extracted code block or the original text if none found.
        """
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

    @staticmethod
    def refiner(text: str) -> str:
        """
        Refines the input text by removing surrounding quotes.

        :param text: The text to refine.
        :return: Refined text.
        """
        return text.strip('"')

    def get_demo_api_key(self, cookie_path: str) -> str:
        """
        Retrieves the demo API key using the provided cookie.

        :param cookie_path: Path to the cookie JSON file.
        :return: Demo API key.
        """
        try:
            with open(cookie_path, 'r') as file:
                cookies = {item['name']: item['value'] for item in json.load(file)}
        except FileNotFoundError:
            raise(f"Cookie file not found at path: {cookie_path}")
        except json.JSONDecodeError:
            raise("Invalid JSON format in the cookie file.")

        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'Origin': 'https://inference.cerebras.ai',
            'Referer': 'https://inference.cerebras.ai/',
            'user-agent': UserAgent().random
        }

        json_data = {
            'operationName': 'GetMyDemoApiKey',
            'variables': {},
            'query': 'query GetMyDemoApiKey {\n  GetMyDemoApiKey\n}',
        }

        try:
            response = requests.post(
                'https://inference.cerebras.ai/api/graphql',
                cookies=cookies,
                headers=headers,
                json=json_data
            )
            response.raise_for_status()
            api_key = response.json()['data']['GetMyDemoApiKey']
            return api_key
        except requests.RequestException as e:
            raise(f"Failed to retrieve API key: {e}")
        except KeyError:
            pass

    def ask(self, question: str, 
            sys_prompt: str = "You are a helpful assistant.", 
            json_response: bool = False) -> Optional[str]:
        """
        Sends a question to the Cerebras model and retrieves the answer.

        :param question: The question to ask.
        :param sys_prompt: System prompt to guide the model.
        :param json_response: Whether to expect a JSON response.
        :return: The model's answer or None if extraction fails.
        """
        messages = [
            {'content': sys_prompt, 'role': 'system'},
            {'content': question, 'role': 'user'}
        ]

        try:
            if json_response:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model
                )
            content = response.choices[0].message.content
            extracted = self.extract_query(content)
            return extracted
        except Exception as e:

            return None

if __name__ == "__main__":

    cerebras = CerebrasWithCookie('cookie.json','llama3.1-8b')

    response = cerebras.ask("What is the meaning of life?", sys_prompt='')
    for chunk in response:
        print(chunk, end="", flush=True)
