from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Generator, Union
import requests
import json
import re
from functools import lru_cache
from cerebras.cloud.sdk import Cerebras
from fake_useragent import UserAgent
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"

class ChatMessage(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = Field(default="llama3.1-8b")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)
    tools: Optional[List[Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None

    @model_validator(mode='before')
    def validate_tools_and_functions(cls, values):
        if values.get("tools") and values.get("functions"):
            raise ValueError("Cannot provide both 'tools' and 'functions' parameters")
        return values

class CerebrasWithCookie:
    """Enhanced Cerebras client with OpenAI compatibility and advanced features"""
    
    def __init__(self, 
                 cookie_path: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 default_model: str = "llama3.1-8b"):
        """
        Initialize the Cerebras client with either cookie path or API key.
        
        Args:
            cookie_path: Path to cookie file for authentication
            api_key: Direct API key for authentication
            default_model: Default model to use
        """
        self.default_model = default_model
        if api_key:
            self.api_key = api_key
        elif cookie_path:
            self.api_key = self._get_demo_api_key(cookie_path)
        else:
            raise ValueError("Either cookie_path or api_key must be provided")
        
        self.client = Cerebras(api_key=self.api_key)

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract code block from text"""
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

    def _get_demo_api_key(self, cookie_path: str) -> str:
        """Get demo API key using cookie authentication"""
        try:
            with open(cookie_path, 'r') as file:
                cookies = {item['name']: item['value'] for item in json.load(file)}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Cookie file error: {str(e)}")

        headers = {
            'Accept': '*/*',
            'Content-Type': 'application/json',
            'Origin': 'https://inference.cerebras.ai',
            'Referer': 'https://inference.cerebras.ai/',
            'user-agent': UserAgent().random
        }

        try:
            response = requests.post(
                'https://inference.cerebras.ai/api/graphql',
                cookies=cookies,
                headers=headers,
                json={
                    'operationName': 'GetMyDemoApiKey',
                    'variables': {},
                    'query': 'query GetMyDemoApiKey {\n  GetMyDemoApiKey\n}'
                }
            )
            response.raise_for_status()
            return response.json()['data']['GetMyDemoApiKey']
        except Exception as e:
            raise ValueError(f"Failed to get API key: {str(e)}")

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Cerebras format"""
        prepared_messages = []
        for msg in messages:
            message_dict = {"role": msg.role, "content": msg.content}
            if msg.name:
                message_dict["name"] = msg.name
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            if msg.function_call:
                message_dict["function_call"] = msg.function_call
            prepared_messages.append(message_dict)
        return prepared_messages

    def generate_stream(self, request: ChatRequest) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming response"""
        try:
            stream = self.client.chat.completions.create(
                messages=self._prepare_messages(request.messages),
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                tools=request.tools,
                response_format=request.response_format
            )

            for chunk in stream:
                if chunk.choices[0].delta.content or getattr(chunk.choices[0].delta, 'tool_calls', None):
                    yield {
                        "data": json.dumps({
                            "id": chunk.id,
                            "object": "chat.completion.chunk",
                            "created": chunk.created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": chunk.choices[0].delta.content,
                                    "tool_calls": getattr(chunk.choices[0].delta, 'tool_calls', None)
                                },
                                "finish_reason": chunk.choices[0].finish_reason
                            }]
                        })
                    }
            
            yield {"data": "[DONE]"}

        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def generate(self, request: ChatRequest) -> Dict[str, Any]:
        """Generate non-streaming response"""
        try:
            response = self.client.chat.completions.create(
                messages=self._prepare_messages(request.messages),
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
                tools=request.tools,
                response_format=request.response_format
            )
            return response

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def ask(self, 
            question: str, 
            sys_prompt: str = "You are a helpful assistant.", 
            json_response: bool = False,
            tools: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        Simple interface for asking questions
        
        Args:
            question: Question to ask
            sys_prompt: System prompt
            json_response: Whether to format response as JSON
            tools: Optional tools to use
        """
        messages = [
            ChatMessage(content=sys_prompt, role=Role.SYSTEM),
            ChatMessage(content=question, role=Role.USER)
        ]

        request = ChatRequest(
            messages=messages,
            model=self.default_model,
            response_format={"type": "json_object"} if json_response else None,
            tools=tools
        )

        try:
            response = self.generate(request)
            content = response.choices[0].message.content
            return self._extract_code_block(content) if json_response else content
        except Exception as e:
            logger.error(f"Ask error: {str(e)}")
            return None

# FastAPI Server Setup
app = FastAPI(title="Cerebras API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache()
def get_cerebras_client():
    """Cached Cerebras client instance"""
    return CerebrasWithCookie(cookie_path='cookie.json')

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatRequest,
    client: CerebrasWithCookie = Depends(get_cerebras_client)
):
    """OpenAI-compatible chat completions endpoint"""
    if request.stream:
        generator = client.generate_stream(request)
        return EventSourceResponse(generator)
    return client.generate(request)


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "llama3.1-8b",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "cerebras"
            },
            {
                "id": "llama3.1-70b",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "cerebras"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
