import threading
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generator, Iterable
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
import argparse
import signal
import asyncio
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"

class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'arguments': self.arguments
        }

class ToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[FunctionCall] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'function': self.function.to_dict() if self.function else None
        }

class ChatMessage(BaseModel):
    role: Role
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    audio: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        message_dict = {'role': self.role}
        if self.content is not None:
            message_dict['content'] = self.content
        if self.function_call is not None:
            message_dict['function_call'] = self.function_call.to_dict()
        if self.tool_calls:
            message_dict['tool_calls'] = [tool_call.to_dict() for tool_call in self.tool_calls]
        if self.audio is not None:
            message_dict['audio'] = self.audio
        if self.role == 'tool':
            if self.tool_call_id:
                message_dict['tool_call_id'] = self.tool_call_id
            if self.name:
                message_dict['name'] = self.name
        return message_dict

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = Field(default="llama3.1-8b")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)
    response_format: Optional[Dict[str, str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'message': self.message.to_dict(),
            'finish_reason': self.finish_reason
        }

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'object': self.object,
            'created': self.created,
            'model': self.model,
            'choices': [choice.to_dict() for choice in self.choices],
            'usage': self.usage
        }

class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List['ChatCompletionChunkChoice']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'object': self.object,
            'created': self.created,
            'model': self.model,
            'choices': [choice.to_dict() for choice in self.choices]
        }

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'delta': self.delta.to_dict(),
            'finish_reason': self.finish_reason
        }

class CerebrasWithCookie:
    def __init__(self, 
                 cookie_path: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 default_model: str = "llama3.1-8b"):
        self.default_model = default_model
        if api_key:
            self.api_key = api_key
        elif cookie_path:
            self.api_key = self._get_demo_api_key(cookie_path)
        else:
            raise ValueError("Either cookie_path or api_key must be provided")
        
        self.client = Cerebras(api_key=self.api_key)
        self.server = None
        self.server_thread = None

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
        return [message.to_dict() for message in messages]

    def generate_stream(self, request: ChatRequest) -> Generator[str, None, None]:
        """Generate streaming response with complete delta fields"""
        
        stream = self.client.chat.completions.create(
            messages=self._prepare_messages(request.messages),
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            delta_dict = delta.to_dict() if hasattr(delta, 'to_dict') else delta
            
            # Ensure 'role' is present in delta_dict
            if 'role' not in delta_dict:
                delta_dict['role'] = 'assistant'  # Default role for delta messages
            
            yield {
                "data": json.dumps(ChatCompletionChunk(
                    id=chunk.id,
                    object="chat.completion.chunk",
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(**delta_dict),
                            finish_reason=chunk.choices[0].finish_reason
                        )
                    ]
                ).to_dict())
            }

        yield {"data": "[DONE]"}

    def generate(self, request: ChatRequest) -> ChatCompletionResponse:
        """Generate non-streaming response"""
        try:
            response = self.client.chat.completions.create(
                messages=self._prepare_messages(request.messages),
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
                response_format=request.response_format,
                tools=request.tools,
                tool_choice=request.tool_choice
            )
            return ChatCompletionResponse(**response.to_dict())
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def ask(self, 
            question: str, 
            sys_prompt: str = "You are a helpful assistant.", 
            json_response: bool = False,
            ) -> Optional[str]:
        """
        Simple interface for asking questions
        
        Args:
            question: Question to ask
            sys_prompt: System prompt
            json_response: Whether to format response as JSON
        """
        messages = [
            ChatMessage(content=sys_prompt, role=Role.SYSTEM),
            ChatMessage(content=question, role=Role.USER)
        ]
        request = ChatRequest(
            messages=messages,
            model=self.default_model,
            response_format={"type": "json_object"} if json_response else None,
        )
        try:
            response = self.generate(request)
            content = response.choices[0].message.content
            return self._extract_code_block(content) if json_response else content
        except Exception as e:
            logger.error(f"Ask error: {str(e)}")
            return None

    async def stop_server(self):
        """Stops the uvicorn server gracefully and kills the server thread."""
        if self.server:
            await self.server.shutdown()
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)  # Wait for up to 5 seconds
            if self.server_thread.is_alive():
                logger.warning("Server thread did not terminate gracefully. Forcing termination.")
                os.kill(os.getpid(), signal.SIGINT)  # Force termination if thread doesn't join

    def start_server(self, host="0.0.0.0", port=8000):
        """Starts the FastAPI server in a separate thread."""
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        self.server = uvicorn.Server(config)

        def run_server():
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.run(self.server.serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"Server started on {host}:{port}")

    def stop_server_sync(self):
        """Synchronous wrapper to stop the server from non-async contexts."""
        asyncio.run(self.stop_server())

    def run(self, host="0.0.0.0", port=8000):
        """Run the server and set up signal handlers"""
        def signal_handler(signum, frame):
            logger.info("Received termination signal. Shutting down...")
            self.stop_server_sync()
            exit(0)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.start_server(host=host, port=port)

        # Keep the main thread alive
        try:
            while True:
                asyncio.run(asyncio.sleep(1))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
            self.stop_server_sync()

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
            {"id": "llama3.1-8b", "object": "model", "created": int(datetime.now().timestamp()), "owned_by": "cerebras"},
            {"id": "llama3.1-70b", "object": "model", "created": int(datetime.now().timestamp()), "owned_by": "cerebras"}
        ]
    }
