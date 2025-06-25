from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import os 
from dotenv import load_dotenv
import openai
import asyncio
from abc import ABC, abstractmethod
from anthropic import Anthropic

load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class AIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ChatRequest(BaseModel):
    prompt: str
    provider: AIProvider

class AIProviderInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass    

class OpenAIProvider(AIProviderInterface):
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    async def generate(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        def _sync_generate():
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        return await loop.run_in_executor(None, _sync_generate)
    
class AnthropicProvider(AIProviderInterface):
    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    async def generate(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        def _sync_generate():
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.content[0].text
            except Exception as e:
                raise e
        return await loop.run_in_executor(None, _sync_generate)



@app.post("/chat")
async def chat(request: ChatRequest):
    if request.provider == AIProvider.OPENAI:
        provider = OpenAIProvider()
    elif request.provider == AIProvider.ANTHROPIC:
        provider = AnthropicProvider()
    else:
        response = "Hello from unknown provider"

    response = await provider.generate(request.prompt)
    return {
        "provider":request.provider,
        "response":response,
    }