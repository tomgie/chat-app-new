import os
from pathlib import Path
from typing import List, Literal, Optional
from jinja2 import Environment, FileSystemLoader, TemplateError
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, gt=0)
    top_p: Optional[float] = Field(default=0.95, ge=0, le=1)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = Field(default=False)


class ChatResponse(BaseModel):
    response: str
    usage: dict


class StreamingChatResponse(BaseModel):
    text: str
    usage: Optional[dict] = None
    done: bool = False


class CustomEnvironment(Environment):
    def raise_exception(self, message):
        raise TemplateError(message)


def create_jinja_env():
    """Create Jinja environment with custom settings"""
    templates_dir = Path(__file__).parent / 'templates'
    loader = FileSystemLoader(templates_dir)
    env = CustomEnvironment(loader=loader)
    env.globals['raise_exception'] = env.raise_exception
    return env

def format_prompt(messages: List[Message]) -> str:
    """Format messages using the Jinja2 template from file."""
    message_dicts = [message.dict() for message in messages]

    try:
        env = create_jinja_env()
        template = env.get_template('mistral-instruct.jinja')
        return template.render(
            messages=message_dicts,
            bos_token="<s>",
            eos_token="</s>",
        )
    except TemplateError as e:
        raise ValueError(f"Template error: {str(e)}")
    except FileNotFoundError:
        raise ValueError("Template file 'mistral-instruct.jinja' not found in templates directory")


app = FastAPI()
model: Llama | None = None


def initialize_model():
    model_path = Path('models/Cydonia-22B-v2q-Q2_K.gguf')
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found at {model_path}')

    try:
        return Llama(
            model_path=str(model_path),
            n_gpu_layers=0,
            n_threads=os.cpu_count(),
            n_ctx=8192,
            n_batch=512,
            verbose=True
        )
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise RuntimeError(f'Failed to load the LLM model: {str(e)}')

@app.on_event('startup')
async def startup_event():
    global model
    model = initialize_model()

async def generate_streaming_response(
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str]
):
    """Generator function for streaming responses"""
    try:
        stream = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
            echo=False
        )

        current_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for chunk in stream:
            if chunk:
                token = chunk['choices'][0]['text']
                if token:
                    # Update token counts
                    current_usage["completion_tokens"] += 1
                    current_usage["total_tokens"] += 1

                    # Yield the chunk
                    yield StreamingChatResponse(
                        text=token,
                        usage=current_usage,
                        done=False
                    ).json() + "\n"

        # Send final message indicating completion
        yield StreamingChatResponse(
            text="",
            usage=current_usage,
            done=True
        ).json() + "\n"

    except Exception as e:
        yield StreamingChatResponse(
            text=f"Error during streaming: {str(e)}",
            done=True
        ).json() + "\n"

@app.post('/v1/chat')
async def chat_endpoint(request: ChatRequest):
    if not model:
        raise HTTPException(status_code=503, detail='Model not initialized')

    try:
        prompt = format_prompt(request.messages)
        stop_sequences = request.stop or ['[INST]', '</s>']

        # Handle streaming response
        if request.stream:
            return StreamingResponse(
                generate_streaming_response(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop_sequences
                ),
                media_type='text/event-stream'
            )

        # Handle non-streaming response
        response = model(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop_sequences,
            echo=False
        )

        generated_text = response['choices'][0]['text']
        usage = {
            'prompt_tokens': response['usage']['prompt_tokens'],
            'completion_tokens': response['usage']['completion_tokens'],
            'total_tokens': response['usage']['total_tokens']
        }

        return ChatResponse(
            response=generated_text.strip(),
            usage=usage
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))