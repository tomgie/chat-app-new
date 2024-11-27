import os
import json
from pathlib import Path
from typing import List, Literal, Optional
from jinja2 import Environment, FileSystemLoader, TemplateError
from pydantic import BaseModel, Field
from llama_cpp import Llama
import runpod
from runpod.serverless.utils import rp_cleanup
import torch


# Pydantic models
class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class GenerationParams(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, gt=0)
    top_p: Optional[float] = Field(default=0.95, ge=0, le=1)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = Field(default=False)


class CustomEnvironment(Environment):
    def raise_exception(self, message):
        raise TemplateError(message)


# Global variables
MODEL = None


def create_jinja_env():
    """Create Jinja2 environment with custom settings"""
    templates_dir = Path(__file__).parent / 'templates'
    loader = FileSystemLoader(templates_dir)
    env = CustomEnvironment(loader=loader)
    env.globals['raise_exception'] = env.raise_exception
    return env


def format_prompt(messages: List[Message]) -> str:
    """Format messages using the Jinja2 template"""
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
        raise ValueError("Template file not found")


def load_model():
    """Initialize the model with GPU acceleration"""
    global MODEL

    if MODEL is not None:
        return MODEL

    model_path = Path('/runpod-volume/Donnager-70B-v1-Q4_K_M.gguf')
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found at {model_path}')

    try:
        MODEL = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,  # Use all layers on GPU
            n_threads=os.cpu_count(),
            n_ctx=8192,
            n_batch=512,
            verbose=True,
            offload_kqv=True,
            use_mmap=False,
            use_mlock=False,
        )
        return MODEL
    except Exception as e:
        raise RuntimeError(f'Failed to load model: {str(e)}')


def generate_response(prompt: str, params: GenerationParams):
    """Generate response from the model"""
    model = load_model()
    stop_sequences = params.stop or ['[INST]', '</s>']

    try:
        response = model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=stop_sequences,
            echo=False
        )

        generated_text = response['choices'][0]['text']
        usage = {
            'prompt_tokens': response['usage']['prompt_tokens'],
            'completion_tokens': response['usage']['completion_tokens'],
            'total_tokens': response['usage']['total_tokens']
        }

        return {
            "response": generated_text.strip(),
            "usage": usage
        }
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")


# RunPod handler
def handler(event):
    """
    RunPod serverless handler function
    """
    try:
        # Parse and validate input
        job_input = event["input"]

        # Convert input to Pydantic model
        params = GenerationParams(**job_input)

        # Format prompt
        prompt = format_prompt(params.messages)

        # Generate response
        result = generate_response(prompt, params)

        return runpod.ServerlessResponse(
            status_code=200,
            json=result
        )

    except Exception as e:
        return runpod.ServerlessResponse(
            status_code=500,
            json={"error": str(e)}
        )
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rp_cleanup.clean()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})