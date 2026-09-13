import os
from crewai import LLM as CrewAILLM
from autogen_ext.models.ollama import OllamaChatCompletionClient
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

def get_crewai_ollama_llm() -> CrewAILLM:
    """Configures CrewAI to use Ollama via native OpenAI-compatible mode."""
    return CrewAILLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

def get_autogen_ollama_client() -> OllamaChatCompletionClient:
    """Configures AutoGen v0.4 model client for Ollama."""
    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen2.5"
        }
    )

def get_langchain_ollama_chat() -> ChatOllama:
    """Configures LangChain ChatOllama with enlarged context window."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192
    )