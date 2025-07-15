# backend/agent/llm_setup.py or similar
import os
from langchain_ollama import OllamaLLM as Ollama
from langchain.llms.fake import FakeListLLM
from backend.agent.tracing_wrapper import wrap_run  # optional wrapper for tracing

def get_llm():
    if os.getenv("CI") == "true":
        return FakeListLLM(responses=["CSV", "DOCX", "CSV", "DOCX"])
    
    llm = Ollama(model="qwen3:32b", temperature=0.2)
    return wrap_run(llm)  # remove wrap_run if unused
