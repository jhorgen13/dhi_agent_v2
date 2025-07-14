from langchain_ollama import OllamaLLM as Ollama

# --- Basic LLM Setup (no tracing) --- #
def get_llm():
    llm = Ollama(model="qwen3:32b", temperature=0.2)
    return wrap_run(llm)

# --- Simple LLM/Agent Invocation --- #
def call_llm(llm_or_agent, prompt: str):
    """
    Runs the LLM or agent without LangSmith tracing.
    """
    return llm_or_agent.invoke(prompt)
