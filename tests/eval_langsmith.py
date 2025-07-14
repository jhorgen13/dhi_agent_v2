# tests/eval_langsmith.py
from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import Client
from agents.router_agent import answer_query, router_chain
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

client = Client()

# Wrap target
def target_fn(inputs: dict) -> dict:
    result = answer_query(inputs["question"])
    return {"answer": result.get("answer", "")}

# Accuracy evaluator using OpenAI judge
def accuracy_evaluator(inputs, outputs, reference):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:gpt-3.5-turbo",
        feedback_key="accuracy"
    )
    return evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference)

# Routing evaluator
def routing_evaluator(inputs, outputs, reference, metadata):
    route = router_chain.invoke({"query": inputs["question"]})
    chosen = route.get("destination", "").upper()
    expected = metadata.get("expected_agent", "").upper()
    return {"routing": 1.0 if chosen == expected else 0.0}

if __name__ == "__main__":
    os.environ["LANGSMITH_TRACING"] = "true"
    dataset = client.read_dataset(name="MultiAgent QA Test Set")
    results = client.evaluate(
        function=target_fn,
        data=dataset.id,
        evaluators=[accuracy_evaluator, routing_evaluator],
        experiment_name="multi_agent_eval_v1",
        max_concurrency=5
    )
    print("✅ Evaluation URL:", results.run_url)
