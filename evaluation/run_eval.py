from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import Client
from evaluation.evaluation import dataset
from router_agent import answer_query
import re

client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
    project_name=os.getenv("LANGSMITH_PROJECT")
)

# Evaluators
def correctness_check(inputs, outputs, reference_outputs):
    predicted = outputs.get("answer", "").strip().lower()
    expected = reference_outputs.get("answer", "").strip().lower()
    return {"correct": 1 if expected in predicted or predicted in expected else 0}

def numeric_presence(inputs, outputs, reference_outputs):
    answer = outputs.get("answer", "")
    return {"has_number": 1 if any(char.isdigit() for char in answer) else 0}

def out_of_range_check(inputs, outputs, reference_outputs):
    question = inputs["question"]
    answer = outputs.get("answer", "")
    years = re.findall(r"\\b(19\\d{2}|20\\d{2}|21\\d{2})\\b", question)
    if years:
        asked_year = max(map(int, years))
        if asked_year > 2047 and any(char.isdigit() for char in answer):
            return {"out_of_range": 0}
    return {"out_of_range": 1}

def run_query(inputs):
    return answer_query(inputs["question"])

def run_batch_eval():
    print("🔍 Running batch evaluation over dataset:", dataset.name)
    results = client.evaluate(
        target=run_query,
        data=dataset.id,
        evaluators=[correctness_check, numeric_presence, out_of_range_check],
        experiment_name="csv_agent_batch_eval"
    )
    print("✅ Evaluation complete.")
    print("🔗 View results:", results.url)

if __name__ == "__main__":
    run_batch_eval()
