from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import Client

client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
    project_name=os.getenv("LANGSMITH_PROJECT")
)

# Create or read dataset
dataset_name = "CSV QA Eval"
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"📦 Dataset already exists: {dataset.name} (id={dataset.id})")
except Exception:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test questions for CSV data accuracy validation"
    )
    print(f"✅ Created new dataset: {dataset.name}")

# Test questions
examples = [
    {"inputs": {"question": "What was the mean age at first birth in Thimphu in 2017?"}, "outputs": {"answer": "Expected value from Table 1.7"}},
    {"inputs": {"question": "What is the projected male population in 2042?"}, "outputs": {"answer": "Expected value from Table 1.8"}},
    {"inputs": {"question": "What is the projected female population in Bumthang Dzongkhag in 2022?"}, "outputs": {"answer": "Expected value from Table 1.9"}},
    {"inputs": {"question": "How many BHUs were there in Trashigang in 2021?"}, "outputs": {"answer": "Expected value from Table 2.2"}},
    {"inputs": {"question": "What is the total number of doctors in 2021?"}, "outputs": {"answer": "Expected count from Table 2.3"}},
    {"inputs": {"question": "Which disease had the highest morbidity in 2020?"}, "outputs": {"answer": "Expected disease name from Table 2.4"}},
    {"inputs": {"question": "What was the number of hospitals in Bhutan in 2021?"}, "outputs": {"answer": "Expected value from Table 2.1"}},
    {"inputs": {"question": "What was the male population in Trashigang Thromde in 2017?"}, "outputs": {"answer": "Expected value from Table 1.2"}},
    {"inputs": {"question": "What was the total population of Bhutan in 2017?"}, "outputs": {"answer": "Expected value from Table 1.1"}}
]

# Upload to LangSmith
print("✅ Uploading examples...")
client.create_examples(dataset_id=dataset.id, examples=examples)

print("✅ Evaluation dataset is ready.")
print(f"🔗 View: https://smith.langchain.com/datasets/{dataset.id}")
