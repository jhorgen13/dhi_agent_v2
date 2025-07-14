# tests/build_dataset.py

from langsmith import Client

client = Client()

dataset = client.create_dataset(
    dataset_name="MultiAgent QA Test Set",
    description="Q&A examples to test CSV vs DOCX agents."
)

examples = [
    {
        "inputs": {"question": "What is Bhutan’s projected total population in 2030?"},
        "outputs": {"answer": "Bhutan’s projected population in 2030 is 815755."},
        "metadata": {"expected_agent": "CSV"}
    },
    {
        "inputs": {"question": "What was the mean age at first birth in Thimphu in 2017?"},
        "outputs": {"answer": "In 2017, the mean age at first birth in Thimphu was 22.3 years."},
        "metadata": {"expected_agent": "CSV"}
    },
    {
        "inputs": {"question": "What is the projected male population in 2042?"},
        "outputs": {"answer": "The projected male population in 2042 is 447180."},
        "metadata": {"expected_agent": "CSV"}
    },
    {
        "inputs": {"question": "Who is considered an unemployed person in Bhutan’s labour statistics?"},
        "outputs": {"answer": "An unemployed person is a person without work, looking for a job and available for work during the reference period."},
        "metadata": {"expected_agent": "DOCX"}
    },
    {
        "inputs": {"question": "How often is the Labour Force Survey conducted in Bhutan?"},
        "outputs": {"answer": "It is conducted annually."},
        "metadata": {"expected_agent": "DOCX"}
    }
]

client.create_examples(dataset_id=dataset.id, examples=examples)
print(f"✅ Dataset '{dataset.name}' created with {len(examples)} examples.")
