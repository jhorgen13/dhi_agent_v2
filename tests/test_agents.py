# tests/test_agents.py

import sys
import os
import re
import pytest

# Ensure the root project directory is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.router_agent import router_chain, answer_query

# === Sample test cases (question, expected_snippet, expected_agent) ===
test_cases = [
    ("What is Bhutan’s projected total population in 2030?", "815755", "CSV"),
    ("What was the mean age at first birth in Thimphu in 2017?", "22.3", "CSV"),
    ("What is the projected male population in 2042?", "447180", "CSV"),
    ("Who is considered an unemployed person in Bhutan’s labour statistics?", "without work", "DOCX"),
    ("How often is the Labour Force Survey conducted in Bhutan?", "annually", "DOCX")
]

@pytest.mark.parametrize("question, expected_snippet, expected_agent", test_cases)
def test_agent_response_and_routing(question, expected_snippet, expected_agent):
    """Test that the router picks the correct agent and the answer contains the expected content."""
    # 1. Test routing classification
    route = router_chain.invoke({"query": question})
    assert "destination" in route, "Router did not return a destination"
    assert route["destination"].upper() == expected_agent, f"Router misclassified (expected {expected_agent})"

    # 2. Test the final answer content
    result = answer_query(question)
    answer = result["answer"].lower()
    assert expected_snippet.lower() in answer, f"Expected snippet '{expected_snippet}' not found in answer: {answer}"

    # 3. Sanity format checks
    if expected_agent == "CSV":
        assert re.search(r"\d", answer), "Expected numeric data in CSV response"
    else:
        assert len(answer.split()) > 3, "DOCX answer appears too short"
