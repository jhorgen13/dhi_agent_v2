# tests/test_agents.py

import re
import pytest
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
    """Test router routing and final answer content."""
    route = router_chain.invoke({"query": question})
    assert "destination" in route
    assert route["destination"].upper() == expected_agent, f"Expected {expected_agent}, got {route['destination']}"

    result = answer_query(question)
    answer = result["answer"].lower()
    assert expected_snippet.lower() in answer, f"Expected snippet '{expected_snippet}' not in answer: {answer}"

    # Sanity format checks
    if expected_agent == "CSV":
        assert re.search(r"\d", answer), "Expected a numeric value in CSV answer."
    else:
        assert len(answer.split()) > 3, "Expected descriptive answer for DOCX."
