# tests/test_agents.py

import sys
import os
import re
import pytest

# Ensure the root project directory is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from router_agent import router_chain, answer_query

# === Sample test cases (question, expected_snippet, expected_agent) ===
test_cases = [
    # From Table 1.8 - Projected Total Population by Sex
    ("What is the projected male population in 2042?", "447180", "CSV"),
    
    # From Table 1.7 - Mean Age at First Birth by Dzongkhag
    ("What was the mean age at first birth in Thimphu in 2017?", "22.29", "CSV"),

    # From Chapter 4 - Labour and Employment
    ("Who is considered an unemployed person in Bhutan’s labour statistics?", "without work", "DOCX"),
    ("How often is the Labour Force Survey conducted in Bhutan?", "conducted annually", "DOCX"),

    # From Table 2.3 - Health Personnel
    ("How many General Nurses were there in 2021?", "general nurse", "CSV"),

    # From Table 2.4 - Top Ten Morbidity Diseases
    ("What was the top morbidity disease in 2021?", "top ten morbidity", "CSV"),

    # From Table 2.19 - Traditional Medicine Services
    ("How many males aged 25-29 availed traditional therapeutic services in 2021?", "25-29", "CSV"),

    # From Education file - Number of Students by Dzongkhag
    ("How many students were enrolled in higher secondary schools in Thimphu in 2022?", "higher secondary", "CSV"),

    # From Chapter 5 - Landuse and Agriculture
    ("What farming trend has increased due to better road access in Bhutan?", "commercial agriculture", "DOCX"),

    # From Chapter 5 - Landuse and Agriculture
    ("What are some strategies to increase crop production in Bhutan?", "farm mechanization", "DOCX"),
]


@pytest.mark.parametrize("question, expected_snippet, expected_agent", test_cases)
def test_agent_response_and_routing(question, expected_snippet, expected_agent):
    """Test that the router picks the correct agent and the answer contains the expected content."""
    # 1. Test routing classification
    route = router_chain.invoke({"query": question})
    assert "destination" in route, "Router did not return a destination"
    assert route["destination"].upper() == expected_agent, f"Router misclassified (expected {expected_agent})"

    # 2. Test the final answer content
    result = answer_query(question, source_type="In Company Documents")
    answer = result["answer"].lower()
    assert expected_snippet.lower() in answer, f"Expected snippet '{expected_snippet}' not found in answer: {answer}"

    # 3. Sanity format checks
    if expected_agent == "CSV":
        assert re.search(r"\d", answer), "Expected numeric data in CSV response"
    else:
        assert len(answer.split()) > 3, "DOCX answer appears too short"
