# === csv_agent_tester.py ===
import streamlit as st
from router_agent import answer_query

st.set_page_config(page_title="CSV Agent Tester", layout="wide")
st.title("📊 CSV Agent Test Interface")

# --- Predefined Tests ---
test_queries = {
    "Factual Lookup": "What is the male population in Bumthang?",
    "Dataset Summary": "Summarize the dataset for Table 1.2 - Population by Sex and Dzongkhag",
    "Projection Total": "What is the total population of Thimphu?",
    "Maximum Population": "Which dzongkhag has the highest population?",
    "Unanswerable Query": "How many dragons live in Bumthang?",
    "Semantic Retrieval": "How many elderly people lived in Thimphu in 2017?",
    "DOCX Routing": "What is the difference between the crude birth rate and the total fertility rate?",
    "CSV Routing": "What was the total population of Thimphu in 2017?"
}

# --- Sidebar ---
st.sidebar.header("Run Test Suite")
run_all = st.sidebar.button("▶️ Run All Tests")
test_result = {}

# --- Individual Test Execution ---
for label, query in test_queries.items():
    col1, col2 = st.columns([1, 5])
    with col1:
        trigger = st.button("Run", key=label)
    with col2:
        st.markdown(f"**{label}**: _\"{query}\"_")

    if run_all or trigger:
        result = answer_query(query)
        answer = result.get("answer", "<no answer>")
        st.success(answer) if answer else st.error("No response returned.")
        if result.get("source_rows") is not None:
            st.write("🔍 Source Rows:")
            st.dataframe(result["source_rows"])

# --- Manual Test Box ---
st.markdown("---")
st.subheader("Manual Query Test")
custom_query = st.text_input("Enter a custom question:", "")
if st.button("Ask") and custom_query.strip():
    result = answer_query(custom_query)
    st.write("### 💬 Answer:", result.get("answer", "No response."))
    if result.get("source_rows") is not None:
        st.write("🔍 Source Rows:")
        st.dataframe(result["source_rows"])
