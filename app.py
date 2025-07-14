# === app.py ===
import os
import streamlit as st
from dotenv import load_dotenv
from langsmith import Client
from router_agent import answer_query
from langchain_core.tracers.context import collect_runs, tracing_v2_enabled

load_dotenv()
project_name = os.getenv("LANGSMITH_PROJECT", "dhi-agent-v2")
client = Client()

st.set_page_config(page_title="🇧🇹 DHI AI Agent", layout="wide")
st.title("🇧🇹 DHI AI Agent")

if "status" not in st.session_state:
    st.session_state.status = ""

st.subheader("🔎 Ask a Question")
query = st.text_input("Bhutan AI")
source_type = st.selectbox("Where should I search?", ["In Company Documents", "Web Search"])

if st.button("Submit") and query:
    with st.spinner("🔍 Finding the answer..."):
        with collect_runs() as run_collector:
            with tracing_v2_enabled(project_name=project_name):
                result = answer_query(query, source_type=source_type)
            run = run_collector.traced_runs[0]

    st.session_state.status = "✅ Answer complete"
    st.markdown(f"**Answer:** {result['answer']}")

    if result.get("source_rows") is not None:
        with st.expander("📄 Show source data rows"):
            st.dataframe(result["source_rows"])

    st.sidebar.markdown(f"[📈 View Trace in LangSmith]({client.read_run(run.id).url})")
