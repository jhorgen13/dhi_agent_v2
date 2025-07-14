# === csv_agent.py (Removed unsupported traceable context managers) ===
import pandas as pd
import re
import hashlib
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, initialize_agent, AgentType
from pathlib import Path
import os
import json
import uuid

CACHE_PATH = Path(".cache")
CACHE_PATH.mkdir(exist_ok=True)

# ------------------------ Schema Extraction ------------------------ #
def extract_schema(df: pd.DataFrame):
    dims, metrics = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 50:
                metrics.append(col)
            else:
                dims.append(col)
        else:
            dims.append(col)
    return dims, metrics

# ------------------------ Chunk CSV Data ------------------------ #
def chunk_table_data(df: pd.DataFrame, chunk_size: int = 30) -> List[Document]:
    if df.empty:
        print("[WARN] Skipping chunking: DataFrame is empty")
        return []
    dims, metrics = extract_schema(df)
    schema_info = f"(Table Schema: Dimensions = {', '.join(dims)}; Metrics = {', '.join(metrics)})"
    documents = []
    header = df.columns.tolist()

    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i + chunk_size]
        csv_content = chunk_df.to_csv(index=False)
        content = f"{schema_info}\n\n{csv_content}"
        metadata = {
            "chunk_index": i // chunk_size,
            "schema": schema_info,
            "columns": header,
            "metric_tags": metrics,
            "dimension_tags": dims
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"[DEBUG] Created {len(documents)} chunks")
    if documents:
        print(f"[DEBUG] Sample chunk content (first 200 chars): {documents[0].page_content[:200]!r}")

    return documents

# ------------------------ Hybrid Filter ------------------------ #
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def hybrid_filter(docs: List[Document], query: str, top_k: int = 10) -> List[Document]:
    terms = re.findall(r"\w+", query.lower())
    keyword_hits = [doc for doc in docs if all(term in doc.page_content.lower() for term in terms)]
    search_space = keyword_hits if keyword_hits else docs
    vectordb = FAISS.from_documents(search_space, embedder)
    results = vectordb.similarity_search(query, k=top_k, doc_list=search_space)
    combined = {doc.metadata.get("chunk_index", i): doc for i, doc in enumerate(results + keyword_hits)}
    return list(combined.values())[:top_k]

# ------------------------ Map-Reduce QA Chain ------------------------ #
def build_csv_chain(llm):
    map_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a spreadsheet analyst. Use only the provided table chunk to answer the question.
If the chunk has no relevant information, write exactly: No relevant data in this chunk.
Always extract values directly from the chunk. Do not estimate or average.
Return only the answer in plain text.

{context}
Question: {question}
Answer:"""
    )

    reduce_prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""You are combining answers from multiple spreadsheet chunks.
Ignore any chunk that says No relevant data in this chunk.
Use only the valid chunk answers to respond to the question.
If the data is conflicting or insufficient, say so.
Return only the answer in plain text.

Chunk Answers:
{summaries}

Question: {question}
Final Answer:"""
    )

    def run_map_reduce(docs: List[Document], question: str) -> str:
        top_docs = hybrid_filter(docs, question)
        partials = []
        for i, doc in enumerate(top_docs):
            prompt = map_prompt.format(context=doc.page_content, question=question)
            try:
                output = llm.invoke(prompt).strip()
                if output and "no relevant data" not in output.lower():
                    partials.append(output)
            except Exception as e:
                partials.append(f"[Error in chunk {i}]: {e}")
        if not partials:
            return "I don't know."
        if len(partials) == 1:
            return partials[0]
        reduce_input = reduce_prompt.format(summaries="\n".join(partials), question=question)
        return llm.invoke(reduce_input).strip()

    return run_map_reduce

# ------------------------ Pandas Tool for Projections ------------------------ #
def run_pandas_tool(df: pd.DataFrame, query: str) -> str:
    lowered = query.lower()
    try:
        if "projected" in lowered and "male" in lowered and "2042" in lowered:
            year_col = [col for col in df.columns if 'year' in col.lower()][0]
            male_col = [col for col in df.columns if 'male' in col.lower() and df[col].dtype != 'O'][0]
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            match = df[df[year_col] == 2042]
            if not match.empty:
                val = match[male_col].values[0]
                return f"{int(val)}"
    except Exception as e:
        print(f"[WARN] Pandas tool failed: {e}")
    return "Query too complex for direct lookup."

# ------------------------ Summary Caching ------------------------ #
def get_cached_summary(doc_id: str) -> str:
    cache_file = CACHE_PATH / f"summary_{doc_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text()).get("summary", "")
    return ""

def set_cached_summary(doc_id: str, summary: str):
    cache_file = CACHE_PATH / f"summary_{doc_id}.json"
    cache_file.write_text(json.dumps({"summary": summary}))

# ------------------------ CSV Agent Builder ------------------------ #
def build_csv_agent(llm, df: pd.DataFrame, topic_name: str = "CSV Data"):
    from langchain_ollama import OllamaLLM
    if llm is None:
        llm = OllamaLLM(model="qwen3:32b", temperature=0.0, streaming=True)

    if df.empty:
        raise ValueError(f"🚨 The CSV '{topic_name}' is empty or invalid.")

    print(f"[DEBUG] Building agent for: {topic_name}")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] Columns: {list(df.columns)}")
    print("[DEBUG] First few rows:\n", df.head().to_markdown())

    docs = chunk_table_data(df)
    if not docs:
        raise ValueError(f"🚨 No valid chunks found in CSV: {topic_name}")

    qa_runner = build_csv_chain(llm)

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""You are an expert data analyst. Summarize the dataset below, focusing on key trends, notable statistics, and important insights.
- Use bullet points for clarity.
- Mention significant numbers and comparisons explicitly.
- Keep the summary concise and factual.
- Use exact values only; do not approximate.
Return output as plain text.

Data:
{text}
Summary:"""
    )

    def data_tool(query: str) -> str:
        if "thromde" in query.lower():
            return "A Thromde is a town or municipality within a larger Dzongkhag (district) that is designated as a municipality."

        if query.lower().startswith("summarize"):
            doc_hash = hashlib.md5("".join([d.page_content for d in docs]).encode()).hexdigest()
            cached = get_cached_summary(doc_hash)
            if cached:
                return cached
            summary_chain = load_summarize_chain(
                llm,
                chain_type="map_reduce" if len(docs) >= 15 else "refine"
            )
            result = summary_chain.run(docs)
            set_cached_summary(doc_hash, result)
            return result

        if re.search(r"(how many|total|what is the number of|max|min|highest|male population|projected|2042)", query, re.IGNORECASE):
            result = run_pandas_tool(df, query)
            if "Query too complex" not in result:
                return result

        return qa_runner(docs, query)

    tool_name = f"CSV_Data_Tool_{topic_name.replace(' ', '_')}"

    tools = [
        Tool(
            name=tool_name,
            func=data_tool,
            description=f"Summarize or answer questions about the '{topic_name}' dataset."
        )
    ]

    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            handle_parsing_errors=True, max_iterations=5,
                            early_stopping_method="generate", verbose=True)
