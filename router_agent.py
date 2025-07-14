# === router_agent.py ===
from pathlib import Path
import re
import json
from typing import Dict
from difflib import get_close_matches
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_ollama import OllamaLLM as Ollama
from agents.docx_agent import build_docx_agent, get_retriever
from agents.csv_agent import build_csv_agent
from agents.web_agent import build_web_agent
from utils.source_tracer import trace_sources

# === LLM Setup ===
llm = Ollama(model="qwen3:30b-a3b", temperature=0.1)

# === Embeddings ===
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load CSV Files ===
csv_files = list(Path("data/Numeric Tables").rglob("*.csv"))
label_to_path = {f.stem.strip(): f for f in csv_files}
semantic_docs = []

def load_and_validate_csv(file_path: Path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if df.empty or df.isnull().all().all() or len(df.dropna(how='all')) == 0:
                return None
            return df
        except Exception:
            continue
    return None

for name, path in label_to_path.items():
    try:
        df = load_and_validate_csv(path)
        if df is None:
            continue
        schema = f"Table: {name}. Columns: {', '.join(df.columns)}"
        preview = df.head(2).to_csv(index=False)
        doc = Document(page_content=f"{schema}\n\n{preview}", metadata={"label": name, "file_path": str(path)})
        semantic_docs.append(doc)
    except Exception:
        continue

if not semantic_docs:
    raise ValueError("🚨 No valid CSV files for embedding.")

vectorstore = FAISS.from_documents(semantic_docs, embed_model)

# === Internal routing: DOCX vs CSV ===
def classify_internal_doc_source(query: str) -> str:
    try:
        route = router_chain.invoke({"query": query})
        return route.get("destination", "DOCX").upper()
    except:
        return "DOCX"

classification_prompt = PromptTemplate(
    template='''You are a routing agent for internal documents.

Classify the query:
- "DOCX" → for descriptive, chapter-based, narrative answers
- "CSV" → for numeric or data-table questions

Respond ONLY with:
{{"destination": "CSV"}} or {{"destination": "DOCX"}}

Q: {query}
A:''',
    input_variables=["query"]
)

router_chain = classification_prompt | llm | RunnableLambda(lambda x: json.loads(re.search(r"\{.*?\}", x.strip()).group()))

# === Answer Handlers ===
def answer_docx(input: Dict) -> Dict:
    query = input.get("query")
    topic = input.get("topic", "")
    docx_dir = Path("data/Chapter Writeups")
    all_chapters = [f.stem for f in docx_dir.glob("*.docx")]
    matched = get_close_matches(topic, all_chapters, n=1, cutoff=0.75)
    chapters = matched if matched else ["Chapter 1 - Population"]
    agent = build_docx_agent(llm, chapters)
    try:
        return {"answer": agent.run(query).strip(), "source_rows": None}
    except:
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=get_retriever(), chain_type="stuff")
        return {"answer": qa.run(query).strip(), "source_rows": None}

def answer_csv(input: Dict) -> Dict:
    query = input.get("query")
    matches = vectorstore.similarity_search(query, k=1)
    if not matches:
        return {"answer": "No matching CSV table found.", "source_rows": None}
    best_label = matches[0].metadata["label"].strip()
    df_path = label_to_path.get(best_label)
    df = load_and_validate_csv(df_path)
    if df is None:
        return {"answer": f"Failed to load CSV: {best_label}", "source_rows": None}
    agent = build_csv_agent(llm, df, topic_name=best_label)
    try:
        answer = agent.run(query)
        sources_df = trace_sources(df, query)
        return {
            "answer": str(answer).strip(),
            "source_rows": sources_df if not sources_df.empty else None
        }
    except Exception as e:
        return {"answer": f"CSV agent error: {str(e)}", "source_rows": None}

def answer_web(input: Dict) -> Dict:
    query = input.get("query", "")
    agent = build_web_agent(llm)
    try:
        return {"answer": str(agent.run(query)).strip(), "source_rows": None}
    except Exception as e:
        return {"answer": f"Web search error: {e}", "source_rows": None}

# === Unified Public Function ===
def answer_query(query: str, source_type: str = None) -> Dict:
    if not query.strip():
        return {"answer": "Please enter a question.", "source_rows": None}

    if source_type == "Web Search":
        return answer_web({"query": query})
    elif source_type == "In Company Documents":
        internal_type = classify_internal_doc_source(query)
        if internal_type == "CSV":
            return answer_csv({"query": query})
        return answer_docx({"query": query})

    return {"answer": "❌ Invalid source type selected.", "source_rows": None}
