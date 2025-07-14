# === docx_agent.py (Refined for Scoped Summarization & Query Accuracy) ===
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.agents import Tool, initialize_agent, AgentType

# ---- Global Configuration ----
doc_dir = Path("data/Chapter Writeups")
persist_dir = "chroma_db"
embed_model = "sentence-transformers/all-mpnet-base-v2"

# ---- Load and Chunk All Documents ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
all_chunks = {}

for doc_path in doc_dir.glob("*.docx"):
    loader = Docx2txtLoader(str(doc_path))
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    all_chunks[doc_path.stem] = chunks

# ---- Load Persisted Vectorstore ----
embedder = HuggingFaceEmbeddings(model_name=embed_model)
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedder)
global_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- Agent Builder ----
def build_docx_agent(llm, selected_chapters):
    selected_chunks = []
    matched_chapters = []
    for chapter in selected_chapters:
        norm = chapter.strip().replace("chapter", "Chapter").title()
        match = next((k for k in all_chunks if k.lower().startswith(norm.lower())), None)
        if match:
            selected_chunks.extend(all_chunks[match])
            matched_chapters.append(match)

    print("🔎 Matched Chapters:", matched_chapters)
    print("📚 Selected Chunks:", len(selected_chunks))

    retriever = Chroma.from_documents(selected_chunks, embedder).as_retriever(search_kwargs={"k": 3}) if selected_chunks else global_retriever

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an assistant answering questions about a document.
Use the following context as the **only** source of information.
If the answer is not explicitly in the text, say \"I don't know based on the document.\"

Context:
{context}

Question: {question}
Answer (in 1–2 concise sentences):"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""You are an expert report analyst. Summarize the document below, focusing on the main programs, key outcomes, and statistics.
- Use bullet points.
- Preserve important figures and program names.
- Keep the summary concise and faithful.

Document:
{text}
Summary:"""
    )

    def doc_search_tool(query: str) -> str:
        stripped = query.lower().strip()
        if stripped in ["summarize", "summarize the document", "summarize all"]:
            all_text = " ".join([doc.page_content for doc in selected_chunks])
            summary_chain = load_summarize_chain(llm, chain_type="map_reduce", prompt=summary_prompt)
            return summary_chain.run([Document(page_content=all_text)])

        if stripped.startswith("summarize"):
            chapter_key = query.replace("summarize", "").strip().replace("chapter", "Chapter").title()
            matched = next((k for k in all_chunks if k.lower().startswith(chapter_key.lower())), None)
            if not matched:
                return f"Chapter '{chapter_key}' not found. Available: {list(all_chunks.keys())}"
            chapter_chunks = all_chunks[matched]
            chain_type = "refine" if len(chapter_chunks) < 15 else "map_reduce"
            summary_chain = load_summarize_chain(
                llm, chain_type=chain_type, chain_type_kwargs={"prompt": summary_prompt}
            )
            return summary_chain.run(chapter_chunks)

        try:
            return qa_chain.run(query)
        except Exception:
            fallback_chain = RetrievalQA.from_chain_type(llm=llm, retriever=global_retriever, chain_type="stuff")
            return fallback_chain.run(query)

    tools = [
        Tool(
            name="SearchDocs",
            func=doc_search_tool,
            description="Summarize or answer questions about the selected chapters."
        )
    ]

    return initialize_agent(
        tools, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate"
    )

# ---- Access Retriever for Fallback ----
get_retriever = lambda: global_retriever
