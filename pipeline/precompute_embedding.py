from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# === Paths & Config ===
DOC_DIR = Path("data/Chapter Writeups")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

print("🔍 Scanning for DOCX files in:", DOC_DIR)

# === Load documents ===
all_docs = []
for path in DOC_DIR.glob("*.docx"):
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
    all_docs.extend(docs)

print(f"📄 Loaded {len(all_docs)} documents.")

# === Chunk documents ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(all_docs)

print(f"✂️ Chunked into {len(chunks)} segments.")

# === Generate embeddings ===
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

print("⚙️ Embedding and saving to vector store...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory=CHROMA_DIR
)

vectordb.persist()
print("✅ Embeddings saved to:", CHROMA_DIR)
