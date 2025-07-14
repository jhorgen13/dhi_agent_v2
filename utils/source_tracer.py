# === source_tracer.py (Refined for Keyword Matching & Grounding) ===
import pandas as pd
import re

STOPWORDS = {"the", "a", "an", "of", "to", "in", "what", "is", "was", "are", "and"}

# Clean and extract meaningful keywords
def extract_keywords(query: str):
    words = re.findall(r'\w+', query.lower())
    return [w for w in words if w not in STOPWORDS]

def trace_sources(df: pd.DataFrame, query: str) -> pd.DataFrame:
    keywords = extract_keywords(query)
    matched_rows = []

    for _, row in df.iterrows():
        row_text = row.astype(str).str.lower().str.cat(sep=" ")
        if all(kw in row_text for kw in keywords):
            matched_rows.append(row)

    return pd.DataFrame(matched_rows)
