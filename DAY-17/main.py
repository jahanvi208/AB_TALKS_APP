import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# -----------------------------
# 1. Load model
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# 2. Knowledge base with metadata
# -----------------------------
documents = [
    {
        "text": "RAG stands for Retrieval Augmented Generation.",
        "source": "AI Handbook",
        "category": "AI",
        "date": "2024-01-10",
        "document_type": "technical"
    },
    {
        "text": "FAISS is used for efficient similarity search.",
        "source": "Meta Research",
        "category": "AI",
        "date": "2023-05-12",
        "document_type": "technical"
    },
    {
        "text": "Marketing strategies evolve with AI tools.",
        "source": "Marketing Blog",
        "category": "Marketing",
        "date": "2022-08-01",
        "document_type": "blog"
    },
    {
        "text": "Chunking improves retrieval quality in RAG systems.",
        "source": "AI Guide",
        "category": "AI",
        "date": "2024-02-15",
        "document_type": "technical"
    },
    {
        "text": "Old AI systems relied on keyword matching.",
        "source": "Legacy Docs",
        "category": "AI",
        "date": "2020-06-20",
        "document_type": "historical"
    }
]

# -----------------------------
# 3. Create embeddings
# -----------------------------
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)

# -----------------------------
# 4. Basic retrieval
# -----------------------------
def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    ranked = sorted(
        [(documents[i], scores[i]) for i in range(len(scores))],
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]

# -----------------------------
# 5. Metadata filtering
# -----------------------------
def apply_filters(docs, filters):
    filtered = []

    for doc, score in docs:
        match = True

        # Category filter
        if "category" in filters:
            if doc["category"] != filters["category"]:
                match = False

        # Date filter
        if "min_date" in filters:
            doc_date = datetime.strptime(doc["date"], "%Y-%m-%d")
            min_date = datetime.strptime(filters["min_date"], "%Y-%m-%d")
            if doc_date < min_date:
                match = False

        if match:
            filtered.append((doc, score))

    return filtered

# -----------------------------
# 6. Filtered retrieval
# -----------------------------
def filtered_retrieve(query, filters={}, top_k=3):
    initial_results = retrieve(query, top_k=5)  # get more first
    filtered = apply_filters(initial_results, filters)

    return filtered[:top_k]

# -----------------------------
# 7. Compare results
# -----------------------------
queries = [
    "What is RAG?",
    "Explain FAISS",
    "How does marketing use AI?",
    "What is chunking?",
    "Old AI systems?"
]

print("\n--- WITHOUT FILTERS ---")
for q in queries:
    results = retrieve(q)
    print(f"\nQuery: {q}")
    for r in results:
        print(f"- {r[0]['text']} ({r[0]['category']}, {r[0]['date']})")

print("\n--- WITH FILTERS (category = AI, min_date = 2023) ---")
filters = {
    "category": "AI",
    "min_date": "2023-01-01"
}

for q in queries:
    results = filtered_retrieve(q, filters)
    print(f"\nQuery: {q}")
    for r in results:
        print(f"- {r[0]['text']} ({r[0]['category']}, {r[0]['date']})")

# -----------------------------
# 8. Simple Precision Observation
# -----------------------------
print("\n--- OBSERVATION ---")
print("With metadata filters, irrelevant or outdated documents are removed.")
print("This improves retrieval precision but may reduce recall.")