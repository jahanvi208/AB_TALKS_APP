# Day 14 - Semantic Search with FAISS

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------

print("Loading embedding model...\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------------------
# 50 DOCUMENTS
# -------------------------------------------------

documents = [

    # Technology
    "Artificial intelligence is transforming industries.",
    "Machine learning improves systems through data.",
    "Cloud computing enables online data storage.",
    "Cybersecurity protects digital systems.",
    "Neural networks are inspired by the human brain.",
    "Python is popular for AI development.",
    "Data science combines statistics and computing.",
    "Smartphones changed global communication.",
    "Blockchain enables secure digital transactions.",
    "Quantum computing solves complex problems.",

    # Sports
    "Football is the world's most popular sport.",
    "Cricket is widely played in India.",
    "Olympic athletes train for years.",
    "Basketball requires teamwork and coordination.",
    "Tennis matches can last many hours.",
    "Swimming improves physical fitness.",
    "The team celebrated their championship victory.",
    "Running daily improves stamina.",
    "Sports competitions attract huge audiences.",
    "Coaches help players improve performance.",

    # Cooking
    "Baking requires precise ingredient measurements.",
    "Pasta is a popular Italian dish.",
    "Soup is healthy and easy to prepare.",
    "Spices improve food flavor.",
    "Grilling vegetables creates smoky taste.",
    "Chefs prepare meals professionally.",
    "Fresh ingredients improve cooking quality.",
    "Chocolate cakes are popular desserts.",
    "Recipes guide cooking preparation.",
    "Kitchen hygiene is important for safety.",

    # Travel
    "Tourists visit historical monuments.",
    "Airplanes make travel faster.",
    "Beach resorts attract vacation travelers.",
    "Backpacking across Europe is adventurous.",
    "Travel guides help visitors explore cities.",
    "Hotels provide temporary accommodation.",
    "Mountains are popular tourist destinations.",
    "Maps help travelers navigate routes.",
    "Vacation planning saves time and money.",
    "Airports connect international destinations.",

    # Health
    "Exercise improves cardiovascular health.",
    "Doctors diagnose medical conditions.",
    "Healthy diets improve immunity.",
    "Yoga reduces stress and anxiety.",
    "Hospitals provide emergency treatment.",
    "Meditation improves mental focus.",
    "Vaccines prevent dangerous diseases.",
    "Drinking water supports body functions.",
    "Sleep is essential for recovery.",
    "Walking daily improves overall wellness."
]

# -------------------------------------------------
# GENERATE EMBEDDINGS
# -------------------------------------------------

print("Generating embeddings...\n")

embeddings = model.encode(documents)

# Convert to float32 for FAISS

embeddings = np.array(embeddings).astype("float32")

# -------------------------------------------------
# BUILD FAISS INDEX
# -------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS Index Size: {index.ntotal} documents\n")

# -------------------------------------------------
# SEMANTIC SEARCH FUNCTION
# -------------------------------------------------

def semantic_search(query, top_k=3):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in range(top_k):

        results.append({
            "document": documents[indices[0][i]],
            "distance": float(distances[0][i])
        })

    return results

# -------------------------------------------------
# KEYWORD SEARCH FUNCTION
# -------------------------------------------------

def keyword_search(query, corpus, top_k=3):

    query_words = set(query.lower().split())

    scores = []

    for doc in corpus:

        doc_words = set(doc.lower().split())

        overlap = len(query_words.intersection(doc_words))

        scores.append((doc, overlap))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]

# -------------------------------------------------
# TEST QUERIES
# -------------------------------------------------

queries = [

    "AI changing businesses",
    "healthy food and nutrition",
    "sports training",
    "vacation near ocean",
    "internet safety",
    "mental relaxation techniques",
    "how to prepare meals",
    "global transportation",
    "exercise for body fitness",
    "computer programs learning from data"
]

# -------------------------------------------------
# RUN TESTS
# -------------------------------------------------

print("========== SEMANTIC SEARCH vs KEYWORD SEARCH ==========\n")

for query in queries:

    print(f"\nQUERY: {query}\n")

    # Semantic Search
    print("Semantic Search Results:\n")

    semantic_results = semantic_search(query)

    for result in semantic_results:

        print(f"Distance: {result['distance']:.4f}")
        print(f"Document: {result['document']}")
        print()

    # Keyword Search
    print("Keyword Search Results:\n")

    keyword_results = keyword_search(query, documents)

    for doc, score in keyword_results:

        print(f"Overlap Score: {score}")
        print(f"Document: {doc}")
        print()

    print("-" * 70)

# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

print("\n========== ANALYSIS ==========\n")

semantic_better = """

1. Query: 'AI changing businesses'
Semantic search correctly matches artificial intelligence topics
even though the exact words differ.

2. Query: 'vacation near ocean'
Semantic search understands beach resort meaning
without requiring exact vocabulary overlap.

3. Query: 'mental relaxation techniques'
Semantic search connects meditation and yoga concepts
despite vocabulary mismatch.
"""

keyword_better = """

1. Query: 'Python'
Keyword search precisely finds Python-related documents immediately.

2. Query: 'vaccines'
Keyword search performs better for exact technical terms.

3. Query: 'blockchain'
Keyword search is highly precise when exact terminology matters.
"""

tradeoffs = """

Semantic Search:
- Understands meaning and intent
- Handles synonyms and paraphrases
- Better for natural language queries
- More computationally expensive
- Requires embedding generation and vector databases

Keyword Search:
- Fast and simple
- Very precise for exact terms
- Lower infrastructure cost
- Fails on synonym and vocabulary mismatch problems

Production Trade-offs:
- Semantic systems need GPUs or optimized vector databases
- Embedding APIs introduce latency and cost
- Keyword search scales cheaply and quickly
- Modern search engines often combine both approaches
"""

print("Semantic Search Better Cases:")
print(semantic_better)

print("Keyword Search Better Cases:")
print(keyword_better)

print("Trade-offs:")
print(tradeoffs)