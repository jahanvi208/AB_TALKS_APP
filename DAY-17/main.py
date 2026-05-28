# Day 17 - Advanced Retrieval with Metadata

import faiss
import numpy as np
import pandas as pd

from datetime import datetime

from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------

print("Loading embedding model...\n")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------------------
# KNOWLEDGE BASE WITH METADATA
# -------------------------------------------------

documents = [

    {
        "text": "NovaTech uses QuantumCore processors in enterprise servers.",
        "source": "TechReport",
        "category": "technology",
        "date": "2024-01-10",
        "document_type": "technical"
    },

    {
        "text": "LunaCafe launched the Moonlight Caramel Latte campaign.",
        "source": "FoodMagazine",
        "category": "food",
        "date": "2023-08-12",
        "document_type": "marketing"
    },

    {
        "text": "RiverMind developed AI software for flood prediction.",
        "source": "AINews",
        "category": "technology",
        "date": "2024-05-01",
        "document_type": "technical"
    },

    {
        "text": "Vertex Motors released the Falcon X electric car.",
        "source": "AutoWorld",
        "category": "automobile",
        "date": "2022-11-15",
        "document_type": "product"
    },

    {
        "text": "PulseFit smartwatches can detect dehydration levels.",
        "source": "HealthTech",
        "category": "health",
        "date": "2024-03-20",
        "document_type": "technical"
    },

    {
        "text": "AstroPort became the first hotel built in orbit.",
        "source": "SpaceDaily",
        "category": "space",
        "date": "2021-06-18",
        "document_type": "news"
    },

    {
        "text": "ZenFoods created a snack called NutriBite.",
        "source": "FoodMagazine",
        "category": "food",
        "date": "2024-02-11",
        "document_type": "product"
    },

    {
        "text": "BlueWave Robotics builds underwater cleaning robots.",
        "source": "RoboticsWeekly",
        "category": "technology",
        "date": "2023-10-09",
        "document_type": "technical"
    },

    {
        "text": "Nimbus Airlines now allows pets in premium cabins.",
        "source": "TravelNews",
        "category": "travel",
        "date": "2024-04-14",
        "document_type": "policy"
    },

    {
        "text": "TerraFarm grows crops using vertical farming systems.",
        "source": "GreenEarth",
        "category": "agriculture",
        "date": "2022-07-25",
        "document_type": "technical"
    }
]

# -------------------------------------------------
# EXTRACT TEXTS
# -------------------------------------------------

texts = [doc["text"] for doc in documents]

# -------------------------------------------------
# GENERATE EMBEDDINGS
# -------------------------------------------------

print("Generating embeddings...\n")

embeddings = embedding_model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

# -------------------------------------------------
# BUILD FAISS INDEX
# -------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS Index Size: {index.ntotal}\n")

# -------------------------------------------------
# NORMAL RETRIEVAL
# -------------------------------------------------

def retrieve(query, top_k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in range(top_k):

        idx = indices[0][i]

        results.append({

            "text": documents[idx]["text"],
            "score": float(distances[0][i]),
            "metadata": {
                "source": documents[idx]["source"],
                "category": documents[idx]["category"],
                "date": documents[idx]["date"],
                "document_type": documents[idx]["document_type"]
            }
        })

    return results

# -------------------------------------------------
# FILTERED RETRIEVAL
# -------------------------------------------------

def filtered_retrieve(query, filters=None, top_k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, len(documents))

    filtered_results = []

    for i in range(len(indices[0])):

        idx = indices[0][i]

        doc = documents[idx]

        match = True

        # CATEGORY FILTER
        if filters and "category" in filters:

            if doc["category"] != filters["category"]:
                match = False

        # DATE FILTER
        if filters and "date_after" in filters:

            doc_date = datetime.strptime(doc["date"], "%Y-%m-%d")

            cutoff = datetime.strptime(filters["date_after"], "%Y-%m-%d")

            if doc_date < cutoff:
                match = False

        if match:

            filtered_results.append({

                "text": doc["text"],
                "score": float(distances[0][i]),
                "metadata": {
                    "source": doc["source"],
                    "category": doc["category"],
                    "date": doc["date"],
                    "document_type": doc["document_type"]
                }
            })

        if len(filtered_results) >= top_k:
            break

    return filtered_results

# -------------------------------------------------
# TEST QUERIES
# -------------------------------------------------

queries = [

    {
        "query": "AI software systems",
        "filters": {
            "category": "technology"
        }
    },

    {
        "query": "food products",
        "filters": {
            "category": "food"
        }
    },

    {
        "query": "health wearable devices",
        "filters": {
            "category": "health"
        }
    },

    {
        "query": "space hotels",
        "filters": {
            "date_after": "2022-01-01"
        }
    },

    {
        "query": "robotics technology",
        "filters": {
            "category": "technology",
            "date_after": "2023-01-01"
        }
    }
]

# -------------------------------------------------
# COMPARISON
# -------------------------------------------------

print("\n========== RETRIEVAL COMPARISON ==========\n")

for item in queries:

    query = item["query"]

    filters = item["filters"]

    print(f"\nQUERY: {query}\n")

    # WITHOUT FILTERS
    print("WITHOUT FILTERS:\n")

    normal_results = retrieve(query)

    for result in normal_results:

        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}\n")

    # WITH FILTERS
    print("WITH FILTERS:\n")

    filtered_results = filtered_retrieve(query, filters)

    for result in filtered_results:

        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}\n")

    print("=" * 80)

# -------------------------------------------------
# ARCHITECTURE UPDATE
# -------------------------------------------------

print("\n========== UPDATED RAG ARCHITECTURE ==========\n")

architecture = """

USER QUERY
     |
     v
QUERY EMBEDDING
     |
     v
METADATA FILTERING
(Category / Date Constraints)
     |
     v
FAISS VECTOR SEARCH
     |
     v
FILTERED RELEVANT DOCUMENTS
     |
     v
PROMPT CONSTRUCTION
     |
     v
LLM GENERATION
     |
     v
FINAL ANSWER

"""

print(architecture)

# -------------------------------------------------
# EDGE CASES
# -------------------------------------------------

print("\n========== EDGE CASES ==========\n")

edge_cases = """

1. Missing Metadata
If documents do not contain category or date fields,
metadata filtering may exclude useful results incorrectly.

-----------------------------------------------------

2. Conflicting Categories
Some documents may belong to multiple categories,
but strict filtering forces only one category match.
"""

print(edge_cases)

# -------------------------------------------------
# PRECISION ANALYSIS
# -------------------------------------------------

print("\n========== PRECISION ANALYSIS ==========\n")

analysis = """

Metadata filtering improved retrieval precision by removing
semantically similar but contextually irrelevant documents.

Category filtering helped restrict retrieval to the correct domain.

Date filtering prevented outdated documents from being retrieved,
improving relevance for time-sensitive queries.
"""

print(analysis)