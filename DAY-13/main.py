# Day 13 - Dense Embeddings (FREE OFFLINE VERSION)

import time
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------

print("Loading embedding model...\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------------------
# 20 SENTENCES (4 TOPICS)
# -------------------------------------------------

sentences = [

    # Sports
    "The football team won the championship after a thrilling match.",
    "Basketball players require excellent coordination and stamina.",
    "The athlete trained every day to prepare for the Olympics.",
    "Cricket is one of the most popular sports in India.",
    "Tennis matches can last several hours during tournaments.",

    # Technology
    "Artificial intelligence is transforming modern industries.",
    "Cloud computing allows data storage over the internet.",
    "Cybersecurity protects systems from digital attacks.",
    "Smartphones have changed the way people communicate.",
    "Machine learning models improve through experience.",

    # Cooking
    "The chef prepared a delicious pasta recipe for dinner.",
    "Baking cakes requires accurate measurements of ingredients.",
    "Spices add flavor and aroma to traditional dishes.",
    "Grilling vegetables enhances their smoky taste.",
    "Homemade soup is healthy and easy to prepare.",

    # Travel
    "Tourists visited the ancient temples during their vacation.",
    "Airplanes make international travel faster and easier.",
    "The beach resort offered stunning ocean views.",
    "Backpacking through Europe can be an exciting adventure.",
    "Travel guides help visitors explore unfamiliar cities."
]

# -------------------------------------------------
# GENERATE EMBEDDINGS
# -------------------------------------------------

print("Generating embeddings...\n")

start_time = time.time()

embeddings = model.encode(sentences)

end_time = time.time()

generation_time = end_time - start_time

print(f"Embedding generation completed in {generation_time:.2f} seconds.")

# -------------------------------------------------
# ESTIMATED API COST
# -------------------------------------------------

print("Estimated API cost: FREE (offline embeddings)\n")

# -------------------------------------------------
# TF-IDF VECTORIZATION
# -------------------------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(sentences)

# -------------------------------------------------
# TF-IDF vs EMBEDDING SIMILARITY
# -------------------------------------------------

print("\n========== TF-IDF vs EMBEDDING SIMILARITY ==========\n")

pairs = [
    (0, 2),
    (5, 9),
    (10, 14),
    (15, 19),
    (3, 16)
]

for i, j in pairs:

    tfidf_score = cosine_similarity(
        tfidf_matrix[i],
        tfidf_matrix[j]
    )[0][0]

    embedding_score = cosine_similarity(
        [embeddings[i]],
        [embeddings[j]]
    )[0][0]

    print(f"Sentence A: {sentences[i]}")
    print(f"Sentence B: {sentences[j]}")

    print(f"TF-IDF Similarity: {tfidf_score:.4f}")
    print(f"Embedding Similarity: {embedding_score:.4f}")

    print("-" * 60)

# -------------------------------------------------
# PARAPHRASE EXAMPLES
# -------------------------------------------------

print("\n========== PARAPHRASE EXAMPLES ==========\n")

paraphrase_pairs = [

    ("The athlete trained every day for the Olympics.",
     "Sports players practice daily before competitions."),

    ("Artificial intelligence is transforming industries.",
     "AI is changing the way companies operate."),

    ("Homemade soup is healthy and easy to prepare.",
     "Making soup at home is simple and nutritious."),

    ("Travel guides help visitors explore unfamiliar cities.",
     "Tour books assist tourists in discovering new places."),

    ("Cloud computing allows internet-based storage.",
     "Online servers are used to store digital files.")
]

all_text = []

for p in paraphrase_pairs:
    all_text.extend(p)

temp_tfidf = TfidfVectorizer(stop_words='english')
temp_matrix = temp_tfidf.fit_transform(all_text)

for idx, pair in enumerate(paraphrase_pairs):

    s1 = pair[0]
    s2 = pair[1]

    tfidf_score = cosine_similarity(
        temp_matrix[idx * 2],
        temp_matrix[idx * 2 + 1]
    )[0][0]

    emb1 = model.encode([s1])[0]
    emb2 = model.encode([s2])[0]

    embedding_score = cosine_similarity(
        [emb1],
        [emb2]
    )[0][0]

    print(f"Sentence 1: {s1}")
    print(f"Sentence 2: {s2}")

    print(f"TF-IDF Similarity: {tfidf_score:.4f}")
    print(f"Embedding Similarity: {embedding_score:.4f}")

    print("-" * 60)

# -------------------------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------------------------

def embed_and_recommend(query_sentence, corpus_sentences, top_k=3):

    query_embedding = model.encode([query_sentence])[0]

    scores = []

    for i, emb in enumerate(embeddings):

        score = cosine_similarity(
            [query_embedding],
            [emb]
        )[0][0]

        scores.append((corpus_sentences[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]

# -------------------------------------------------
# TEST RECOMMENDATION FUNCTION
# -------------------------------------------------

print("\n========== RECOMMENDATION RESULTS ==========\n")

query = "AI systems are improving business operations."

results = embed_and_recommend(query, sentences)

print(f"Query: {query}\n")

for sentence, score in results:

    print(f"Similarity Score: {score:.4f}")
    print(f"Sentence: {sentence}")
    print()

# -------------------------------------------------
# K-MEANS CLUSTERING
# -------------------------------------------------

print("\n========== K-MEANS CLUSTERING ==========\n")

kmeans = KMeans(n_clusters=4, random_state=42)

clusters = kmeans.fit_predict(embeddings)

for i, sentence in enumerate(sentences):

    print(f"Cluster {clusters[i]} --> {sentence}")

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------


# -------------------------------------------------
# EXPLANATION
# -------------------------------------------------

print("\n========== TF-IDF vs EMBEDDINGS ==========\n")

explanation = """

TF-IDF vectors are sparse vectors based on exact word frequency.
They only understand direct vocabulary overlap.

Embeddings are dense vectors that capture semantic meaning.
They understand similarity even when different words are used.

This is why embeddings perform much better for semantic search,
recommendation systems, and modern AI retrieval tasks.
"""

print(explanation)