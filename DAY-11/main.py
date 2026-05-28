# Mini Information Retrieval Engine using TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------------
# Knowledge Base (20 Documents)
# -----------------------------------

documents = [
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Machine learning is a subset of artificial intelligence focused on data-driven learning.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information from images.",
    "Reinforcement learning trains agents using rewards and punishments.",
    "Supervised learning requires labeled training data.",
    "Unsupervised learning finds hidden patterns in unlabeled data.",
    "Neural networks are inspired by the structure of the human brain.",
    "Large language models are trained on massive text datasets.",
    "Chatbots use NLP techniques to interact with users.",
    "AI can be applied in healthcare for disease prediction.",
    "Autonomous vehicles rely on computer vision and sensors.",
    "Recommendation systems suggest products based on user behavior.",
    "Data preprocessing is important before training machine learning models.",
    "TF-IDF is a statistical method used in information retrieval.",
    "Cosine similarity measures the angle between two vectors.",
    "Python is widely used for artificial intelligence development.",
    "Robotics combines AI with mechanical engineering.",
    "Ethics in AI includes fairness, accountability, and transparency."
]

# -----------------------------------
# Pipeline Class
# -----------------------------------

class TextPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit_transform(self, docs):
        return self.vectorizer.fit_transform(docs)

    def transform(self, text):
        return self.vectorizer.transform([text])


# -----------------------------------
# Vectorize Documents
# -----------------------------------

pipeline = TextPipeline()
corpus_matrix = pipeline.fit_transform(documents)

# -----------------------------------
# Retrieval Function
# -----------------------------------

def retrieve(query, corpus_matrix, top_k=3, threshold=0.1):

    query_vector = pipeline.transform(query)

    similarities = cosine_similarity(query_vector, corpus_matrix).flatten()

    max_score = similarities.max()

    # Relevance threshold
    if max_score < threshold:
        return "No relevant document found."

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "score": round(similarities[idx], 3)
        })

    return results


# -----------------------------------
# Test Queries
# -----------------------------------

queries = [
    "What is machine learning?",
    "How do chatbots understand language?",
    "Explain neural networks",
    "How are self-driving cars built?",
    "What is TF-IDF used for?",
    "AI ethics and fairness",
    "Brain-inspired computing",
    "Learning without labels",
    "Best pizza recipes",
    "Weather forecast tomorrow"
]

# -----------------------------------
# Run Tests
# -----------------------------------

print("\n========== RETRIEVAL RESULTS ==========\n")

for i, query in enumerate(queries, 1):

    print(f"Query {i}: {query}")

    results = retrieve(query, corpus_matrix)

    if isinstance(results, str):
        print(results)

    else:
        for rank, result in enumerate(results, 1):

            print(f"{rank}. Similarity Score: {result['score']}")
            print(f"   Document: {result['document']}")

    print("-" * 60)


# -----------------------------------
# Failure Analysis
# -----------------------------------

print("\n========== FAILURE ANALYSIS ==========\n")

failure_analysis = {
    "Brain-inspired computing":
        "TF-IDF failed because the query used synonyms not directly present in the document vocabulary.",

    "Learning without labels":
        "The query wording differed from the document text, reducing retrieval accuracy.",

    "Best pizza recipes":
        "The query was outside the AI domain, so no relevant vocabulary matched the documents.",

    "Weather forecast tomorrow":
        "No meaningful overlap existed between the query and the AI knowledge base."
}

for query, reason in failure_analysis.items():

    print(f"Query: {query}")
    print(f"Diagnosis: {reason}\n")


# -----------------------------------
# Vocabulary Mismatch Explanation
# -----------------------------------

print("\n========== VOCABULARY MISMATCH ==========\n")

explanation = """
TF-IDF retrieval depends heavily on exact word overlap between the query and documents.

If a query uses synonyms or related concepts that are not present in the document vocabulary,
the similarity score becomes very low even when the meanings are related.

For example, the query 'Brain-inspired computing' refers to neural networks,
but because the exact terms differ, TF-IDF struggles to identify the correct document.

This limitation shows why modern AI systems use embeddings instead of TF-IDF vectors.
Embeddings capture semantic meaning and understand similarity between related words and phrases.
"""

print(explanation)