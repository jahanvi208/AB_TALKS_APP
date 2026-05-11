
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

stopwords = {
    "is", "am", "are", "the", "a", "an", "and", "or", "in", "on", "at",
    "to", "for", "of", "with", "this", "that", "it", "was"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def remove_stopwords(text):
    words = text.split()
    return " ".join([w for w in words if w not in stopwords])

def process(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

sentences = [
    "Artificial intelligence is transforming the world",
    "Machine learning is a part of AI",
    "I love learning new technologies",
    "The weather is very hot today",
    "It is sunny outside",
    "AI is used in recommendation systems",
    "Deep learning powers modern applications",
    "I enjoy coding in Python",
    "Today is a bright sunny day",
    "Technology is evolving rapidly"
]

processed_sentences = [process(s) for s in sentences]

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(processed_sentences)

while True:
    query = input("\nEnter your query (or type 'exit'): ")

    if query.lower() == "exit":
        break

    query_processed = process(query)
    query_embedding = model.encode([query_processed])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = scores.argsort()[-3:][::-1]

    print("\nTop 3 Results:")
    for idx in top_indices:
        print(f"- {sentences[idx]} (score: {round(scores[idx], 2)})")