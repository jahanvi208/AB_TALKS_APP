from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Artificial intelligence is changing the world",
    "Machine learning helps computers learn from data",
    "Deep learning is a branch of AI",
    "Neural networks are inspired by the human brain",
    "Python is popular for AI development",
    "Chatbots can answer user questions",
    "Semantic search understands meaning",
    "Embeddings convert text into vectors",
    "Natural language processing works with text data",
    "Recommendation systems suggest relevant content",
    "Football is a popular sport",
    "Cricket is loved by millions",
    "The weather is very hot today",
    "It is raining outside",
    "I enjoy listening to music",
    "Movies are a great source of entertainment",
    "Pizza tastes delicious",
    "Traveling helps people explore new places",
    "Books improve knowledge and imagination",
    "Exercise keeps the body healthy"
]

embeddings = model.encode(sentences)

query = input("Enter your search query: ")

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]

top_indices = similarities.argsort()[-3:][::-1]

print("\nTop 3 Semantic Search Results:\n")

for i in top_indices:
    print(f"Sentence: {sentences[i]}")
    print(f"Similarity Score: {similarities[i]:.4f}")
    print()


print("Keyword Search Results:\n")

for sentence in sentences:
    if query.lower() in sentence.lower():
        print(sentence)