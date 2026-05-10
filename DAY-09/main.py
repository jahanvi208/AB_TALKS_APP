
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = []

print("Enter 5 sentences:")
for i in range(5):
    s = input(f"Sentence {i+1}: ")
    sentences.append(s)

embeddings = model.encode(sentences)

similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity Matrix:\n")
for row in similarity_matrix:
    print(["{:.2f}".format(score) for score in row])

print("\nCompare two new sentences:")

s1 = input("Sentence 1: ")
s2 = input("Sentence 2: ")

emb1 = model.encode([s1])
emb2 = model.encode([s2])

score = cosine_similarity(emb1, emb2)[0][0]

print("\nSimilarity Score:", round(score, 2))