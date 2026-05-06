from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = []
for i in range(5):
    s = input(f"Enter sentence {i+1}: ")
    sentences.append(s)

embeddings = model.encode(sentences)

similarity_matrix = cosine_similarity(embeddings)

print("\nSentences:")
for i, s in enumerate(sentences):
    print(f"{i+1}. {s}")


print("\nSimilarity Matrix:\n", similarity_matrix)

print("\nSentence 1 vs Sentence 2:",
      cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
