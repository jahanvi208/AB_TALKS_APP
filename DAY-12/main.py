import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

print("Enter a long paragraph:\n")
user_text = input()

cleaned_text = clean_text(user_text)

chunks = chunk_text(cleaned_text)

print("\nGenerated Chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("-" * 50)