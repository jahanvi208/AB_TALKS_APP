import re
import string

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
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

def process_text(text):
    cleaned = clean_text(text)
    no_stopwords = remove_stopwords(cleaned)
    tokens = no_stopwords.split()

    return {
        "cleaned_text": cleaned,
        "without_stopwords": no_stopwords,
        "tokens": tokens
    }
while True:
    user_input = input("\nEnter text (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    result = process_text(user_input)

    print("\nCleaned Text:", result["cleaned_text"])
    print("Without Stopwords:", result["without_stopwords"])
    print("Tokens:", result["tokens"])