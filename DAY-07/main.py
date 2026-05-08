
texts = [
    "I love this movie",
    "This film was amazing",
    "I really liked the story",
    "This was a great experience",
    "Absolutely fantastic",

    "I hated this movie",
    "This was terrible",
    "Worst film ever",
    "I did not like this",
    "Very bad experience"
]

labels = [
    "positive","positive","positive","positive","positive",
    "negative","negative","negative","negative","negative"
]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, labels)


while True:
    user_input = input("\nEnter a sentence (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)

    print("Prediction:", prediction[0])


from sklearn.metrics import accuracy_score
train_predictions = model.predict(X)
accuracy = accuracy_score(labels, train_predictions)

print("\nTraining Accuracy:", accuracy)