from flask import Flask, request, jsonify
# import os
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from transformers import pipeline
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from collections import defaultdict
# import h5py
from intent import intents

vectorizer = TfidfVectorizer()
corpus = [pattern for intent in intents for pattern in intent["patterns"]]
vectorizer.fit(corpus)

X = vectorizer.transform(corpus)
y = [intent["tag"] for intent in intents for _ in intent["patterns"]]
classifier = LogisticRegression(random_state=0, max_iter=10000)
classifier.fit(X, y)

filename = r"mdl_chat.pkl"
pickle.dump((vectorizer, classifier), open(filename, "wb"))

loaded_vectorizer, loaded_classifier = pickle.load(open(filename, "rb"))


def chatbot_response(text):
    input_text = loaded_vectorizer.transform([text])
    predicted_tag = loaded_classifier.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            return response


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.get_json()["message"]  # Assuming JSON input
    response = chatbot_response(user_input)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(port=5000)