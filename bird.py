import os
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import h5py
from intent import intents
# from flask import Flask, request, jsonify


# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
corpus = [pattern for intent in intents for pattern in intent['patterns']]
vectorizer.fit(corpus)

# Transform the patterns and train the classifier
X = vectorizer.transform(corpus)
y = [intent['tag'] for intent in intents for _ in intent['patterns']]
classifier = LogisticRegression(random_state=0, max_iter=10000)
classifier.fit(X, y)

# Save the fitted vectorizer and classifier
filename = r"mdl_chat.pkl"
pickle.dump((vectorizer, classifier), open(filename, "wb"))

# Load the trained model
loaded_vectorizer, loaded_classifier = pickle.load(open(filename, "rb"))

def chatbot_response(text):
    input_text = loaded_vectorizer.transform([text])
    predicted_tag = loaded_classifier.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            return response

def roberta_classifier(query):
    classifier = pipeline(
        task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
    )
    model_outputs = classifier(query)
    data = model_outputs[0][0]
    label_value = data["label"]
    score_value = data["score"]
    return label_value, score_value

def process_query(query, sum, cnt, results):
    label, score = roberta_classifier(str(query))
    quotient = float(score) * 100
    sum += quotient
    results.append([label, quotient])
    cnt += 1
    return sum, cnt, results

# Check if model and tokenizer files exist model.h5
model_path = r"model.h5"
tokenizer_path = r"tokenizer.pkl"

if not os.path.exists(model_path):
    print(f"Error: The model file '{model_path}' does not exist.")
    exit(1)

if not os.path.exists(tokenizer_path):
    print(f"Error: The tokenizer file '{tokenizer_path}' does not exist.")
    exit(1)

# Check if the model file is a valid HDF5 file
def is_hdf5_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            return True
    except OSError:
        return False

if not is_hdf5_file(model_path):
    print(f"Error: The model file '{model_path}' is not a valid HDF5 file.")
    exit(1)

try:
    model_form = load_model(model_path)
except Exception as e:
    print(f"Error loading the model from '{model_path}': {e}")
    exit(1)

try:
    with open(tokenizer_path, 'rb') as file:
        token_form = pickle.load(file)
except Exception as e:
    print(f"Error loading the tokenizer from '{tokenizer_path}': {e}")
    exit(1)

results = []
sum = 0
cnt = 0
predicted_value = 0
counter = 0

def depression_measure_severity(query, predicted_value, counter):
    twt = token_form.texts_to_sequences([query])
    twt = pad_sequences(twt, maxlen=50)
    prediction = model_form.predict(twt)[0][0]
    predicted_value += prediction
    counter += 1
    return predicted_value, counter

def depression_measure_dynamic(query):
    query_seq = token_form.texts_to_sequences([query])
    query_padded = pad_sequences(query_seq, maxlen=50)
    prediction = model_form.predict(query_padded)[0][0]
    return prediction

def depression_severity(prediction):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    if prediction < thresholds[0]:
        print(f"Percentage of depression: {round(prediction * 100, 3)}% - No depression")
    elif prediction < thresholds[1]:
        print(f"Percentage of depression: {round(prediction * 100, 3)}% - Mild depression")
    elif prediction < thresholds[2]:
        print(f"Percentage of depression: {round(prediction * 100, 3)}% - Moderate depression")
    elif prediction < thresholds[3]:
        print(f"Percentage of depression: {round(prediction * 100, 3)}% - Moderately severe depression")
    else:
        print(f"Percentage of depression: {round(prediction * 100, 3)}% - Severe depression")

def overall_emotional_quotient(results):
    category_data = defaultdict(lambda: [0, 0])
    for category, value in results:
        category_data[category][0] += value
        category_data[category][1] += 1
    averages = {category: sum_value / count for category, (sum_value, count) in category_data.items()}
    max_category = max(averages, key=averages.get)
    max_average = averages[max_category]
    print("Category with the highest average:", max_category)
    print("Average value:", max_average)
    return max_category, max_average

def emotional_quotient_avg_each_cat(results):
    emotion_data = defaultdict(lambda: [0, 0])
    for emotion, value in results:
        emotion_data[emotion][0] += value
        emotion_data[emotion][1] += 1
    averages = {emotion: sum_value / count for emotion, (sum_value, count) in emotion_data.items()}
    for emotion, average in averages.items():
        print(f"Average for {emotion}: {average}")
    return averages

while True:
    try:
        query = input("User-> ")
        output = chatbot_response(query)
        print("Chatbot-> {}".format(output))

        sum, cnt, results = process_query(query, sum, cnt, results)
        dynamic_depression_prediction = depression_measure_dynamic(query)
        depression_severity(dynamic_depression_prediction)
        # print("This value is dynamic and can be plotted in a scatterplot and KDE chart for depression.")
        print("---------------")

        goodbye_phrases = ["thank you", "bye", "goodbye", "see you later", "see you soon", "take care"]
        exit_loop = False
        for phrase in goodbye_phrases:
            if phrase in query.lower():
                exit_loop = True
                overall_depression_value, counts = depression_measure_severity(query, predicted_value, counter)
                print(f"Overall depression value: {round(overall_depression_value / counts * 100, 3)}%")
                print("================")
                overall_emotional_quotient_results = overall_emotional_quotient(results)
                print("Overall Emotional quotient:", overall_emotional_quotient_results)
                averages = emotional_quotient_avg_each_cat(results)
                print("Averages:", averages)
                print("================")
                print(results)
                print("================")
                # print("This data is dynamic emotional distribution and can be used to plot KDE plot, scatter plot, and top 5 for pie chart distribution.")
                break
        if exit_loop:
            break
    except Exception as e:
        print("Error:", e)


# @app.route('/api', methods=['POST'])

# def api():
#     data = request.get_json(force=True)  # get the incoming data
#     message = data['message']  # extract the message from the incoming JSON
#     response = chatbot_response(message)  # get the response from your chatbot
#     return jsonify(response=response)  # return the response as JSON


# if __name__ == '__main__':
#     app.run(port=5000)