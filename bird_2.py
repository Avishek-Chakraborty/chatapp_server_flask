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
import json
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from intent import intents


# Hugging Face's Transformers library is commonly used for natural language processing tasks like text classification, language modeling, 
# The API token you're referencing is likely required for authentication.



x=os.environ['Huggingface_Api_token']="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"

def remove_newlines(text):
    """Remove newline characters from the text."""
    return text.replace('\n', '')

def replace_slash_with_or(text):
    """Replace '/' with 'or' in the text."""
    return text.replace('/', ' or ')

def remove_brackets(text):
    """Remove brackets from the text."""
    return text.replace('(', '').replace(')', '')

def replace_dashes_with_space(text):
    """Replace '____' with a space in the text."""
    return text.replace('____', '')

def replace_multiple_dash_with_space_and_respective_text(text):
    # Replace multiple underscores with an empty string
    text = text.replace('____', '')

    # Replace single underscore with "what is your choice.."
    if '_' in text:
        text = text.replace('_', 'what is your choice')

    return text

def preprocess_text(text):
    """Preprocess the text using all defined functions."""
    text = remove_newlines(text)
    text = replace_slash_with_or(text)
    text = remove_brackets(text)
    text = replace_dashes_with_space(text)
    text = replace_multiple_dash_with_space_and_respective_text(text)
    return text


# Specify encodings to try
encodings = ['utf-8', 'latin-1', 'utf-16']

# Loop through encodings and attempt to load the document
for encoding in encodings:
    try:
        loader = TextLoader(r"C:\Users\biswa\Downloads\VULNERABILITY TO DEPRESSION.txt", encoding=encoding)
        depression_document = loader.load()
        print("Document loaded successfully using encoding:", encoding)
        break  # Break out of the loop if successful
    except Exception as e:
        print("Error loading document with encoding:", encoding)
        print(e)


# trying to preprocess text from a document about depression

final_text=preprocess_text(str(depression_document[0]))

#  to split text into chunks based on character count with certain parameters 
# like separator, chunk size, and chunk overlap specified.

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len,
    is_separator_regex=False,
)


# Split the document using the splitter
depression_docs = text_splitter.split_documents(depression_document)

# # FAISS: FAISS is a library for efficient similarity search and clustering of dense vectors.
# HuggingFaceEmbeddings:to be a class or module that provides embeddings using models from Hugging Face's Transformers library.

embeddings=HuggingFaceEmbeddings()
db = FAISS.from_documents(depression_docs,embeddings)


# FLAN-T5 is a family of large language models trained at Google, 
huggingface_hub = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.8, "max_length": 2048}, huggingfacehub_api_token=x)

# Load the question-answering chain
chain = load_qa_chain(huggingface_hub, chain_type="stuff")

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
filename = "mdl_chat.pkl"
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

# Check if model and tokenizer files exist
model_path = "model.h5"
tokenizer_path = "tokenizer.pkl"

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
        return f"Percentage of depression: {round(prediction * 100, 3)}% - No depression"
    elif prediction < thresholds[1]:
        return f"Percentage of depression: {round(prediction * 100, 3)}% - Mild depression"
    elif prediction < thresholds[2]:
        return f"Percentage of depression: {round(prediction * 100, 3)}% - Moderate depression"
    elif prediction < thresholds[3]:
        return f"Percentage of depression: {round(prediction * 100, 3)}% - Moderately severe depression"
    else:
        return f"Percentage of depression: {round(prediction * 100, 3)}% - Severe depression"

def overall_emotional_quotient(results):
    category_data = defaultdict(lambda: [0, 0])
    for category, value in results:
        category_data[category][0] += value
        category_data[category][1] += 1
    averages = {category: sum_value / count for category, (sum_value, count) in category_data.items()}
    max_category = max(averages, key=averages.get)
    max_average = averages[max_category]
    return {"Category with the highest average": max_category, "Average value": max_average}

def emotional_quotient_avg_each_cat(results):
    emotion_data = defaultdict(lambda: [0, 0])
    for emotion, value in results:
        emotion_data[emotion][0] += value
        emotion_data[emotion][1] += 1
    averages = {emotion: sum_value / count for emotion, (sum_value, count) in emotion_data.items()}
    return {f"Average for {emotion}": average for emotion, average in averages.items()}

output_json = {}

while True:
    try:
        query = input("User-> ")
        output = chatbot_response(query)
        print("Chatbot-> {}".format(output))
        docsResult = db.similarity_search(query)
        print(f"answring the questions through precise manner :{chain.run(input_documents=docsResult,question = query)}")
        print("broad result:")
        print(preprocess_text(str(docsResult[0].page_content)))

        sum, cnt, results = process_query(query, sum, cnt, results)
        dynamic_depression_prediction = depression_measure_dynamic(query)
        depression_result = depression_severity(dynamic_depression_prediction)
        print(depression_result)
        output_json["depression_result"] = depression_result
        output_json["depression_result"] = depression_result
        print("This value is dynamic and can be plotted in a scatterplot and KDE chart for depression.")
        print("---------------")

        goodbye_phrases = ["thank you", "bye", "goodbye", "see you later", "see you soon", "take care"]
        exit_loop = False
        for phrase in goodbye_phrases:
            if phrase in query.lower():
                exit_loop = True
                overall_depression_value, counts = depression_measure_severity(query, predicted_value,counter)
                overall_depression_value, counts = depression_measure_severity(query, predicted_value, counter)
                overall_depression_percentage = round(overall_depression_value / counts * 100, 3)
                print(f"Overall depression value: {overall_depression_percentage}%")
                output_json["overall_depression_percentage"] = overall_depression_percentage
                print("================")
                overall_emotional_quotient_results = overall_emotional_quotient(results)
                print("Overall Emotional quotient:", overall_emotional_quotient_results)
                output_json["overall_emotional_quotient_results"] = overall_emotional_quotient_results
                averages = emotional_quotient_avg_each_cat(results)
                print("Averages:", averages)
                output_json["emotional_quotient_avg_each_cat"] = averages
                print("================")
                print(results)
                output_json["results"] = results
                print("================")
                print("This data is dynamic emotional distribution and can be used to plot KDE plot, scatter plot, and top 5 for pie chart distribution.")
                break
        if exit_loop:
            break
    except Exception as e:
        print("Error:", e)

# Convert the output to JSON
output_json_str = json.dumps(output_json, indent=4)
print("Output JSON:")
print(output_json_str)

# Write the JSON to a file
with open("output.json", "w") as json_file:
    json_file.write(output_json_str)

