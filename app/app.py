from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import requests
import ssl
from transformers import BertTokenizer, BertModel
import torch
from flask_cors import CORS
import os

MODEL_URL = 'https://drive.google.com/file/d/1yV2ruB2b6L6Y1lnY_E_daDCPBxntrDuZ/view?usp=sharing'
MODEL_PATH = 'random_forest_model.joblib'

def download_model(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download model: {response.status_code}")

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

app = Flask(__name__)
CORS(app)
app.debug = False
random_forest_model = load('random_forest_model.joblib')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_feature_engineering = BertModel.from_pretrained('bert-base-uncased')

def fetch_issue_info(issue_url):
    # Assuming issue_url format is "https://github.com/{owner}/{repo}/issues/{issue_number}"
    parts = issue_url.split('/')
    owner, repo, issue_number = parts[-4], parts[-3], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    response = requests.get(api_url)
    if response.status_code == 200:
        issue_data = response.json()
        return {
            "title": issue_data.get("title", ""),
            "body": issue_data.get("body", ""),
            "author_association": issue_data.get("author_association", ""),
            "is_pull_request": issue_data.get("is_pull_request", False),
        }
    else:
        return None  # or handle errors as needed


def preprocess_issue_text(issue_body, issue_title):
    issue_body = issue_body if issue_body is not None else ""
    issue_title = issue_title if issue_title is not None else ""
    text = issue_title + " " + issue_body
    text = text.replace("\r", " ").replace("\n", " ").lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word not in stop_words]
    preprocessed_text = ' '.join(filtered_words)
    return preprocessed_text


def transform_text_to_features(text):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=64, pad_to_max_length=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get embeddings from BERT
    with torch.no_grad():
        outputs = model_feature_engineering(input_ids, attention_mask=attention_mask)
    # Use pooled output for simplicity
    pooled_output = outputs[1].squeeze().numpy()
    
    return pooled_output


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    issue_url = data['issue_url']
    issue_info = fetch_issue_info(issue_url)
    if issue_info is None:
        return jsonify({'error': 'Could not fetch issue info'}), 404
    
    preprocessed_text = preprocess_issue_text(issue_info['body'], issue_info['title'])
    text_features = transform_text_to_features(preprocessed_text)
    author_associations = {'COLLABORATOR': 0, 'CONTRIBUTOR': 1, 'MEMBER': 2, 'NONE': 3}
    author_feature = [0] * len(author_associations)
    if issue_info['author_association'] in author_associations:
        author_feature[author_associations[issue_info['author_association']]] = 1
    
    is_pull_request = 1 if issue_info['is_pull_request'] else 0
    features = np.concatenate((text_features, author_feature, [is_pull_request]))
    # If features somehow became a 3D array, reshape it to 2D.
    features_2d = features.reshape(1, -1)
    print("Corrected features shape for prediction:", features_2d.shape)
    prediction = random_forest_model.predict(features_2d)


    # Convert prediction to a meaningful output
    time_to_close = np.expm1(prediction[0])  # If you used log1p during training
    return jsonify({'predicted_time_to_close_hours': time_to_close})


if __name__ == '__main__':
    app.run()
