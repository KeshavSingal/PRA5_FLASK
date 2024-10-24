from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
import time

# Initialize Flask app
app = Flask(__name__)


def load_model():
    """Load the ML model and vectorizer"""
    try:
        # Loading the model from pickle file
        loaded_model = None
        with open('basic_classifier.pkl', 'rb') as fid:
            loaded_model = pickle.load(fid)

        vectorizer = None
        with open('count_vectorizer.pkl', 'rb') as vd:
            vectorizer = pickle.load(vd)

        return loaded_model, vectorizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise


# Load models at startup
model, vectorizer = load_model()


@app.route('/')
def home():
    return "Fake News Detector API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Start timing for latency measurement
        start_time = time.time()

        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        news_text = data['text']

        # Vectorize the text
        text_vectorized = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(text_vectorized)[0]

        # Convert prediction to integer if it's a string
        if isinstance(prediction, str):
            prediction_value = 1 if prediction == 'FAKE' else 0
        else:
            prediction_value = int(prediction)

        # Calculate latency
        latency = time.time() - start_time

        # Return result (1 for fake news, 0 for real news)
        return jsonify({
            'text': news_text,
            'prediction': prediction_value,
            'is_fake_news': bool(prediction_value),
            'latency': latency
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)