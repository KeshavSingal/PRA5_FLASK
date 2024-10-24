from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
import time

# Initialize Flask app - name must be 'application' for Elastic Beanstalk
application = Flask(__name__)
app = application  # For local testing convenience

def load_model():
    """Load the ML model and vectorizer"""
    try:
        # Loading the model from pickle file
        with open('basic_classifier.pkl', 'rb') as fid:
            loaded_model = pickle.load(fid)

        with open('count_vectorizer.pkl', 'rb') as vd:
            vectorizer = pickle.load(vd)

        return loaded_model, vectorizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

# Load models at startup
model, vectorizer = load_model()

@application.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Fake News Detector API is running!"
    })

@application.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict if news is fake or real"""
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
    # Use application.run() for consistency
    application.run(debug=True, port=5000)