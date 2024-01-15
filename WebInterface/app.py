# pip install Flask

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from transformers import DistilBertTokenizer, TFDistilBertModel
import numpy as np


app = Flask(__name__)

# Load your DistilBERT model
model = keras.models.load_model('../Models/DistilBERT_Model')

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')



def predict_helpfulness(review_text):
    # Tokenize and pad the input text
    inputs = tokenizer(review_text, return_tensors='tf', truncation=True, padding=True, max_length=512)

    # Extract tensors from BatchEncoding
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Make predictions
    outputs = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})

    # Extract the logits from the output dictionary
    logits = outputs['logits']

    # Convert logits to probabilities using a sigmoid activation
    probabilities = tf.sigmoid(logits)

    # Convert probabilities to binary predictions
    binary_predictions = tf.cast(probabilities > 0.5, tf.int32)

    # Extract confidence level (probability)
    confidence = 1 - probabilities.numpy()[0, 0]

    # Determine the helpfulness information
    prediction_result = "Review is Helpful" if binary_predictions.numpy()[0, 0] == 0 else "Review is not Helpful"

    # Return result and confidence
    return prediction_result, confidence



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        prediction_result, confidence = predict_helpfulness(review)
        return render_template('index.html', review=review, result=prediction_result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)