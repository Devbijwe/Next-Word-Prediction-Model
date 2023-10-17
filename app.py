from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Read the text file and initialize the model, tokenizer, and other parameters
with open('dataset/sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Initialize the model
max_sequence_len = 17
model = tf.keras.models.load_model('trained_model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.form['seed_text']
    next_words = int(request.form['next_words'])
    num_results = 3  # Number of result sets to generate

    # Make sure max_sequence_len matches what was used during training
    max_sequence_len = 17  # Use the same value as used during training

    # Temperature for diversity (adjust as needed)
    temperature = 0.6

    # Generate the text
    results = []

    for _ in range(num_results):
        predicted_words = []
        current_seed = seed_text

        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([current_seed])[0]

            # Ensure that the token_list has the same length as max_sequence_len
            # You may need to pad or trim the input sequence as necessary
            token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

            # Predict the probabilities and apply temperature scaling
            predictions = model.predict(token_list)[0]
            predictions = np.log(predictions) 
            exp_predictions = np.exp(predictions)
            predicted_probabilities = exp_predictions / np.sum(exp_predictions)

            # Sample the next word based on the modified probabilities
            predicted_index = np.random.choice(len(predicted_probabilities), p=predicted_probabilities)
            output_word = ""

            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break

            current_seed += " " + output_word
            predicted_words.append(output_word)

        predicted_text = " ".join(predicted_words)
        results.append(predicted_text)

    return jsonify({'results': results})

