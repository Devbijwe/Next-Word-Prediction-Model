import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# Define a list of datasets with their source URLs
datasets = [
    {"name": "Shakespeare", "url": "https://www.gutenberg.org/files/100/100-0.txt"},
    {"name": "Wikipedia", "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"},
    {"name": "News", "url": "https://www.kaggle.com/asad1m9a9h6mood/news-articles"},
    {"name": "SongLyrics", "url": "https://www.kaggle.com/mousehead/songlyrics"},
    {"name": "MovieScripts", "url": "https://www.kaggle.com/jrobischon/wikipedia-movie-plots"},
    {"name": "Books", "url": "https://www.gutenberg.org"},
    {"name": "TechDocs", "url": "https://www.sciencedirect.com/journal/computers-in-industry"},
    {"name": "Conversations", "url": "https://www.kaggle.com/c/siim-covid19-detection/data"},
    # Add more datasets with names and URLs as needed
]


def preprocess_data(text_data):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_data])
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences and labels
    input_sequences = []
    for line in text_data.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len =17 or max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

    return X, y

# Function to download and preprocess a dataset
def download_and_preprocess_dataset(dataset):
    # Download the dataset
    response = requests.get(dataset["url"])
    text = response.text

    # Preprocess the text (e.g., remove metadata, clean up)
    if dataset["name"] == "Shakespeare":
        # Customize preprocessing for Shakespeare's dataset
        # For example, remove the Gutenberg header/footer
        text = text[text.find("<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM"):]
        text = text[:text.find("FINIS.")]

    # Tokenize and preprocess the text (generic example)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Save the preprocessed text to a file
    with open(f"{dataset['name']}_text.txt", "w", encoding="utf-8") as file:
        file.write(text)

    return text

# Function to train a model on a given dataset
def train_model_on_dataset(dataset_name, text_data,max_sequence_len = 17):
    
    # Tokenize and preprocess the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_data])
    total_words = len(tokenizer.word_index) + 1

    # Define the model architecture and training process
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    X, y = preprocess_data(text_data)
    model.fit(X, y, epochs=100, verbose=1)

    # Save the model for this dataset
    model.save(f"models/{dataset_name}_model")

# Main script
for dataset in datasets:
    dataset_name = dataset["name"]
    text_data = download_and_preprocess_dataset(dataset)
    if not os.path.exists('models'):
        os.makedirs('models')
    train_model_on_dataset(dataset_name, text_data)

