import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


# Load the LSTM model
model = load_model("LSTM_model.h5")

# Load your tokenizer (make sure to save and load your tokenizer)
with open('C://Users//91770//Desktop//TwitterBotDetectionPproject//tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Streamlit app interface
st.title("LSTM Model Prediction App")

# User input for the model
st.write("Provide input text for prediction:")
input_text = st.text_area("Enter text:")

# Function to preprocess the input data
def preprocess_input(text):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Use the same maxlen used during training
    return padded_sequence

# When user clicks "Predict"
if st.button("Predict"):
    if input_text:
        data = preprocess_input(input_text)
        prediction = model.predict(data)
        st.write("Prediction:", prediction[0][0])  # Assuming binary output; adjust if necessary
        if prediction[0][0] > 0.5:
            st.write(f"Prediction: This user is likely a bot with {prediction[0][0]:.2f} confidence.")
        else:
            st.write(f"Prediction: This user is likely not a bot with {1-prediction[0][0]:.2f} confidence.")

    else:
        st.error("Please enter some text for prediction.")
