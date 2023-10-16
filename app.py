# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Load the pickled models and keras model
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model = load_model('iris_nn_model.h5')

# Function to predict
@st.cache
def predict(model, scaler, encoder, features):
    # Scale the features
    scaled_features = scaler.transform(features)
    scaled_features = encoder.transform(features)
    scaled_features = pd.get_dummies(scaled_features)

    # Predict using the Neural Network model
    prediction = model.predict(scaled_features)
    
    # Return the class with the highest probability
    return np.argmax(prediction, axis=1)

# Title of the app
st.title("Home Credit Default Risk")

# Instructions
st.write("""
Upload the csv file of record you want to classify:
""")

# Title and instructions
st.title("CSV File Upload and Prediction Display")
st.write("Upload a CSV file, and we'll display if there is a chance of default.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

df = pd.DataFrame()
if uploaded_file is not None:
    # Check if a file has been uploaded
    df = pd.read_csv(uploaded_file)  # Read the uploaded CSV file into a DataFrame

    # Display the DataFrame
    st.subheader("DataFrame:")
    st.dataframe(df)  # Display the DataFrame in the app

    # You can perform further data analysis or visualization on the DataFrame here

# Get the prediction
prediction = predict(model, scaler, encoder, df)

# Display the prediction
st.subheader("Prediction:")
st.write(['Default', 'Does Not Default'][prediction[0]])
