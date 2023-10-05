# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
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

    # Predict using the Neural Network model
    prediction = model.predict(scaled_features)
    
    # Return the class with the highest probability
    return np.argmax(prediction, axis=1)

# Title of the app
st.title("Iris Flower Classifier")

# Instructions
st.write("""
Select the features of the Iris flower you'd like to classify:
""")

# Sliders to input features
sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.7)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.4)

# Arrange the features as an array
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Get the prediction
prediction = predict(model, scaler, encoder, features)

# Display the prediction
st.subheader("Prediction:")
st.write(['setosa', 'versicolor', 'virginica'][prediction[0]])
