import yfinance as yf
import tensorflow as tf
import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pandas_market_calendars as mcal

# Load the trained CNN model
model = load_model('cnn_model.h5')

# Define the sequence length and the scaler
seq_length = 5
scaler = MinMaxScaler(feature_range=(0, 1))

st.header("My Header")
st.subheader("Another header")

st.title("Stock Price Predictor")
st.write("This application predicts stock prices using a trained neural network model.")

# List of tickers used for building the model
tickers = ['AAPL', 'MSFT', 'JNJ', 'PFE', 'XOM', 'CVX']

# Create a dropdown menu for ticker selection
ticker = st.selectbox('Select a stock ticker', tickers)

# Create a date picker for date selection
date = st.date_input('Select a date for prediction')

# Function to preprocess the data
def preprocess_data(data):
    # Scale the data
    data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Create sequences
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
    X = np.array(X)
    
    # Reshape the data for the CNN model
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X

if st.button('Predict'):
    # Get the date 60 days before the selected date
    start_date = date - pd.DateOffset(days=seq_length)

    # Download the data from the start date to the selected date
    data = yf.download(ticker, start=start_date, end=date)
    
    # Check if there are enough data points
    if len(data) < seq_length:
        st.write('Not enough historical data to make a prediction. Please select a different date.')
        st.stop()

    # Preprocess the data
    data = preprocess_data(data['Close'])

    # Use the CNN model to predict the stock price
    prediction = model.predict(data)
    
    # Convert the prediction to the original price scale
    prediction = scaler.inverse_transform(prediction)
    
    # Display the prediction
    st.write(f'The predicted stock price for {ticker} on {date} is {prediction[-1][0]}')
