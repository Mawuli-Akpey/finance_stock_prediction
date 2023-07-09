import yfinance as yf
import tensorflow as tf
import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_market_calendars as mcal

# Load the trained CNN model
model = load_model('cnn_model.h5')

# Define the sequence length and the scaler
seq_length = 60
scaler = MinMaxScaler(feature_range=(0, 1))

st.header("My Header")
st.subheader("Another header")

st.title("Stock Price Predictor")
st.write("This application predicts stock prices using a trained neural network model.")

# List of tickers used for building the model
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

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

# Function to check if a date is a valid trading day
def is_trading_day(date):
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date='2018-01-01', end_date=date)
    return np.isin([date], trading_days)

if st.button('Predict'):
    # Check if the selected date is a valid trading day
    if not is_trading_day(date):
        st.write('The selected date is not a valid trading day. Please select a different date.')
        st.stop()

    # Download the data
    data = yf.download(ticker, end=date, periods='1y')

    # Check if the downloaded data has enough data points
    if len(data) < seq_length:
        st.write('Not enough data to make a prediction. Please select a different date or ticker.')
        st.stop()

    # Preprocess the data
    data = preprocess_data(data['Close'])

    # Use the CNN model to predict the stock price
    try:
        prediction = model.predict(data)
    except Exception as e:
        st.write('An error occurred during the prediction: ', e)
        st.stop()

    # Convert the prediction to the original price scale
    prediction = scaler.inverse_transform(prediction)
    
    # Display the prediction
    st.write(f'The predicted stock price for {ticker} on {date} is {prediction[-1][0]}')
