import yfinance as yf
import tensorflow as tf
import streamlit as st
from keras.models import load_model
import numpy as np

# Load the trained CNN model
model = load_model('cnn_model.h5')

st.header("My Header")
st.subheader("Another header")

st.title("Stock Price Predictor")
st.write("This application predicts stock prices using a trained neural network model.")

# List of tickers used for building the model
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']

# Create a dropdown menu for ticker selection
ticker = st.selectbox('Select a stock ticker', tickers)

# Create a date picker for date selection
date = st.date_input('Select a date for prediction')


if st.button('Predict'):
  data = yf.download(ticker, end=date, periods='1y')
  st.line_chart(data['Close'])

  # Preprocess the data in the same way as before
  data = preprocess_data(data)

  # Reshape the data for the CNN model
  data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    
  # Use the CNN model to predict the stock price
  prediction = model.predict(data)
    
  # Convert the prediction to the original price scale
  prediction = scaler.inverse_transform(prediction)
  
  
  # Display the predictions
  st.line_chart(predictions)
  
   # Display the prediction
   st.write(f'The predicted stock price for {ticker} on {date} is {prediction[-1][0]}')
