import streamlit as st

st.title("Stock Price Predictor")
st.write("This application predicts stock prices using a trained neural network model.")

# List of tickers used for building the model
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']

# Create a dropdown menu for ticker selection
ticker = st.selectbox('Select a stock ticker', tickers)

# Create a date picker for date selection
date = st.date_input('Select a date for prediction')

# Display the prediction
#st.write(f'The predicted stock price for {ticker} on {date} is)
