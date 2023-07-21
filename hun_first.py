import joblib

# Load the objects from the joblib files
data_scaled = joblib.load('data_scaled.joblib')
sequence_length = joblib.load('sequence_length.joblib')
gru_model = joblib.load('gru_model.joblib')
scaler = joblib.load('scaler.joblib')


import streamlit as st
import datetime
import numpy as np

# Your function to predict the future closing price
def predict_future_close_price(model, scaler, initial_sequence, future_date, start_date):
    # Calculate the number of days in the future the future_date is from the start_date
    delta_days = (future_date - start_date).days

    # Initialize the sequence with the most recent known closing prices
    sequence = initial_sequence

    # Loop over each day up to the future_date
    for _ in range(delta_days):
        # Prepare the sequence for prediction
        sequence_reshaped = sequence.reshape((1, sequence_length, 1))

        # Make a prediction for the next day
        prediction_scaled = model.predict(sequence_reshaped)

        # Append the prediction to the sequence
        sequence = np.append(sequence[1:], prediction_scaled)

    # Inverse transform the final prediction
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction[0][0]

# Your function to get the most recent weekday
def most_recent_weekday():
    today = datetime.date.today()
    offset = max(1, (today.weekday() + 6) % 7 - 3)
    most_recent = today - datetime.timedelta(offset)
    return most_recent

# Define a dictionary to map month names to numbers
month_to_number = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Start of the Streamlit app
st.write("""
# Stock Closing Price Prediction App
Warning: This app is for informational purposes only. Please make your financial decisions with caution.
""")

# Ask the user to input the year, month, and day
year = st.number_input('Enter a year', min_value=2023, max_value=2030, value=2023, step=1)
month = st.selectbox('Select a month', list(month_to_number.keys()))
day = st.number_input('Enter a day', min_value=1, max_value=31, value=1, step=1)

# When the 'Predict' button is clicked, make the prediction
if st.button('Predict'):
    # Display a message while the prediction is being calculated
    st.write('Please hold on while we calculate...')

    # The start date (the date of the last known closing price)
    start_date = most_recent_weekday()

    # Convert the month from text to number
    month_number = month_to_number[month]

    # The future date (the date for which the user wants to predict the closing price)
    future_date = datetime.date(year, month_number, day)

    # Check if the future date is a weekday
    if future_date.weekday() >= 5:  # 0-4 denotes Monday-Friday
        st.write('Error: The future date must be a weekday.')
    elif future_date <= start_date:
        st.write('Error: The future date must be after the most recent weekday.')
    else:
        # The initial sequence of the most recent known closing prices
        # Replace with your actual data
        initial_sequence = data_scaled[-sequence_length:]

        # Use the function to predict the closing price for the future date
        predicted_close_price = predict_future_close_price(gru_model, scaler, initial_sequence, future_date, start_date)

        # Display the predicted closing price
        st.write(f"The predicted closing price for {future_date.strftime('%Y-%m-%d')} is {predicted_close_price}")
