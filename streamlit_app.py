import streamlit as st
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Keras model
model_1 = load_model("keras_model.h5")

# Set the start and end dates for the stock price data
start_date = "2010-01-01"
end_date = "2019-12-31"

# Create the Streamlit app
st.title("Stock Price Prediction App")

# Create an input field for the stock ticker
stock_ticker = st.text_input("Enter a stock ticker")

# Create a button to trigger the prediction
predict_button = st.button("Predict")

if predict_button:
    # Load the stock price data
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    # Preprocess the stock price data (if necessary)
    # ...

    # Perform the prediction using the loaded model
    predicted_price = model_1.predict(stock_data)

    # Display the predicted price
    st.subheader("Predicted Price")
    st.write(predicted_price)

    # Plot the stock price data and predicted prices
    st.subheader("Stock Price Chart")
    st.line_chart(stock_data["Close"])
    st.line_chart(predicted_price)
