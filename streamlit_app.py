import streamlit as st
import numpy as np
import yfinance as yf
from tensorflow import keras
import pickle
import pandas as pd

# Load the Keras LSTM model
lstm_model = keras.models.load_model("keras_model.h5")

# Load the saved Linear Regression model
with open('linear_regression_model.pkl', 'rb') as f:
    linear_regression_model = pickle.load(f)

# Load the saved CNN model
cnn_model = keras.models.load_model("cnn_model.h5")

# Function to preprocess data for the LSTM model
\def preprocess_data_lstm(data, timesteps):
    normalized_data = (data - np.mean(data)) / np.std(data)
    if len(normalized_data) < timesteps:
        # Pad the data with zeros if the length is less than timesteps
        pad_length = timesteps - len(normalized_data)
        normalized_data = np.pad(normalized_data, (pad_length, 0), mode='constant')
    reshaped_data = np.reshape(normalized_data[-timesteps:], (1, timesteps, 1))
    return reshaped_data


# Function to preprocess data for the Linear Regression model
def preprocess_data_linear_regression(data):
    return data.reshape(-1, 1)

# Function to preprocess data for the CNN model
def preprocess_data_cnn(data, timesteps):
    normalized_data = (data - np.mean(data)) / np.std(data)
    if len(normalized_data) > timesteps:
        # Truncate the data if its length exceeds timesteps
        normalized_data = normalized_data[-timesteps:]
    elif len(normalized_data) < timesteps:
        # Pad the data with zeros at the beginning if the length is less than timesteps
        pad_length = timesteps - len(normalized_data)
        normalized_data = np.pad(normalized_data, (pad_length, 0), mode='constant')
    reshaped_data = np.reshape(normalized_data, (1, timesteps, 1))
    return reshaped_data


# Streamlit app
def main():
    # Set the page title
    st.title("Stock Price Prediction App")
    
    # Get user input for stock ticker
    stock_ticker = st.text_input("Enter a stock ticker")
    
    # Define the date range
    start = '2010-01-01'
    end = '2019-12-31'
    
    # Download the stock price data
    df = yf.download(stock_ticker, start, end)
    
    if not df.empty:
        # Preprocess the stock price data for LSTM model
        stock_price_data_lstm = df['Close'].values
        processed_data_lstm = preprocess_data_lstm(stock_price_data_lstm, timesteps=9)
        
        # Preprocess the stock price data for Linear Regression model
        stock_price_data_linear_regression = df['Close'].values
        processed_data_linear_regression = preprocess_data_linear_regression(stock_price_data_linear_regression)
        
        # Preprocess the stock price data for CNN model
        stock_price_data_cnn = df['Close'].values
        processed_data_cnn = preprocess_data_cnn(stock_price_data_cnn, timesteps=6)
        
        # Make predictions using each model
        lstm_predicted_price = lstm_model.predict(processed_data_lstm)[0][0]
        linear_regression_predicted_price = linear_regression_model.predict(processed_data_linear_regression)[0]
        cnn_predicted_price = cnn_model.predict(processed_data_cnn)[0][0]
        
        # Display the predicted prices
        st.write(f"The predicted price for {stock_ticker} (LSTM) is ${lstm_predicted_price:.2f}")
        st.write(f"The predicted price for {stock_ticker} (Linear Regression) is ${linear_regression_predicted_price:.2f}")
        st.write(f"The predicted price for {stock_ticker} (CNN) is ${cnn_predicted_price:.2f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
