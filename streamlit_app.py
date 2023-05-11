import streamlit as st
import numpy as np
import yfinance as yf
from tensorflow import keras

# Load the Keras model
model_1 = keras.models.load_model("keras_model.h5")

# Function to preprocess the stock data
# Function to preprocess the stock data
def preprocess_data(data):
    # Normalize the data
    normalized_data = (data - np.mean(data)) / np.std(data)
    
    # Create sequences of length 9
    sequence_length = 9
    sequences = []
    for i in range(sequence_length, len(normalized_data)):
        sequence = normalized_data[i - sequence_length:i]
        sequences.append(sequence)
    
    # Convert sequences to numpy array
    sequences = np.array(sequences)
    
    # Reshape the sequences to match the model input shape
    reshaped_data = np.reshape(sequences, (sequences.shape[0], sequence_length, 1))
    
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
        # Preprocess the stock price data
        stock_price_data = df['Close'].values
        processed_data = preprocess_data(stock_price_data)
        
        # Make predictions using the model
        predicted_price = model_1.predict(processed_data)
        
        # Display the predicted price
        st.write(f"The predicted price for {stock_ticker} is ${predicted_price[0][0]:.2f}")
    else:
        st.write("No data available for the given stock ticker.")
    
# Run the Streamlit app
if __name__ == '__main__':
    main()
