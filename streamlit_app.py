import streamlit as st
import numpy as np
import yfinance as yf
from tensorflow import keras

# Load the Keras model
model_1 = keras.models.load_model("keras_model.h5")

# Function to preprocess the stock data
def preprocess_data(data, timesteps):
    # Normalize the data
    normalized_data = (data - np.mean(data)) / np.std(data)
    
    # Create input sequences
    sequences = []
    for i in range(timesteps, len(normalized_data)):
        sequences.append(normalized_data[i - timesteps:i])
    
    # Convert to numpy array
    sequences = np.array(sequences)
    
    # Reshape the data to match the model input shape
    reshaped_data = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))
    
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
        timesteps = 9  # Update this value according to your LSTM model
        processed_data = preprocess_data(stock_price_data, timesteps)
        
        # Make predictions using the model
        predicted_price = model_1.predict(processed_data)
        
        # Display the predicted price
        st.write(f"The predicted price for {stock_ticker} is ${predicted_price[-1][0]:.2f}")
    else:
        st.write("No data available for the given stock ticker.")
    
# Run the Streamlit app
if __name__ == '__main__':
    main()
