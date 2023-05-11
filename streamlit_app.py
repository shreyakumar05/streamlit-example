import streamlit as st
import numpy as np
import yfinance as yf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# Load the Keras model
model_1 = keras.models.load_model("keras_model.h5")

# Function to preprocess the stock data
def preprocess_data(data):
    # Normalize the data
   
    
    return data

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
        
        # Create a dataframe with actual and predicted prices
        # Create a dataframe with actual and predicted prices
        dates = df.index[9:]  # Exclude initial dates with no predictions
        actual_predicted_df = pd.DataFrame({'Date': dates, 'Actual': stock_price_data[9:], 'Predicted': predicted_price.flatten()})
        actual_predicted_df['Predicted'] = np.nan  # Set all predicted values as NaN initially
        actual_predicted_df.iloc[-predicted_price.shape[0]:, -1] = predicted_price.flatten()  # Set the predicted values at the corresponding indices

        
        # Plot the actual and predicted prices
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(actual_predicted_df['Date'], actual_predicted_df['Actual'], label='Actual')
        ax.plot(actual_predicted_df['Date'], actual_predicted_df['Predicted'], label='Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f"Stock Price Prediction for {stock_ticker}")
        ax.legend()
        st.pyplot(fig)
        
        # Display the actual and predicted prices
        st.write(actual_predicted_df)
    else:
        st.write("No data available for the given stock ticker.")
    
# Run the Streamlit app
if __name__ == '__main__':
    main()
