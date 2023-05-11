\import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the trained model
model = load_model('keras_model.h5')

# Function to preprocess the data
def preprocess_data(df):
    df = df.reset_index()
    df = df.drop(['Date', 'Adj Close'], axis=1)
    return df

# Function to prepare input data for the LSTM model
def prepare_input_data(data, scaler):
    data_array = scaler.transform(data)
    x_data = []
    y_data = []
    for i in range(100, data_array.shape[0]):
        x_data.append(data_array[i-100: i])
        y_data.append(data_array[i, 0])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

# Function to plot the stock prices
def plot_stock_prices(original, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(original, 'b', label='Original Price')
    plt.plot(predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Stock Price Prediction')
    st.pyplot()

# Streamlit app
def main():
    st.title('Stock Price Prediction App')

    # Stock ticker input from user
    stock_ticker = st.text_input('Enter the stock ticker (e.g., AAPL):')

    if stock_ticker:
        start = '2010-01-01'
        end = '2019-12-31'

        # Download stock data
        df = yf.download(stock_ticker, start, end)

        if not df.empty:
            # Preprocess data
            df_processed = preprocess_data(df)

            # Split data into training and testing
            data_training = pd.DataFrame(df_processed['Close'][0:int(len(df_processed)*0.70)])
            data_testing = pd.DataFrame(df_processed['Close'][int(len(df_processed)*0.70): int(len(df_processed))])

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)
            input_data = scaler.transform(pd.concat([data_training, data_testing], ignore_index=True))

            # Prepare input data for the model
            x_test, y_test = prepare_input_data(input_data, scaler)

            # Predict stock prices
            y_predicted = model.predict(x_test)
            scale_factor = 1 / 0.02099517
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Display stock prices and plot
            st.subheader('Predicted Stock Prices:')
            st.write(y_predicted)

            st.subheader('Stock Price Chart:')
            plot_stock_prices(y_test, y_predicted)
        else:
            st.write('Error: Invalid stock ticker or no data available.')

if __name__ == '__main__':
    main()
