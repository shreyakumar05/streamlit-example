import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

# Describing here
st.subheader('Data from 2010-2019')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting the Data into Testing and Training
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Load my models
model = load_model('keras_model.h5')
cnn_model = load_model('cnn_model.h5')

# Testing Part
sequence_length = 4

past_100_days = data_training.tail(sequence_length)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
x_test_cnn = []
y_test = []

for i in range(sequence_length, input_data.shape[0]):
    x_test.append(input_data[i - sequence_length: i])
    x_test_cnn.append(input_data[i - sequence_length: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test_cnn = np.array(x_test_cnn)
x_test_cnn = np.reshape(x_test_cnn, (x_test_cnn.shape[0], x_test_cnn.shape[1], 1))

y_predicted = model.predict(x_test)
y_predicted_cnn = cnn_model.predict(x_test_cnn)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
y_predicted_cnn = y_predicted_cnn * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price (Keras Model)')
plt.plot(y_predicted_cnn, 'g', label='Predicted Price (CNN Model)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
