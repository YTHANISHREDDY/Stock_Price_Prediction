import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID").strip().upper()

if not stock:
    st.warning("Please enter a stock ticker symbol to begin.")
    st.stop()

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

try:
    Stock_data = yf.download(stock, start=start, end=end)
    if Stock_data.empty:
        st.error(f"No data found for stock symbol: {stock}")
        st.stop()
except Exception as e:
    st.error(f"Failed to download stock data: {e}")
    st.stop()

try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Model file not found or couldn't be loaded: {e}")
    st.stop()

st.subheader("Stock Data")
st.write(Stock_data)

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
Stock_data['MA_for_250_days'] = Stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), Stock_data['MA_for_250_days'], Stock_data))

st.subheader('Original Close Price and MA for 200 days')
Stock_data['MA_for_200_days'] = Stock_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), Stock_data['MA_for_200_days'], Stock_data))

st.subheader('Original Close Price and MA for 100 days')
Stock_data['MA_for_100_days'] = Stock_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), Stock_data['MA_for_100_days'], Stock_data))

st.subheader('Original Close Price with MA for 100 and 250 days')
st.pyplot(plot_graph((15, 6), Stock_data['MA_for_100_days'], Stock_data, 1, Stock_data['MA_for_250_days']))

close_price = Stock_data["Close"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_price.values.reshape(-1, 1))

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

with st.spinner("Predicting stock prices..."):
    predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

num_samples = len(inv_y_test)
plot_index = Stock_data.index[-num_samples:]

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=plot_index
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(Stock_data.Close[:len(Stock_data.Close) - num_samples], label='Data not used')
plt.plot(ploting_data['original_test_data'], label='Original Test Data')
plt.plot(ploting_data['predictions'], label='Predicted Test Data')
plt.legend()
st.pyplot(fig)

