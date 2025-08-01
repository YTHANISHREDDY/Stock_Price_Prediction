# import streamlit as st
# import pandas as pd
# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import yfinance as yf

# st.title("Stock Price Predictor App")

# stock = st.text_input("Enter the Stock ID")

# from datetime import datetime
# end = datetime.now()
# start = datetime(end.year-20,end.month,end.day)

# Stock_data = yf.download(stock, start, end)

# model = load_model("Latest_stock_price_model.keras")
# st.subheader("Stock Data")
# st.write(Stock_data)

# splitting_len = int(len(Stock_data)*0.7)
# x_test = pd.DataFrame(Stock_data.Close[splitting_len:])

# def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(values,'Orange')
#     plt.plot(full_data.Close, 'b')
#     if extra_data:
#         plt.plot(extra_dataset)
#     return fig

# st.subheader('Original Close Price and MA for 250 days')
# Stock_data['MA_for_250_days'] = Stock_data.Close.rolling(250).mean()
# st.pyplot(plot_graph((15,6), Stock_data['MA_for_250_days'],Stock_data,0))

# st.subheader('Original Close Price and MA for 200 days')
# Stock_data['MA_for_200_days'] = Stock_data.Close.rolling(200).mean()
# st.pyplot(plot_graph((15,6), Stock_data['MA_for_200_days'],Stock_data,0))

# st.subheader('Original Close Price and MA for 100 days')
# Stock_data['MA_for_100_days'] = Stock_data.Close.rolling(100).mean()
# st.pyplot(plot_graph((15,6), Stock_data['MA_for_100_days'],Stock_data,0))

# st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
# st.pyplot(plot_graph((15,6), Stock_data['MA_for_100_days'],Stock_data,1,Stock_data['MA_for_250_days']))

# close_price = Stock_data["Close"]["MSFT"]

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(close_price.values.reshape(-1, 1))

# x_data = []
# y_data = []

# for i in range(100, len(scaled_data)):
#     x_data.append(scaled_data[i-100:i])
#     y_data.append(scaled_data[i])
    
# x_data, y_data = np.array(x_data), np.array(y_data)

# predictions = model.predict(x_data)

# inv_pre = scaler.inverse_transform(predictions)
# inv_y_test = scaler.inverse_transform(y_data)

# ploting_data = pd.DataFrame(
#  {
#   'original_test_data': inv_y_test.reshape(-1),
#     'predictions': inv_pre.reshape(-1)
#  } ,
#     index = Stock_data.index[splitting_len+100:]
# )
# st.subheader("Original values vs Predicted values")
# st.write(ploting_data)

# st.subheader('Original Close Price vs Predicted Close price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(pd.concat([Stock_data.Close[:splitting_len+100],ploting_data], axis=0))
# plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
# st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import yfinance as yf
# from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler

# # --- App Title ---
# st.title("Stock Price Predictor App")

# # --- User Input ---
# stock = st.text_input("Enter the Stock ID")

# # --- Fetch Stock Data ---
# if stock:
#     end = datetime.now()
#     start = datetime(end.year - 20, end.month, end.day)

#     data = yf.download(stock, start=start, end=end)

#     if data.empty:
#         st.error("No data found for this stock. Please check the ticker symbol.")
#     else:
#         st.subheader("Stock Data")
#         st.write(data)

#         # --- Moving Averages ---
#         def plot_graph(figsize, values, full_data, extra_data=False, extra_dataset=None):
#             fig = plt.figure(figsize=figsize)
#             plt.plot(values, 'orange')
#             plt.plot(full_data['Close'], 'blue')
#             if extra_data:
#                 plt.plot(extra_dataset, 'green')
#             return fig

#         data['MA_100'] = data['Close'].rolling(100).mean()
#         data['MA_200'] = data['Close'].rolling(200).mean()
#         data['MA_250'] = data['Close'].rolling(250).mean()

#         st.subheader("Close Price with 250-day MA")
#         st.pyplot(plot_graph((15, 6), data['MA_250'], data))

#         st.subheader("Close Price with 200-day MA")
#         st.pyplot(plot_graph((15, 6), data['MA_200'], data))

#         st.subheader("Close Price with 100-day MA")
#         st.pyplot(plot_graph((15, 6), data['MA_100'], data))

#         st.subheader("Close Price with 100-day and 250-day MA")
#         st.pyplot(plot_graph((15, 6), data['MA_100'], data, extra_data=True, extra_dataset=data['MA_250']))

#         # --- Load Model ---
#         model = load_model("Latest_stock_price_model.keras")

#         # --- Preprocessing ---
#         close_price = data["Close"]
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_data = scaler.fit_transform(close_price.values.reshape(-1, 1))

#         x_data = []
#         y_data = []

#         for i in range(100, len(scaled_data)):
#             x_data.append(scaled_data[i - 100:i])
#             y_data.append(scaled_data[i])

#         x_data, y_data = np.array(x_data), np.array(y_data)

#         # --- Split into training/testing ---
#         split_idx = int(len(x_data) * 0.7)
#         x_test = x_data[split_idx:]
#         y_test = y_data[split_idx:]
#         test_index = data.index[split_idx + 100:]

#         # --- Predictions ---
#         predictions = model.predict(x_test)
#         inv_predictions = scaler.inverse_transform(predictions)
#         inv_y_test = scaler.inverse_transform(y_test)

#         # --- Plot Predictions ---
#         plot_df = pd.DataFrame({
#             'Original': inv_y_test.flatten(),
#             'Predicted': inv_predictions.flatten()
#         }, index=test_index)

#         st.subheader("Original vs Predicted Close Prices (Test Set)")
#         st.line_chart(plot_df)

#         # --- Overlay Plot ---
#         st.subheader("Overlay: Full Close Price vs Predictions")
#         fig = plt.figure(figsize=(15, 6))
#         plt.plot(data['Close'][:split_idx + 100], label="Training + Unused")
#         plt.plot(test_index, inv_y_test.flatten(), label="Original Test", color='blue')
#         plt.plot(test_index, inv_predictions.flatten(), label="Predicted Test", color='orange')
#         plt.legend()
#         st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Title and user input
st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID").strip().upper()

if not stock:
    st.warning("Please enter a stock ticker symbol to begin.")
    st.stop()

# Download data
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

# Load model
try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Model file not found or couldn't be loaded: {e}")
    st.stop()

# Show data
st.subheader("Stock Data")
st.write(Stock_data)

# Plotting helper
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving averages
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

# Close price and scaling
close_price = Stock_data["Close"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_price.values.reshape(-1, 1))

# Data preparation
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
with st.spinner("Predicting stock prices..."):
    predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Align index
num_samples = len(inv_y_test)
plot_index = Stock_data.index[-num_samples:]

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=plot_index
)

# Show prediction results
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(Stock_data.Close[:len(Stock_data.Close) - num_samples], label='Data not used')
plt.plot(ploting_data['original_test_data'], label='Original Test Data')
plt.plot(ploting_data['predictions'], label='Predicted Test Data')
plt.legend()
st.pyplot(fig)
