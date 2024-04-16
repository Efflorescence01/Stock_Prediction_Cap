
# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.write('# Alpha Vantage stock price data')

API_KEY = 'FYHS11VFOEALEUF3' # Replace it with real value

# Ask user for stock symbol
symbol = st.text_input('Enter stock symbol:', 'GOOG').upper()

# API Endpoint to retrieve Daily Time Series
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"


# Request the data, parse JSON response and store it in Python variable
r = requests.get(url, timeout=5)
data = r.json()

#function to preprocess data
# def preprocessing(data):
#     """
#     Function to preprocess the json file obtained from the AlphaVantage API. This function will turn the json file into a dataframe and preprocess the data.
#     """
#     data_1 = pd.DataFrame(data['Time Series (Daily)']).T.copy()
#     data_1.columns = ['open', 'high', 'low', 'close', 'adjusted close', 'volume','dividend amount','split coefficient']
#     del data_1['dividend amount']
#     del data_1['split coefficient']
#     data_1.index = pd.to_datetime(data_1.index)

#     for column in data_1.columns:
#         data_1[column] = pd.to_numeric(data_1[column])

#     return data_1

# preprocessing(data)



# Extract basic information from collected data
information = data['Meta Data']['1. Information']
symbol = data['Meta Data']['2. Symbol']
last_refreshed = data['Meta Data']['3. Last Refreshed']

# Display the collected data to user using Streamline functions
st.write('## ' + information)
st.write('### ' + symbol)
st.write('### Last update: ' + last_refreshed)

st.write('## Time Series (Daily)')

# Use Pandas' Data Frame to prepare data to be displayed in charts
df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

df = df.reset_index()
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(int)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Display Streamline charts
st.write("Open")
st.line_chart(df.set_index('date')['open'],color = "#9100cd")
st.write("High")
st.line_chart(df.set_index('date')['high'],color = "#e600cd")
st.write("Low")
st.line_chart(df.set_index('date')['low'],color = "#5100cd")
st.write("Close")
st.line_chart(df.set_index('date')['close'],color = "#017d00")
st.write("Daily Volume Chart")
st.bar_chart(df.set_index('date')['volume'])