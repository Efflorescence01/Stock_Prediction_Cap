
# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt 

API_KEY = 'FYHS11VFOEALEUF3' # Replace it with real value


new_model = tf.keras.models.load_model('C:/Users/linwi/Documents/Stock_test/Stock Modeling/lstm-timeseries.h5')
new_data = pd.read_csv('C:/Users/linwi/Documents/Stock_test/Cleaned_Dataset/GOOG.csv',index_col = 0)
mm_scaler = MinMaxScaler(feature_range=(0,1))

def preprocessing (data):
	data.index = pd.to_datetime(data.index)
	del data['close']
	data.dropna(inplace = True)
	data.sort_index(inplace = True, ascending = True)
	data1 = data.filter(['adjusted close']).values
	dataset = data1.astype('float32')
	scaled_dataset = mm_scaler.fit_transform(dataset)

	return dataset, scaled_dataset

def train_test_splits(scaled_dataset):
	sequence_length = 60

	train_data = scaled_dataset[:int(len(scaled_dataset)*0.8), :]
	x_train = []
	y_train = []

	for i in range(sequence_length, len(train_data)):
		x_train.append(train_data[i-sequence_length:i, 0])
		y_train.append(train_data[i, 0])

	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	test_data = scaled_dataset[int(len(scaled_dataset)*0.8)-sequence_length:, :]
	x_test = []
	y_test = scaled_dataset[int(len(scaled_dataset)*0.8):, :]

	for i in range(sequence_length, len(test_data)):
		x_test.append(test_data[i-sequence_length:i, 0])

	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return x_train, y_train, x_test, y_test

# def main():
# 	st.write('# Alpha Vantage stock price data')
# 	# Ask user for stock symbol
# 	symbol = st.text_input('Enter stock symbol:', 'GOOG').upper()

# 	# API Endpoint to retrieve Daily Time Series
# 	url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"


# 	# Request the data, parse JSON response and store it in Python variable
# 	r = requests.get(url, timeout=5)
# 	data = r.json()

# 	# Extract basic information from collected data
# 	information = data['Meta Data']['1. Information']
# 	symbol = data['Meta Data']['2. Symbol']
# 	last_refreshed = data['Meta Data']['3. Last Refreshed']

# 	# Display the collected data to user using Streamline functions
# 	st.write('## ' + information)
# 	st.write('### ' + symbol)
# 	st.write('### Last update: ' + last_refreshed)

# 	st.write('## Time Series (Daily)')

# 	# Use Pandas' Data Frame to prepare data to be displayed in charts
# 	df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

# 	df = df.reset_index()
# 	df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# 	df['open'] = df['open'].astype(float)
# 	df['high'] = df['high'].astype(float)
# 	df['low'] = df['low'].astype(float)
# 	df['close'] = df['close'].astype(float)
# 	df['volume'] = df['volume'].astype(int)

# 	df['date'] = pd.to_datetime(df['date'])
# 	df = df.sort_values(by='date')

# 	# Display Streamline charts
# 	st.write("Open")
# 	st.line_chart(df.set_index('date')['open'],color = "#9100cd")
# 	st.write("High")
# 	st.line_chart(df.set_index('date')['high'],color = "#e600cd")
# 	st.write("Low")
# 	st.line_chart(df.set_index('date')['low'],color = "#5100cd")
# 	st.write("Close")
# 	st.line_chart(df.set_index('date')['close'],color = "#017d00")
# 	st.write("Daily Volume Chart")
# 	st.bar_chart(df.set_index('date')['volume'])

# 	scaled_dataset = preprocessing(new_data)

# 	x_train, y_train, x_test, y_test = train_test_splits(scaled_dataset)

# 	# Get the model's predicted price values
# 	# from sklearn.preprocessing import MinMaxScaler
# 	# mm_scaler = MinMaxScaler(feature_range=(0,1))

# 	predictions = new_model.predict(x_test)
# 	#Transforming them back to their original price values
# 	predictions = mm_scaler.inverse_transform(predictions)

	# # Plot the data
	# train = new_data[:int(len(dataset)*0.8)]
	# valid = new_data[int(len(dataset)*0.8):]
	# valid['Predictions'] = predictions

	# plt.figure(figsize=(16,8))
	# plt.title('LSTM Model')
	# plt.xlabel('Date', fontsize=18)
	# plt.ylabel('Close Price USD ($)', fontsize=18)
	# plt.plot(train['adjusted close'])
	# plt.plot(valid[['adjusted close', 'Predictions']])
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	# plt.show()

def main():
    st.write('# Alpha Vantage stock price data')
    # Ask user for stock symbol
    symbol = st.text_input('Enter stock symbol:', 'GOOG').upper()

    # API Endpoint to retrieve Daily Time Series
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"

    # Request the data, parse JSON response and store it in Python variable
    r = requests.get(url, timeout=5)
    data = r.json()

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
    st.line_chart(df.set_index('date')['open'], color="#9100cd")
    st.write("High")
    st.line_chart(df.set_index('date')['high'], color="#e600cd")
    st.write("Low")
    st.line_chart(df.set_index('date')['low'], color="#5100cd")
    st.write("Close")
    st.line_chart(df.set_index('date')['close'], color="#017d00")
    st.write("Daily Volume Chart")
    st.bar_chart(df.set_index('date')['volume'])

    dataset, scaled_dataset = preprocessing(new_data)

    x_train, y_train, x_test, y_test = train_test_splits(scaled_dataset)

    # Get the model's predicted price values
    predictions = new_model.predict(x_test)
    # Transforming them back to their original price values
    predictions = mm_scaler.inverse_transform(predictions)


   # Plot the data
	

	fig = plt.figure(figsize=(16,8))
	plt.title('LSTM Model')
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD ($)', fontsize=18)
	plt.plot(y_train['adjusted close'])
	plt.plot(y_test['adjusted close'])
	plt.plot(predictions)
	plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	plt.show()

	st.pyplot(fig)




if __name__ == '__main__':
	main()