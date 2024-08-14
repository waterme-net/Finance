import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Define the company ticker symbol and the date range
company = "INFY.NS"
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# Load the data
data = yf.download(company, start, end)

# Prepare the data for the neural network
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

prediction_days = 60
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential()

model.add(LSTM(units=70, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=70))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Load the test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, test_start, test_end)
actual_prices = test_data["Close"].values

total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Predict the next 7 days
predictions = []
current_input = model_inputs[-prediction_days:].tolist()

for _ in range(7):
    current_input_array = np.array(current_input[-prediction_days:]).reshape((1, prediction_days, 1))
    predicted_price = model.predict(current_input_array)
    predictions.append(predicted_price[0, 0])
    current_input.append([predicted_price[0, 0]])

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Print the next 7 days predictions
for i, prediction in enumerate(predictions, 1):
    print(f"Day {i} prediction: {prediction[0]}")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")

# Plot the future predictions
future_dates = pd.date_range(start=test_end, periods=7)
plt.plot(range(len(actual_prices), len(actual_prices) + len(predictions)), predictions, color='blue', label='Future Predictions')

plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()
