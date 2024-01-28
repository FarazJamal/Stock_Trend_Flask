from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# List of file names
file_names = ['Processed_DJI.csv', 'Processed_NASDAQ.csv', 'Processed_NYSE.csv', 'Processed_S&P.csv', 'Processed_RUSSELL.csv']

# Create an empty dataframe to concatenate all dataframes
all_data = pd.DataFrame()

# Loop through each file name
for file_name in file_names:
    df = pd.read_csv(file_name)
    
    # Assuming 'Close' column exists in each CSV file
    data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    
    # Concatenate dataframes
    all_data = pd.concat([all_data, data])

scaler = MinMaxScaler(feature_range=(0,1))

data_array = scaler.fit_transform(all_data)

X_train = []
y_train = []

for i in range(100, data_array.shape[0]):
    X_train.append(data_array[i-100: i])
    y_train.append(data_array[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50)

model.save('LSTM_keras_model.h5')
