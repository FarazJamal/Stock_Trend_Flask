import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

files = {
    "S&P": "Processed_S&P.csv",
    "RUSSELL": "Processed_RUSSELL.csv",
    "NYSE": "Processed_NYSE.csv",
    "NASDAQ": "Processed_NASDAQ.csv",
    "DJI": "Processed_DJI.csv"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_file = request.form['file_button']
    else:
        selected_file = "S&P"  # Default selection

    df = pd.read_csv(files[selected_file])

    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    model = load_model('LSTM_keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    X_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        X_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    y_pred = model.predict(X_test)

    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Plotting
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y_test, 'b', label='Original Price')
    ax.plot(y_pred, 'r', label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()

    # Save the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Convert the plot to a base64-encoded string
    plot_image = base64.b64encode(image_stream.read()).decode('utf-8')

    # Close the figure to release resources
    plt.close(fig2)

    return render_template('lstm.html', rmse=rmse, mae=mae, plot_image=plot_image, selected_file=selected_file, files=files)

if __name__ == '__main__':
    app.run(debug=True)
