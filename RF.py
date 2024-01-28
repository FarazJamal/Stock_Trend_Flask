import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# File paths for the CSV files
files = {
    "S&P": "Processed_S&P.csv",
    "RUSSELL": "Processed_RUSSELL.csv",
    "NYSE": "Processed_NYSE.csv",
    "NASDAQ": "Processed_NASDAQ.csv",
    "DJI": "Processed_DJI.csv"
}

def process_dataset(file):
    try:
        df = pd.read_csv(file)

        # Data Preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.fillna(method='ffill', inplace=True)

        # Feature Engineering (using 'Close' to create a moving average)
        df['MA_Close'] = df['Close'].rolling(window=5).mean().shift(1)  # 5 days moving average

        # Feature and Target Selection
        features = ['Close', 'MA_Close']
        df.dropna(inplace=True)
        X = df[features]
        y = df['Close'].shift(-1)  # Predicting the next day's closing price
        y = y[:-1]
        X = X[:-1]

        return X, y
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None



def train_and_evaluate(X, y):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ## Model Development
    # RandomForest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # GradientBoosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    # Model Evaluation
    results = []
    predictions_rf = rf_model.predict(X_test_scaled)
    predictions_gb = gb_model.predict(X_test_scaled)

    for model, name, predictions in zip([rf_model, gb_model], ['RandomForest', 'GradientBoosting'], [predictions_rf, predictions_gb]):
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        results.append((name, np.sqrt(mse), mae, predictions))

    return y_test, results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_dataset = request.form['dataset_button']
    else:
        selected_dataset = "S&P"  # Default selection

    X, y = process_dataset(files[selected_dataset])

    # Training and Evaluation
    y_test, model_results = train_and_evaluate(X, y)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Original Prices', color='blue')
    for result in model_results:
        ax.plot(result[3], label=f'{result[0]} Predicted Prices')
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
    plt.close(fig)

    return render_template('rf.html', selected_dataset=selected_dataset, model_results=model_results, y_test=y_test.values, plot_image=plot_image, files=files)

if __name__ == '__main__':
    app.run(debug=True)
