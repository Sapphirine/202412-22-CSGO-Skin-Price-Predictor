import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import argparse


def parse_item_list(item_arg):
    """
    Parse item list from a single item name, a .txt file, or a .json file.
    """
    if os.path.isfile(item_arg):
        _, ext = os.path.splitext(item_arg)
        if ext == ".txt":
            with open(item_arg, "r", encoding="utf-8") as f:
                items = [line.strip() for line in f if line.strip()]
        elif ext == ".json":
            with open(item_arg, "r", encoding="utf-8") as f:
                items = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .txt or .json.")
    else:
        items = [item_arg]
    return items


def create_dataset(series, window_size=30):
    """
    Create supervised learning dataset for LSTM.
    """
    X, Y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        Y.append(series[i + window_size])
    return np.array(X), np.array(Y)


def main():
    # Add argparse support
    parser = argparse.ArgumentParser(description="LSTM Verification")
    parser.add_argument("--item", type=str, required=True, help="Item name or path to a .txt/.json file containing item names.")
    args = parser.parse_args()

    # Parse item list
    items = parse_item_list(args.item)

    # Directories
    price_dir = "item_smooth_prices"
    output_dir = "consequence/verification/lstm"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/picture", exist_ok=True)
    os.makedirs(f"{output_dir}/comparison_curves", exist_ok=True)

    # Dates
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2024, 12, 31)
    plot_start = datetime(2024, 11, 1)
    window_size = 30

    for item_name in items:
        item_path = os.path.join(price_dir, f"{item_name}.csv")
        if not os.path.exists(item_path):
            print(f"{item_path} not found, skipping {item_name}")
            continue

        item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
        if item_df.empty:
            print(f"No data for {item_name}, skipping.")
            continue

        # Train-test split
        train_df = item_df[item_df.index < test_start]
        test_df = item_df[(item_df.index >= test_start) & (item_df.index <= test_end)]

        if len(train_df) < window_size or test_df.empty:
            print(f"Not enough train or test data for {item_name}, skipping.")
            continue

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_series = scaler.fit_transform(train_df['price'].values.reshape(-1, 1))

        X_train, Y_train = create_dataset(train_series, window_size)
        if len(X_train) == 0:
            print(f"Not enough training data after windowing for {item_name}, skipping.")
            continue

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train model
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

        # Rolling forecast for test set
        full_series = np.concatenate([train_series, scaler.transform(test_df['price'].values.reshape(-1, 1))], axis=0)
        test_start_idx = len(train_series)
        real_values = test_df['price'].values
        test_dates = test_df.index

        # Use last `window_size` days from train_series as the initial window
        cond = full_series[test_start_idx - window_size:test_start_idx].copy().reshape(1, window_size, 1)

        predictions = []
        for i in range(len(test_df)):
            pred = model.predict(cond, verbose=0)
            predictions.append(pred[0][0])
            cond = np.append(cond[:, 1:, :], [[[pred[0][0]]]], axis=1)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions_inv = scaler.inverse_transform(predictions)

        # Metrics
        mse = mean_squared_error(real_values, predictions_inv[:, 0])
        mae = mean_absolute_error(real_values, predictions_inv[:, 0])
        rmse = np.sqrt(mse)

        print(f"{item_name} - December LSTM forecast metrics:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Save comparison results
        result_df = pd.DataFrame({
            "date": test_dates,
            "real_price": real_values,
            "predicted_price": predictions_inv[:, 0]
        })
        result_out_path = os.path.join(output_dir, f"compare_{item_name}.csv")
        result_df.to_csv(result_out_path, index=False, date_format='%Y-%m-%d')
        print(f"✅ LSTM comparison saved: {result_out_path}")

        text_str = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"

        # Plot 1: December range
        plt.figure(figsize=(10, 5))
        plt.plot(test_dates, real_values, label='Real Price', color='blue')
        plt.plot(test_dates, predictions_inv[:, 0], label='Predicted Price', color='red')
        plt.title(f"{item_name} December LSTM Forecast vs Real")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.text(test_dates.min(), max(real_values.max(), predictions_inv[:, 0].max()) * 0.9, text_str,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        picture_path = os.path.join(f"{output_dir}/picture", f"{item_name}_december_comparison.png")
        plt.savefig(picture_path, dpi=100)
        plt.close()
        print(f"✅ December comparison plot saved: {picture_path}")

        # Plot 2: November to December range
        plt.figure(figsize=(10, 5))
        extended_plot_df = item_df.loc[(item_df.index >= plot_start) & (item_df.index <= test_end)]

        plt.plot(extended_plot_df.index, extended_plot_df['price'], label='Historical Price', color='blue')
        plt.plot(test_dates, predictions_inv[:, 0], label='Predicted Price', color='red')
        plt.title(f"{item_name} November to December LSTM Forecast vs Real")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.text(plot_start, max(extended_plot_df['price'].max(), predictions_inv[:, 0].max()) * 0.9, text_str,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        picture_path_nov = os.path.join(f"{output_dir}/picture", f"{item_name}_november_to_december_comparison.png")
        plt.savefig(picture_path_nov, dpi=100)
        plt.close()
        print(f"✅ November to December comparison plot saved: {picture_path_nov}")

        # Save metrics
        metrics_path = os.path.join(f"{output_dir}/comparison_curves", f"{item_name}_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as mf:
            mf.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")
        print(f"✅ Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
