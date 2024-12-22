import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
    parser = argparse.ArgumentParser(description="LSTM Price Forecasting")
    parser.add_argument("--item", type=str, required=True, help="Item name, or path to a .txt/.json file containing item names.")
    args = parser.parse_args()

    items = parse_item_list(args.item)

    price_dir = "item_smooth_prices"
    data_complete_file = "data_complete.json"

    os.makedirs("consequence/lstm", exist_ok=True)
    os.makedirs("consequence/lstm/picture", exist_ok=True)

    # Load external data
    records = []
    with open(data_complete_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            record_date = datetime.strptime(record["date"], "%Y-%m-%d")
            peak_players = record.get("peak_players", 0)
            trends = record.get("trends", {})
            csgo_trend = trends.get("CS:GO", 0)
            records.append([record_date, peak_players, csgo_trend])

    df_data = pd.DataFrame(records, columns=["date", "peak_players", "csgo_trend"])
    df_data = df_data.set_index("date").sort_index()

    forecast_steps = 7
    window_size = 30

    for item_name in items:
        item_path = os.path.join(price_dir, f"{item_name}.csv")
        if not os.path.exists(item_path):
            print(f"{item_path} not found, skipping {item_name}")
            continue

        item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")

        if df_data.empty or item_df.empty:
            print(f"No overlapping data for {item_name}, skipping LSTM forecast.")
            continue

        common_start = max(df_data.index.min(), item_df.index.min())
        common_end = min(df_data.index.max(), item_df.index.max())

        if common_start > common_end:
            print(f"No overlapping date range for {item_name}.")
            continue

        item_df = item_df.loc[common_start:common_end]

        if item_df.empty:
            print(f"No data after intersection for {item_name}, skipping.")
            continue

        series = item_df['price'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_series = scaler.fit_transform(series)

        X, Y = create_dataset(scaled_series, window_size)
        if len(X) == 0:
            print(f"Not enough data to create a dataset for {item_name}, skipping.")
            continue

        model = Sequential()
        model.add(LSTM(50, input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X, Y, epochs=10, batch_size=32, verbose=1)

        last_window = scaled_series[-window_size:]
        current_input = last_window.copy().reshape(1, window_size, 1)
        future_preds = []

        for _ in range(forecast_steps):
            pred = model.predict(current_input, verbose=0)
            future_preds.append(pred[0][0])
            current_input = np.append(current_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

        future_preds = np.array(future_preds).reshape(-1, 1)
        future_preds_inv = scaler.inverse_transform(future_preds)

        future_dates = pd.date_range(start=item_df.index.max() + pd.Timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame(future_preds_inv, index=future_dates, columns=["forecast_price"])

        future_path = os.path.join("consequence/lstm", f"future_lstm_{item_name}.csv")
        forecast_df.to_csv(future_path, date_format='%Y-%m-%d')
        print(f"✅ LSTM future forecast saved: {future_path}")

        complete_df = pd.concat([item_df[['price']], forecast_df], axis=1)
        complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
        complete_path = os.path.join("consequence/lstm", f"complete_lstm_{item_name}.csv")
        complete_df.to_csv(complete_path, date_format='%Y-%m-%d')
        print(f"✅ LSTM complete sequence saved: {complete_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(item_df.index, item_df['price'], label='Historical Price', color='blue')
        plt.plot(forecast_df.index, forecast_df['forecast_price'], label='LSTM Forecast', color='green')
        plt.title(f"LSTM Price Forecast for {item_name}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        picture_dir = "consequence/lstm/picture"
        os.makedirs(picture_dir, exist_ok=True)
        picture_path = os.path.join(picture_dir, f"{item_name}_lstm_forecast_plot.png")
        plt.savefig(picture_path, dpi=100)
        plt.close()
        print(f"✅ LSTM forecast plot saved: {picture_path}")


if __name__ == "__main__":
    main()
