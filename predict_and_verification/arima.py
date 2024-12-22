import os
import json
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
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


def smooth_and_interpolate_prices(price_dir, item_name):
    """
    Smooth and interpolate the price data for a single item.
    """
    price_file = os.path.join(price_dir, f"{item_name}.csv")
    if not os.path.exists(price_file):
        print(f"{price_file} not found, skipping.")
        return None

    df = pd.read_csv(price_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).set_index('date').sort_index()

    if df.empty:
        print(f"No data for {item_name}, skipping.")
        return None

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range)
    df['price'] = df['price'].interpolate(method='linear', limit_direction='both')
    df.index.name = "date"

    smooth_path = os.path.join("item_smooth_prices", f"{item_name}.csv")
    df.to_csv(smooth_path, index=True, date_format='%Y-%m-%d')
    print(f"✅ Smoothed price data saved: {smooth_path}")
    return smooth_path


def forecast_prices(df_data, smooth_item_path, item_name, forecast_steps=7):
    """
    Perform ARIMA forecasting for a single item and save results.
    """
    item_df = pd.read_csv(smooth_item_path, parse_dates=["date"], index_col="date")

    if df_data.empty or item_df.empty:
        print(f"No overlapping data for {item_name}, skipping forecast.")
        return

    common_start = max(df_data.index.min(), item_df.index.min())
    common_end = min(df_data.index.max(), item_df.index.max())

    if common_start > common_end:
        print(f"No overlapping range for {item_name}.")
        return

    df_data_item = df_data.loc[common_start:common_end]
    item_df = item_df.loc[common_start:common_end]

    if item_df['price'].isna().all():
        print(f"All prices are NaN for {item_name}, skipping.")
        return

    y = item_df['price'].fillna(method='ffill')
    model = sm.tsa.ARIMA(y, order=(1, 1, 1))
    result = model.fit()
    print(result.summary())

    forecast = result.forecast(steps=forecast_steps)
    forecast_df = pd.DataFrame({
        "date": pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=forecast_steps),
        "forecast_price": forecast
    })
    forecast_df.set_index('date', inplace=True)

    future_out_path = os.path.join("consequence/arima", f"future_{item_name}.csv")
    forecast_df.to_csv(future_out_path, date_format='%Y-%m-%d')
    print(f"✅ Future forecast saved: {future_out_path}")

    complete_df = pd.concat([y, forecast_df['forecast_price']], axis=1)
    complete_df.rename(columns={0: "forecast_price"}, inplace=True)
    complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
    complete_out_path = os.path.join("consequence/arima", f"complete_{item_name}.csv")
    complete_df.to_csv(complete_out_path, date_format='%Y-%m-%d')
    print(f"✅ Complete data saved: {complete_out_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(y.index, y, label='Historical Price', color='blue')
    plt.plot(forecast_df.index, forecast_df['forecast_price'], label='Forecast Price', color='red')
    plt.title(f"Price Forecast for {item_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    picture_path = os.path.join("consequence/arima/picture", f"{item_name}_forecast_plot.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ Plot saved: {picture_path}")


def main():
    parser = argparse.ArgumentParser(description="ARIMA Price Forecasting")
    parser.add_argument("--item", type=str, required=True, help="Item name, or path to a .txt/.json file containing item names.")
    args = parser.parse_args()

    items = parse_item_list(args.item)

    price_dir = "item/prices"
    data_complete_file = "data_complete.json"
    records = []

    os.makedirs("item_smooth_prices", exist_ok=True)
    os.makedirs("consequence/arima", exist_ok=True)
    os.makedirs("consequence/arima/picture", exist_ok=True)

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

    for item_name in items:
        smooth_item_path = smooth_and_interpolate_prices(price_dir, item_name)
        if smooth_item_path:
            forecast_prices(df_data, smooth_item_path, item_name)


if __name__ == "__main__":
    main()
