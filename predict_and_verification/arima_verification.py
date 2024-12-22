import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
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


def main():
    # Add argparse support
    parser = argparse.ArgumentParser(description="ARIMA Verification")
    parser.add_argument("--item", type=str, required=True, help="Item name or path to a .txt/.json file containing item names.")
    args = parser.parse_args()

    # Parse item list
    items = parse_item_list(args.item)

    # Directories
    price_dir = "item_smooth_prices"
    output_dir = "consequence/verification/arima"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/picture", exist_ok=True)
    os.makedirs(f"{output_dir}/comparison_curves", exist_ok=True)

    # Dates
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2024, 12, 31)
    plot_start = datetime(2024, 11, 1)

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

        if test_df.empty or len(train_df) < 10:
            print(f"Not enough train or test data for {item_name}, skipping.")
            continue

        # Fit ARIMA model
        model = ARIMA(train_df['price'], order=(1, 1, 1))
        result = model.fit()

        real_values = test_df['price'].values
        test_dates = test_df.index

        # Rolling forecast
        current_series = train_df['price'].copy()
        predictions = []
        for dt in test_dates:
            fc = result.get_forecast(steps=1)
            pred_val = fc.predicted_mean.iloc[0]
            predictions.append(pred_val)
            current_series.loc[dt] = test_df.loc[dt, 'price']
            model = ARIMA(current_series, order=(1, 1, 1))
            result = model.fit()

        predictions = np.array(predictions)

        # Metrics
        mse = mean_squared_error(real_values, predictions)
        mae = mean_absolute_error(real_values, predictions)
        rmse = np.sqrt(mse)

        print(f"{item_name} - December ARIMA forecast metrics:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Save comparison results
        result_df = pd.DataFrame({
            "date": test_dates,
            "real_price": real_values,
            "predicted_price": predictions
        })
        result_out_path = os.path.join(output_dir, f"compare_{item_name}.csv")
        result_df.to_csv(result_out_path, index=False, date_format='%Y-%m-%d')
        print(f"✅ ARIMA comparison saved: {result_out_path}")

        # Plot 1: December range
        plt.figure(figsize=(10, 5))
        plt.plot(test_dates, real_values, label='Real Price', color='blue')
        plt.plot(test_dates, predictions, label='Predicted Price (ARIMA)', color='red')
        plt.title(f"{item_name} December ARIMA Forecast vs Real")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        text_str = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
        plt.text(test_dates.min(), max(real_values.max(), predictions.max()) * 0.9, text_str,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        picture_path = os.path.join(f"{output_dir}/picture", f"{item_name}_december_comparison.png")
        plt.savefig(picture_path, dpi=100)
        plt.close()
        print(f"✅ December comparison plot saved: {picture_path}")

        # Plot 2: November to December range
        plt.figure(figsize=(10, 5))
        plot_df = item_df.loc[plot_start:test_end]
        plt.plot(plot_df.index, plot_df['price'], label='Historical Price', color='blue')
        plt.plot(test_dates, predictions, label='Predicted Price (ARIMA)', color='red')
        plt.title(f"{item_name} November to December ARIMA Forecast vs Real")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.text(plot_start, max(plot_df['price'].max(), predictions.max()) * 0.9, text_str,
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
