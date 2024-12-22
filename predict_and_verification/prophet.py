import os
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from prophet import Prophet
import argparse


def main():
    # Add argparse support
    parser = argparse.ArgumentParser(description="Prophet Price Forecasting")
    parser.add_argument("--item", type=str, required=True, help="Item name or path to a .txt/.json file containing item names.")
    parser.add_argument("--yearly", type=bool, default=True, help="Enable yearly seasonality.")
    parser.add_argument("--weekly", type=bool, default=False, help="Enable weekly seasonality.")
    parser.add_argument("--daily", type=bool, default=False, help="Enable daily seasonality.")
    parser.add_argument("--future_days", type=int, default=365, help="Number of future days to forecast.")
    args = parser.parse_args()

    # Default to yearly seasonality if all seasonality parameters are False
    if not (args.yearly or args.weekly or args.daily):
        print("All seasonality settings are False. Defaulting to yearly seasonality.")
        args.yearly = True

    # Parse item list
    items = parse_item_list(args.item)

    # Create output directories
    base_dir = "verification/prophet"
    output_dir = f"{base_dir}_{'yearly' if args.yearly else ''}{'_weekly' if args.weekly else ''}{'_daily' if args.daily else ''}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/picture", exist_ok=True)

    price_dir = "item_smooth_prices"
    end_date = datetime(2024, 12, 13)
    start_pred = end_date + timedelta(days=1)
    future_days = args.future_days
    end_pred = start_pred + timedelta(days=future_days - 1)

    for item_name in items:
        item_path = os.path.join(price_dir, f"{item_name}.csv")
        if not os.path.exists(item_path):
            print(f"{item_path} not found, skipping {item_name}")
            continue

        item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
        if item_df.empty:
            print(f"No data for {item_name}, skipping Prophet forecast.")
            continue

        # Limit data to end_date
        item_df = item_df[item_df.index <= end_date]
        if item_df.empty:
            print(f"No data up to {end_date} for {item_name}, skipping.")
            continue

        # Prepare data for Prophet
        df_prophet = item_df.copy()
        df_prophet = df_prophet.rename(columns={"price": "y"})
        df_prophet['ds'] = df_prophet.index
        df_prophet = df_prophet[['ds', 'y']]

        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=args.yearly,
            weekly_seasonality=args.weekly,
            daily_seasonality=args.daily
        )
        model.fit(df_prophet)

        # Create future dates
        future = model.make_future_dataframe(periods=future_days, freq='D')
        forecast = model.predict(future)

        # Filter forecast period
        forecast_period = forecast[(forecast['ds'] >= start_pred) & (forecast['ds'] <= end_pred)]

        # Save future forecast data
        future_path = os.path.join(output_dir, f"future_prophet_{item_name}.csv")
        forecast_period.to_csv(future_path, index=False, date_format='%Y-%m-%d')
        print(f"✅ Future forecast saved: {future_path}")

        # Create complete series
        complete_df = pd.concat([df_prophet.set_index('ds')[['y']],
                                 forecast_period.set_index('ds')[['yhat']]], axis=1)
        complete_df = complete_df.rename(columns={'y': 'price', 'yhat': 'forecast_price'})
        complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
        complete_path = os.path.join(output_dir, f"complete_prophet_{item_name}.csv")
        complete_df.to_csv(complete_path, date_format='%Y-%m-%d')
        print(f"✅ Complete series saved: {complete_path}")

        # Plot historical and forecast data
        plt.figure(figsize=(10, 5))
        plt.plot(item_df.index, item_df['price'], label='Historical Price', color='blue')
        plt.plot(forecast_period['ds'], forecast_period['yhat'], label='Prophet Forecast', color='green')
        plt.title(f"Prophet Forecast for {item_name} ({'Yearly' if args.yearly else ''} {'Weekly' if args.weekly else ''} {'Daily' if args.daily else ''})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        picture_path = os.path.join(f"{output_dir}/picture", f"{item_name}_prophet_forecast_plot.png")
        plt.savefig(picture_path, dpi=100)
        plt.close()
        print(f"✅ Forecast plot saved: {picture_path}")


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


if __name__ == "__main__":
    main()
