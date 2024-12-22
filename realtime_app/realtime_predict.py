import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import statsmodels.api as sm
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import re
import requests
import time


class CSGOMarketScraper:
    def __init__(self):
        self.prices_dir = '../csgo_market/prices'
        os.makedirs(self.prices_dir, exist_ok=True)

    def normalize_filename(self, name):
        name = re.sub(r'[\\/*?:"<>|]', '', name)
        name = name.replace(' ', '_').replace('|', '-')
        return name

    def get_price_history(self, market_hash_name, base_delay=3):
        url = f"https://steamcommunity.com/market/listings/730/{requests.utils.quote(market_hash_name)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        delay = base_delay
        max_delay = 120
        attempt = 1

        while True:
            try:
                print(f"Fetching price history for {market_hash_name}, attempt {attempt}, delay {delay}s...")
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    line_pattern = r'var line1=(\[.*?\]);'
                    match = re.search(line_pattern, response.text)
                    if match:
                        return json.loads(match.group(1))

                if response.status_code == 429:
                    print(f"Request throttled, retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    attempt += 1
                    continue

            except Exception as e:
                print(f"Error fetching price history: {e}")

            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            attempt += 1

    def collect_price_histories(self, items_to_collect):
        """Collect price histories and ensure the data is up to yesterday."""
        yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

        for idx, item_name in enumerate(items_to_collect):
            print(f"\nProcessing item [{idx + 1}/{len(items_to_collect)}]: {item_name}")

            normalized_name = self.normalize_filename(item_name)
            filename = f'{self.prices_dir}/{normalized_name}.csv'

            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                latest_date = existing_df['date'].max()

                if latest_date >= yesterday:
                    print(f"✅ Up-to-date data available for {item_name} (latest: {latest_date.date()}), skipping.")
                    continue

            price_history = self.get_price_history(item_name)
            if price_history:
                cleaned_data = [[entry[0], entry[1], entry[2]] for entry in price_history]
                df = pd.DataFrame(cleaned_data, columns=['date', 'price', 'volume'])
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.to_csv(filename, index=False)
                print(f"✅ Saved price history for {item_name}.")
            else:
                print(f"❌ Failed to fetch price history for {item_name}.")

            time.sleep(3)


class CSGOPricePredictor:
    def __init__(self):
        self.base_dir = '../realtime'
        self.smooth_dir = os.path.join(self.base_dir, 'smooth_price')
        self.predict_dir = os.path.join(self.base_dir, 'predict')
        self.predict_csv_dir = os.path.join(self.predict_dir, 'csv')
        self.predict_picture_dir = os.path.join(self.predict_dir, 'picture')

        for directory in [self.smooth_dir, self.predict_csv_dir, self.predict_picture_dir]:
            os.makedirs(directory, exist_ok=True)

        self.scaler = MinMaxScaler()
        self.arima_days = 30
        self.forecast_days = 365

    def smooth_and_interpolate_data(self, raw_price_path):
        df = pd.read_csv(raw_price_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).set_index('date').sort_index()
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_range)
        df['price'] = df['price'].interpolate(method='linear').ffill().bfill()
        return df

    def train_arima(self, price_df):
        y = price_df['price']
        model = sm.tsa.ARIMA(y, order=(1, 1, 1))
        result = model.fit()
        forecast = result.forecast(steps=self.arima_days)
        forecast_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=self.arima_days)
        return pd.DataFrame({'date': forecast_dates, 'arima_forecast': forecast}).set_index('date')

    def train_prophet(self, price_df):
        df_prophet = price_df.reset_index().rename(columns={'index': 'ds', 'price': 'y'})
        model = Prophet(yearly_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=self.forecast_days)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'prophet_forecast'}).set_index('date')

    def combine_forecasts(self, arima_forecast, prophet_forecast):
        combined = pd.concat([arima_forecast, prophet_forecast], axis=1)
        combined['weighted_forecast'] = combined.apply(
            lambda row: row['arima_forecast'] if pd.notna(row['arima_forecast']) else row['prophet_forecast'], axis=1
        )
        return combined

    def calculate_changes(self, current_price, forecast_series):
        """
        Calculate the percentage change for each specified time period.
        """
        periods = {
            'Tomorrow': 1,
            '3 Days': 3,
            '7 Days': 7,
            '1 Month': 30,
            '1 Quarter': 90,
            '6 Months': 180,
            '1 Year': 365
        }

        changes = {}
        for period_name, days in periods.items():
            if days <= len(forecast_series):
                future_price = forecast_series.iloc[days - 1]
                change_pct = ((future_price - current_price) / current_price) * 100
                changes[period_name] = change_pct
            else:
                changes[period_name] = None

        return changes

    def predict_prices(self, item_name, raw_price_path):
        price_df = self.smooth_and_interpolate_data(raw_price_path)
        arima_forecast = self.train_arima(price_df)
        prophet_forecast = self.train_prophet(price_df)
        combined_forecast = self.combine_forecasts(arima_forecast, prophet_forecast)

        forecast_path = os.path.join(self.predict_csv_dir, f"future_{item_name}.csv")
        combined_forecast.to_csv(forecast_path)

        # Plot results
        plt.figure(figsize=(15, 8))
        plt.plot(price_df.index, price_df['price'], label='Historical Price', color='blue', alpha=0.6)
        plt.plot(combined_forecast.index, combined_forecast['weighted_forecast'], label='Weighted Forecast', color='purple')
        plt.plot(arima_forecast.index, arima_forecast['arima_forecast'], label='ARIMA Forecast', color='red', linestyle='--')
        plt.plot(prophet_forecast.index, prophet_forecast['prophet_forecast'], label='Prophet Forecast', color='green', linestyle='--')
        plt.title(f"Price Forecast for {item_name}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        picture_path = os.path.join(self.predict_picture_dir, f"{item_name}_forecast_plot.png")
        plt.savefig(picture_path, dpi=100)
        plt.close()
        print(f"✅ Forecast plot saved for {item_name}: {picture_path}")

        # Calculate percentage changes
        current_price = price_df['price'].iloc[-1]
        changes = self.calculate_changes(current_price, combined_forecast['weighted_forecast'])

        # Print results
        print(f"\nPrediction Report for {item_name}:")
        print(f"Current Price: {current_price:.2f}")
        for period, change in changes.items():
            if change is not None:
                print(f"{period}: {change:+.2f}%")
            else:
                print(f"{period}: Not enough data")
        print("=" * 50)

        return combined_forecast


def main():
    items_to_collect = [
        "Sealed Graffiti | Recoil AK-47 (Wire Blue)",
        "AK-47 | Blue Laminate (Field-Tested)",
        "AK-47 | Asiimov (Factory New)",
        "AK-47 | Nightwish (Factory New)"
    ]

    scraper = CSGOMarketScraper()
    scraper.collect_price_histories(items_to_collect)

    predictor = CSGOPricePredictor()
    for item_name in items_to_collect:
        normalized_name = predictor.normalize_filename(item_name)
        raw_price_path = f"../csgo_market/prices/{normalized_name}.csv"
        if not os.path.exists(raw_price_path):
            print(f"Price data not found for {item_name}: {raw_price_path}")
            continue
        predictor.predict_prices(normalized_name, raw_price_path)


if __name__ == "__main__":
    main()
