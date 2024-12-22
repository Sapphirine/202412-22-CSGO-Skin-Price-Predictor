from flask import Flask, jsonify, request, render_template, send_from_directory
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from prophet import Prophet
import statsmodels.api as sm
from realtime_predict import CSGOMarketScraper, CSGOPricePredictor
from plot_data import plot_price_history_from_db


# Flask App
app = Flask(__name__, static_folder='static')


# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/<item_name>')
def predict(item_name):
    predictor = CSGOPricePredictor()
    scraper = CSGOMarketScraper()
    raw_price_path = f'../csgo_market/prices/{item_name}.csv'

    # Check if price data exists; if not, trigger data collection
    if not os.path.exists(raw_price_path):
        print(f"Price data for {item_name} not found. Collecting data...")
        scraper.collect_price_histories([item_name])

        # If data still doesn't exist after collection, return an error
        if not os.path.exists(raw_price_path):
            return jsonify({'error': f"Failed to collect price data for {item_name}."}), 404

    # Proceed with prediction
    forecast_df, current_price, changes = predictor.predict_prices(item_name, raw_price_path)

    # Prepare response
    response = {
        'item_name': item_name,
        'current_price': current_price,
        'forecast': forecast_df['weighted_forecast'].to_dict(),
        'changes': changes
    }
    return jsonify(response)


@app.route('/historical/<item_name>')
def historical(item_name):
    try:
        df = pd.read_csv(f'../csgo_market/prices/{item_name}.csv')
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/plot/price_history/<skin_name>')
def plot_price_history(skin_name):
    try:
        plot_path = plot_price_history_from_db(skin_name, db_path="data/ak47_skins.db")
        return send_from_directory('static/plots', f'{skin_name}_trends.png')
    except Exception as e:
        return jsonify({'error': f"Error generating plot: {e}"})


@app.route('/plot/daily_activity')
def plot_daily_activity():
    try:
        # Fetch daily activity data
        activity_data_path = '../csgo_market/daily_activity.csv'
        if not os.path.exists(activity_data_path):
            return jsonify({'error': 'Daily activity data not available.'})

        activity_df = pd.read_csv(activity_data_path)
        activity_df['date'] = pd.to_datetime(activity_df['date'])
        plt.figure(figsize=(12, 6))
        plt.plot(activity_df['date'], activity_df['active_players'], label='Daily Active Players', color='blue')
        plt.title('Daily Active Players Trend')
        plt.xlabel('Date')
        plt.ylabel('Active Players')
        plt.legend()

        # Save plot
        plot_path = 'static/plots/daily_activity_plot.png'
        plt.savefig(plot_path, dpi=100)
        plt.close()

        return send_from_directory('static/plots', 'daily_activity_plot.png')
    except Exception as e:
        return jsonify({'error': f"Error generating daily activity plot: {e}"})


# Helper Methods for Prediction
class CSGOPricePredictor:
    def __init__(self):
        self.predict_dir = '../consequence/predict'
        self.predict_csv_dir = os.path.join(self.predict_dir, 'csv')
        self.predict_picture_dir = os.path.join(self.predict_dir, 'picture')
        os.makedirs(self.predict_csv_dir, exist_ok=True)
        os.makedirs(self.predict_picture_dir, exist_ok=True)
        self.arima_days = 30
        self.forecast_days = 365

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
        Calculate percentage change for specified time periods.
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
        """
        Execute the complete prediction process.
        """
        price_df = pd.read_csv(raw_price_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.set_index('date')

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

        return combined_forecast, current_price, changes


if __name__ == "__main__":
    app.run(debug=True, port=5001)
