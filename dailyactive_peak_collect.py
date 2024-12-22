import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import matplotlib.pyplot as plt


class CSGOPeakDataCollector:
    def __init__(self, save_dir='data_raw', years=5):
        self.base_url = "https://steamcharts.com/app/730/chart-data.json"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.save_dir = save_dir
        self.years = years
        os.makedirs(self.save_dir, exist_ok=True)

    def get_historical_data(self):
        """
        Fetches CS:GO historical player data.
        """
        try:
            response = requests.get(self.base_url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed with status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def process_data(self, data):
        """
        Processes raw data into a DataFrame with only peak players.
        """
        df = pd.DataFrame(data, columns=['timestamp', 'peak_players'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'peak_players']].sort_values('timestamp')
        return df

    def filter_data_by_years(self, df):
        """
        Filters data based on the specified time range in years.
        """
        earliest_date = df['timestamp'].min()
        cutoff_date = datetime.now() - timedelta(days=self.years * 365)

        if cutoff_date < earliest_date:
            cutoff_date = earliest_date

        print(f"Filtering data from: {cutoff_date} to {datetime.now()}")
        return df[df['timestamp'] >= cutoff_date]

    def save_data(self, df):
        """
        Saves data to a CSV file with the format csgo_peak_data_YYYYMMDD_years.csv.
        """
        filename = f"{self.save_dir}/csgo_peak_data_{datetime.now().strftime('%Y%m%d')}_{self.years}years.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to: {filename}")

    def plot_data(self, df):
        """
        Plots peak player data.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['peak_players'], label='Peak Players', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Number of Players')
        plt.title('CS:GO Peak Players Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Executes the full workflow: fetch, process, save, and plot data.
        """
        print("Starting CS:GO peak player data collection...")
        raw_data = self.get_historical_data()
        if raw_data is None:
            return

        df = self.process_data(raw_data)

        # Filter data by the specified number of years
        df_filtered = self.filter_data_by_years(df)

        self.save_data(df_filtered)

        print(f"\nStatistics:")
        print(f"Data collected from: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
        print(f"Total records: {len(df_filtered)}")
        print(f"\nPeak players (max): {df_filtered['peak_players'].max()}")

        self.plot_data(df_filtered)


def main():
    parser = argparse.ArgumentParser(description="CSGO Peak Data Collector")
    parser.add_argument('--save_dir', type=str, default='data_raw', help='Directory to save the collected data')
    parser.add_argument('--years', type=int, default=5, help='Number of years to collect data for (e.g., 5 for last 5 years)')
    args = parser.parse_args()

    collector = CSGOPeakDataCollector(save_dir=args.save_dir, years=args.years)
    collector.run()


if __name__ == "__main__":
    main()
