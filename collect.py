import requests
import json
import pandas as pd
import time
import os
import re
import argparse

class CSGOMarketScraper:
    def __init__(self, base_dir='csgo_market_data'):
        self.data_dir = base_dir
        self.items_dir = f'{self.data_dir}/items'
        self.prices_dir = f'{self.data_dir}/prices'
        os.makedirs(self.items_dir, exist_ok=True)
        os.makedirs(self.prices_dir, exist_ok=True)
       
    def normalize_filename(self, name):
        name = re.sub(r'[\\/*?:"<>|]', '', name)
        name = name.replace(' ', '_').replace('|', '-')
        return name
   
    def get_market_items(self, start=0, count=100, base_delay=2):
        url = "https://steamcommunity.com/market/search/render/"
        params = {'appid': 730, 'norender': 1, 'start': start, 'count': count, 'currency': 1}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
       
        delay = base_delay
        max_delay = 120
        attempt = 1
       
        while True:
            try:
                print(f"Attempt {attempt}, delay {delay}s...")
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('results', []), data.get('total_count', 0)
                if response.status_code == 429:
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    attempt += 1
                    continue
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            attempt += 1

    def load_items_from_csv(self):
        try:
            csv_path = f'{self.items_dir}/items_list.csv'
            df = pd.read_csv(csv_path)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []

    def get_price_history(self, market_hash_name, base_delay=3):
        url = f"https://steamcommunity.com/market/listings/730/{requests.utils.quote(market_hash_name)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
       
        delay = base_delay
        max_delay = 120
        attempt = 1
       
        while True:
            try:
                print(f"Attempt {attempt}, delay {delay}s...")
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    match = re.search(r'var line1=(\[.*?\]);', response.text)
                    if match:
                        return json.loads(match.group(1))
                if response.status_code == 429:
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    attempt += 1
                    continue
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            attempt += 1
   
    def collect_all_items(self):
        start = 0
        items_list = []
        total_items = 1
       
        while start < total_items:
            items, total_count = self.get_market_items(start=start)
            total_items = total_count
           
            if items:
                items_list.extend(items)
                df = pd.DataFrame(items_list)
                df.to_csv(f'{self.items_dir}/items_list.csv', index=False)
                start += 100
                time.sleep(2)
       
        return items_list
   
    def collect_price_histories(self, items_list):
        for idx, item in enumerate(items_list):
            name = item['name']
            normalized_name = self.normalize_filename(name)
           
            filename = f'{self.prices_dir}/{normalized_name}.csv'
            if os.path.exists(filename):
                print("Already exists, skipping")
                continue
           
            price_history = self.get_price_history(name)
            if price_history:
                df = pd.DataFrame(price_history, columns=['date', 'price', 'volume'])
                df['date'] = pd.to_datetime(df['date'].str.split('+').str[0].str.strip())
                df.to_csv(filename, index=False)
            time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="CSGO Market Scraper")
    parser.add_argument('--base_dir', type=str, default='csgo_market_data', help='Data storage root directory')
    args = parser.parse_args()

    scraper = CSGOMarketScraper(base_dir=args.base_dir)
    print("Step 1: Collect all items")
    items_list = scraper.load_items_from_csv()
    print("\nStep 2: Collect price histories")
    scraper.collect_price_histories(items_list)
    print("\nData collection complete!")

if __name__ == "__main__":
    main()
