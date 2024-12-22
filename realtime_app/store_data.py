import sqlite3
import requests
import re
import json

def get_price_history(market_hash_name):
    url = f"https://steamcommunity.com/market/listings/730/{requests.utils.quote(market_hash_name)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            line_pattern = r'var line1=(\[.*?\]);'
            match = re.search(line_pattern, response.text)
            if match:
                return json.loads(match.group(1))
        return None
    except Exception as e:
        print(f"Error fetching price history: {e}")
        return None

def create_database():
    conn = sqlite3.connect('ak47_redline_data.db')
    c = conn.cursor()
    # Create the price_history table if it does not exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skin_name TEXT,
            date TEXT,
            price REAL,
            volume INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("Database and table have been created successfully!")

def store_data_to_db(skin_name, price_data):
    conn = sqlite3.connect('ak47_redline_data.db')
    c = conn.cursor()

    for entry in price_data:
        c.execute('''
            INSERT INTO price_history (skin_name, date, price, volume)
            VALUES (?, ?, ?, ?)
        ''', (skin_name, entry[0], entry[1], entry[2]))

    conn.commit()
    conn.close()
    print(f"Data for {skin_name} has been stored successfully!")

if __name__ == "__main__":
    create_database()
    skin_name = "AK-47 | Redline (Battle-Scarred)"
    print(f"Fetching data for {skin_name}...")
    price_data = get_price_history(skin_name)

    if price_data:
        print(f"Storing data for {skin_name}...")
        store_data_to_db(skin_name, price_data)
    else:
        print(f"Unable to fetch data for {skin_name}.")
