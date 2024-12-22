import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import sqlite3

def plot_price_history_from_db(skin_name, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Fetch data for the specific skin
    c.execute('''
        SELECT date, price, volume
        FROM price_history
        WHERE skin_name = ?
        ORDER BY date ASC
    ''', (skin_name,))
    data = c.fetchall()
    conn.close()

    if not data:
        print(f"No data available for {skin_name}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['date', 'price', 'volume'])
    df['date'] = pd.to_datetime(df['date'], format='%b %d %Y %H:', errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')  # Ensure volume is numeric

    # Debugging statements
    print(df.head())
    print(df.dtypes)

    # Drop rows with missing volume values
    if df['volume'].isnull().any():
        print("Found NaN in 'volume' column, dropping these rows.")
        df.dropna(subset=['volume'], inplace=True)

    # Plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Price trend
    sns.lineplot(data=df, x='date', y='price', ax=ax1, color='#FF4B4B')
    ax1.set_title(f'{skin_name} Price History', pad=20)
    ax1.set_xlabel('')
    ax1.set_ylabel('Price (USD)')

    # Volume trend
    sns.barplot(data=df, x='date', y='volume', ax=ax2, color='#4B69FF', alpha=0.6)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as an image
    plot_path = f'static/plots/{skin_name}_trends.png'
    fig.savefig(plot_path)
    plt.close(fig)

    return plot_path
