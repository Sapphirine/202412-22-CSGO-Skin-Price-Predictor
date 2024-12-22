import time
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import os


def fetch_trend_data_with_backoff(keywords, timeframe="all", region="", max_retries=5):
    """
    Fetch Google Trends data for multiple keywords with exponential backoff.

    Args:
        keywords (list): A list of topics to track (e.g., ["CS:GO", "major"]).
        timeframe (str): The timeframe to query (e.g., "today 12-m", "all").
        region (str): The country-specific interest (default is global).
        max_retries (int): Maximum number of retries after failures.

    Returns:
        pd.DataFrame: A DataFrame containing trends data for all keywords.
    """
    pytrends = TrendReq(hl="en-US", tz=360)

    all_trends = pd.DataFrame()

    for keyword in keywords:
        print(f"Fetching trend data for keyword: {keyword}")
        
        retries = 0
        wait_time = 10  # Initial delay in seconds

        while retries <= max_retries:
            try:
                pytrends.build_payload([keyword], timeframe=timeframe, geo=region)
                data = pytrends.interest_over_time()
                
                if data.empty:
                    print(f"No trend data found for the keyword: {keyword}")
                    break
                
                data = data.drop(columns=["isPartial"])
                
                if all_trends.empty:
                    all_trends = data.rename(columns={keyword: keyword})
                else:
                    all_trends = all_trends.join(data.rename(columns={keyword: keyword}), how="outer")
                
                break
            
            except Exception as e:
                retries += 1
                print(f"Error fetching data for {keyword}: {e}")
                
                if retries > max_retries:
                    print(f"Maximum retries reached for {keyword}. Skipping...")
                    break
                
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                
    return all_trends


def plot_trend_data(data, keywords_to_plot=None, title="Google Trends Data"):
    """
    Plot Google Trends data.

    Args:
        data (pd.DataFrame): The trends data to plot.
        keywords_to_plot (list): A list of keywords to plot. If None, plots all keywords.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    if keywords_to_plot is None:
        keywords_to_plot = data.columns

    for keyword in keywords_to_plot:
        if keyword in data.columns:
            plt.plot(data.index, data[keyword], label=keyword)

    plt.xlabel("Date")
    plt.ylabel("Search Interest")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Ensure the directory for saving data exists
    save_dir = "data_raw"
    os.makedirs(save_dir, exist_ok=True)

    keywords = [
        "CS:GO", "Counter-Strike", "Global Offensive", "CS2", "Valve CS",
        "CS:GO Major", "Major Championship", "tournament", "IEM Katowice", "ESL One",
        "PGL Major", "BLAST Premier", "DreamHack", "Faceit Major", "ESEA league",
        "CS:GO Finals", "CS:GO Playoffs", "Valve Major", "CS:GO esports", "CS2 update",
        "CS:GO qualifiers", "CS:GO playoff match", "CS:GO grand finals", "CS:GO patch",
        "Operation Riptide", "Operation Broken Fang", "Operation Hydra", "Operation Wildfire",
        "CS:GO skins", "CS:GO trade", "CS:GO market", "CS:GO knives", "CS:GO case",  
        "CS:GO trading platform", "CS:GO keys", "CS:GO AK-47 skins", "CS:GO AWP skins",
        "CS:GO knives market", "CS:GO sticker", "CS:GO graffiti", "CS:GO VAC ban",
        "CS:GO hacking", "CS:GO ban detection", "CS2 VAC system", "CS:GO pro players",
        "CS:GO best team", "CS:GO news", "CS:GO highlights", "CS:GO funny moments",
        "CS:GO maps", "CS:GO Mirage", "CS:GO Dust 2", "CS:GO Inferno"
    ]
    
    trend_data = fetch_trend_data_with_backoff(keywords, timeframe="all")
    
    if not trend_data.empty:
        # Save the trend data in the 'dataraw' folder
        file_path = os.path.join(save_dir, "csgo_trend_data.csv")
        trend_data.to_csv(file_path)
        print(f"Trend data saved to: {file_path}")
        
        # Plot specific keywords
        keywords_to_plot = ["CS:GO", "Counter-Strike", "CS:GO Major"]
        plot_trend_data(trend_data, keywords_to_plot=keywords_to_plot, title="CS:GO Search Trends")
