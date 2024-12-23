# 202412-22-CSGO-Skin-Price-Predictor

This project provides a **CSGO Skin Price Prediction System** that integrates **real-time data collection**, **price prediction** (ARIMA, Prophet, LSTM, GAN), and **model verification**. It also includes RESTful APIs for easy access.


---

## Project Youtube Vedio
https://www.youtube.com/watch?v=MA5-19DV08U

---

## Project Structure

```
202412-22-CSGO-Skin-Price-Predictor/
│
├── consequence/                    # Prediction results
├── csgo_market_data/               # Market data (raw and aggregated)
├── data_raw/                       # Reddit and Google trend data
├── predict_and_verification/       # Models and verification scripts
├── realtime_app/                   # Flask application and prediction utilities
│   ├── app.py                      # Main Flask app
│   ├── realtime_predict.py         # Combines ARIMA, Prophet, LSTM, and GAN for predictions
│   ├── store_data.py               # Store and preprocess data
│   ├── plot_data.py                # Generate plots
│   └── csgo_items.db               # Preprocessed SQLite database for skins
├── collect.py                      # General data collection
├── dailyactive_collect.py          # Collect daily active player data
├── dailyactive_peak_collect.py     # Collect peak player data
├── google_trend_collect.py         # Collect Google Trends data
├── reddit_praw.py                  # Reddit post classification
└── data_complete.json              # Combined market and activity data (user-provided)
```

---

## Features

### 1. **Data Collection**

#### **`collect.py`**
- **Purpose**: Collect price history data for CSGO items from Steam Market.
- **Parameters**:
  - `years` (int): Specifies how many years of data to collect. If the range exceeds the data's earliest date, it fetches all available data.
  - Modify `items_to_collect` list in the script to specify items to fetch.
- **Example Usage**:
  ```bash
  python collect.py --years 5
  ```
- **Output**:
  - Saved price history CSV files in `../csgo_market/prices`.

---

#### **`dailyactive_collect.py`**
- **Purpose**: Collect daily active player data.
- **Parameters**:
  - None. Adjust the collection logic in the script if needed.
- **Example Usage**:
  ```bash
  python dailyactive_collect.py
  ```
- **Output**:
  - Generates a CSV file with daily active player data in `data_raw/`.

---

#### **`dailyactive_peak_collect.py`**
- **Purpose**: Collect peak player data.
- **Parameters**:
  - None. Adjust the logic if a different data source is needed.
- **Example Usage**:
  ```bash
  python dailyactive_peak_collect.py
  ```
- **Output**:
  - Generates a CSV file with peak player data in `data_raw/`.

---

#### **`google_trend_collect.py`**
- **Purpose**: Fetch Google Trends data for CSGO-related keywords.
- **Parameters**:
  - `keywords` (list): A list of terms to search trends for (e.g., `"CS:GO"`, `"AK-47"`).
  - `timeframe` (str): Time range for the trends (e.g., `"today 12-m"`, `"all"`).
  - `region` (str): Country or region for localized trends (default is global).
- **Example Usage**:
  ```bash
  python google_trend_collect.py --timeframe "today 12-m"
  ```
- **Output**:
  - Saves Google Trends data to `data_raw/csgo_trend_data.csv`.

---

#### **`reddit_praw.py`**
- **Purpose**: Classify Reddit posts from `r/GlobalOffensive` into predefined categories.
- **Parameters**:
  - Modify `KEYWORDS` in the script to customize post classification.
  - `max_retries` (int): Maximum retries for fetching data in case of errors.
- **Example Usage**:
  ```bash
  python reddit_praw.py
  ```
- **Output**:
  - Classified posts saved to `data_raw/reddit_post/`.

---

### 2. **Prediction Models**

#### **`realtime_predict.py`**
- **Purpose**: Combines **ARIMA**, **Prophet**, **LSTM**, and **GAN** models for comprehensive price prediction.
- **Parameters**:
  - `items_to_collect` (list): List of item names to predict.
  - ARIMA: Modify the `(p, d, q)` parameters in the script.
  - Prophet: Customize seasonality (e.g., yearly, weekly) and changepoint sensitivity.
  - LSTM: Adjust hyperparameters like `window_size`, `epochs`, and `batch_size`.
  - GAN: Modify noise dimensions and training epochs.
- **Example Usage**:
  ```bash
  python realtime_app/realtime_predict.py
  ```
- **Output**:
  - Predictions (CSV and plots) saved in `../realtime/predict/csv` and `../realtime/predict/picture`.

---

### 3. **Model Verification**

#### **`arima_verification.py`**
- **Purpose**: Verify ARIMA model predictions against historical data.
- **Parameters**:
  - `--item` (str): A specific weapon name, or a path to a `.txt` or `.json` file containing a list of weapon names.
- **Example Usage**:
  ```bash
  python predict_and_verification/arima_verification.py --item items.txt
  ```
- **Output**:
  - Metrics and plots saved in `consequence/verification/arima`.

---

#### **`lstm_verification.py`**
- **Purpose**: Verify LSTM model predictions.
- **Parameters**:
  - `--item` (str): A specific weapon name, or a path to a `.txt` or `.json` file containing a list of weapon names.
  - Change `window_size`, `epochs`, or other LSTM parameters within the script.
- **Example Usage**:
  ```bash
  python predict_and_verification/lstm_verification.py --item items.json
  ```
- **Output**:
  - Metrics and plots saved in `consequence/verification/lstm`.

---

#### **`gan_verification.py`**
- **Purpose**: Verify GAN-based predictions.
- **Parameters**:
  - `--item` (str): A specific weapon name, or a path to a `.txt` or `.json` file containing a list of weapon names.
  - Modify GAN architecture or training parameters in the script.
- **Example Usage**:
  ```bash
  python predict_and_verification/gan_verification.py --item items.json
  ```
- **Output**:
  - Metrics and plots saved in `consequence/verification/gan`.

---

#### **`prophet_verification.py`**
- **Purpose**: Verify Prophet model predictions.
- **Parameters**:
  - `--item` (str): A specific weapon name, or a path to a `.txt` or `.json` file containing a list of weapon names.
  - Customize seasonality settings and forecast periods.
- **Example Usage**:
  ```bash
  python predict_and_verification/prophet_verification.py --item items.txt --yearly True --future_days 365
  ```
- **Output**:
  - Metrics and plots saved in `consequence/verification/prophet`.

---

### 4. **Flask Web API**

#### **Run Flask App**
- Navigate to the `realtime_app/` directory and start the Flask server:
  ```bash
  python app.py
  ```

#### **API Endpoints**

- **Homepage**:
  - URL: `http://127.0.0.1:5001/`
  - Description: Displays the introduction page.

- **Price Prediction**:
  - URL: `http://127.0.0.1:5001/predict/<item_name>`
  - **Parameters**:
    - `item_name`: The name of the item to predict (e.g., `AK-47__Redline_(Battle-Scarred)`).
  - Example:
    ```
    http://127.0.0.1:5001/predict/AK-47__Redline_(Battle-Scarred)
    ```
  - **Description**:
    - Collects price history if unavailable.
    - Combines ARIMA, Prophet, LSTM, and GAN predictions.
    - Returns predictions and percentage changes.

- **Historical Data**:
  - URL: `http://127.0.0.1:5001/historical/<item_name>`
  - **Parameters**:
    - `item_name`: The name of the item (e.g., `AK-47__Redline_(Battle-Scarred)`).
  - Example:
    ```
    http://127.0.0.1:5001/historical/AK-47__Redline_(Battle-Scarred)
    ```

- **Price History Plot**:
  - URL: `http://127.0.0.1:5001/plot/price_history/<skin_name>`
  - **Parameters**:
    - `skin_name`: The name of the skin to plot (e.g., `AK-47__Redline_(Battle-Scarred)`).
  - Example:
    ```
    http://127.0.0.1:5001/plot/price_history/AK-47__Redline_(Battle-Scarred)
    ```

- **Daily Activity Plot**:
  - URL: `http://127.0.0.1:5001/plot/daily_activity`
  - **Description**:
    - Generates and returns a plot for daily active player trends.

---

## Data Details

### 1. `data_complete.json`
This is a user-provided file combining market and activity data. Example structure:
```json
[
    {
        "date": "2024-01-01",
        "peak_players": 123456,
        "items": {
            "ak47": {
                "positive_count": 500,
                "negative_count": 50
            }
        },
        "Tournaments": {"positive_ratio": 0.8},
        "Version_Updates": {"positive_ratio": 0.7},
        "Market_and_Price": {"positive_ratio": 0.9},
        "trends": {
            "CS:GO": 120,
            "Global Offensive": 100,
            "CS2": 110,
            "tournament": 90,
            "market_activity": 95,
            "update_trend": 80
        }
    }
]
```

### 2. Price Data
Stored in `../csgo_market/prices`, formatted as:
```csv
date,price,volume
2024-01-01,1.23,500
2024-01-02,1.25,550
...
```

### 3. Daily Activity Data
Stored in `../csgo_market/daily_activity.csv`, formatted as:
```csv
date,active_players
2024-01-01,123456
2024-01-02,124789
...
```

---

## Example Output

### Price Prediction API
**Request**:
```
GET http://127.0.0.1:5001/predict/AK-47__Redline_(Battle-Scarred)
```

**Response**:
```json
{
  "item_name": "AK-47__Redline_(Battle-Scarred)",
  "current_price": 123.45,
  "forecast": {
    "2024-12-01": 124.56,
    "2024-12-02": 125.78,
    ...
  },
  "changes": {
    "Tomorrow": "+2.34%",
    "3 Days": "+5.67%",
    "7 Days": "+8.90%",
    "1 Month": "+12.34%",
    ...
  }
}
```


