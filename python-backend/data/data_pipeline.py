import os
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import time
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Убедитесь, что у вас есть файл capital_request.py в той же папке
from capital_request import CapitalComAPI

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INSTRUMENTS = ["GOLD", "DXY", "VIX"]

# Загрузка переменных окружения (API ключей)
load_dotenv()

class MarketAuxNewsFetcher:
    """Класс для получения новостей с MarketAux.com."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("MarketAux API token not found. Please add MARKETAUX_API_TOKEN to your .env file.")
        self.api_key = api_key
        self.base_url = "https://api.marketaux.com/v1/news/all"

    def fetch_news(self, from_date: datetime, to_date: datetime) -> list:
        """
        Получает новости, связанные с макроэкономикой и золотом.
        """
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # --- ИЗМЕНЕНИЕ: Используем более широкий и эффективный запрос ---
        params = {
            'api_token': self.api_key,
            'group': 'general,market', # Ищем общие и рыночные новости
            'language': 'en',
            'published_after': f'{from_date_str}T00:00:00',
            'published_before': f'{to_date_str}T23:59:59',
        }
        
        logger.info(f"Fetching news from {from_date.date()} to {to_date.date()}...")
        try:
            time.sleep(1) 
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get('data', [])
            logger.info(f"Successfully fetched {len(articles)} news articles from MarketAux for this period.")
            return articles
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch news from MarketAux: {e}")
            if hasattr(response, 'text'):
                logger.error(f"API Response: {response.text}")
            return []

def create_instrument_df(price_data: dict, epic: str) -> pd.DataFrame:
    """
    Converts a price response from Capital.com into a DataFrame 
    with only the Close price for a specific instrument.
    """
    # Return an empty DataFrame if the input data is invalid or has no prices
    if not price_data or 'prices' not in price_data:
        return pd.DataFrame()

    prices_list = []
    # Loop through each hourly price point in the JSON response
    for price_point in price_data['prices']:
        
        # Check to make sure all required data fields exist to avoid errors
        if 'snapshotTime' in price_point and price_point['snapshotTime'] and 'closePrice' in price_point and 'bid' in price_point['closePrice']:
            
            # Convert the timestamp string to a proper timezone-aware Datetime object
            dt_aware = pd.to_datetime(price_point['snapshotTime']).tz_localize('UTC')
            
            # Append the timestamp and the close price to our list
            prices_list.append({
                'Datetime': dt_aware,
                # Create a column name specific to the instrument, e.g., 'GOLD_Close'
                f'{epic}_Close': float(price_point['closePrice']['bid']),
            })
    
    # If we couldn't process any price points, return an empty DataFrame
    if not prices_list:
        return pd.DataFrame()

    # Convert the list of price points into a Pandas DataFrame
    df = pd.DataFrame(prices_list)
    # Set the 'Datetime' column as the index for easy merging later
    df.set_index('Datetime', inplace=True)
    
    return df

def main():
    
    """Главная функция для запуска конвейера данных."""
    logger.info("🚀 Starting Data Pipeline with MarketAux...")
    
    capital_api_key = os.getenv("CAPITAL_API_KEY")
    capital_identifier = os.getenv("CAPITAL_IDENTIFIER")
    capital_password = os.getenv("CAPITAL_PASSWORD")
    
    price_checker = CapitalComAPI(capital_api_key, capital_identifier, capital_password)
    if not price_checker.login_and_get_tokens(): return

    # --- Шаг 1: Загружаем данные о ценах за год, 30-дневными "кусками" ---
    logger.info("Fetching historical price data for the last year...")
    all_price_dfs = []
    total_days_to_fetch = 360
    chunk_days = 30

    INSTRUMENTS = ["GOLD", "DXY", "VIX"]
    # Use a dictionary to store lists of DataFrames for each instrument
    all_instrument_dfs = {epic: [] for epic in INSTRUMENTS}
    total_days_to_fetch = 360
    chunk_days = 30

    #In summary, you are replacing the simple loop that only knew about "GOLD" with a more robust system that can #handle any number of instruments you define in the INSTRUMENTS list.

    for i in range(0, total_days_to_fetch, chunk_days):
        to_date = datetime.now(timezone.utc) - timedelta(days=i)
        from_date = to_date - timedelta(days=chunk_days)
        
        logger.info(f"--- Processing chunk {i//chunk_days + 1}/12: from {from_date.date()} to {to_date.date()} ---")
        
        # This is the new inner loop you asked about. It iterates through GOLD, DXY, and VIX.
        for epic in INSTRUMENTS:
            logger.info(f"Fetching price data for {epic}...")
            # The 'epic' variable is now used to dynamically fetch data for the current instrument
            price_data_json = price_checker.get_historical_prices(
                epic=epic,
                resolution="HOUR",
                max_points=1000, 
                from_date=from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                to_date=to_date.strftime('%Y-%m-%dT%H:%M:%S')
            )
            
            if price_data_json:
                # Use the new generic function to create a DataFrame
                chunk_df = create_instrument_df(price_data_json, epic)
                if not chunk_df.empty:
                    # Append the resulting DataFrame to the correct list in our dictionary
                    all_instrument_dfs[epic].append(chunk_df)
            
            time.sleep(1) # Be respectful to the API

    # --- Combine all price data ---
    combined_price_dfs = []
    for epic in INSTRUMENTS:
        if all_instrument_dfs[epic]:
            # Concatenate all chunks for a single instrument
            instrument_df = pd.concat(all_instrument_dfs[epic])
            # Remove any potential duplicate timestamps
            instrument_df = instrument_df[~instrument_df.index.duplicated(keep='first')]
            combined_price_dfs.append(instrument_df)
        else:
            logger.warning(f"Could not retrieve any price data for {epic}.")

    if not combined_price_dfs:
        logger.error("Could not retrieve any price data for any instrument. Exiting.")
        return

    # Merge all instrument dataframes together on the Datetime index
    price_df = pd.concat(combined_price_dfs, axis=1)
    price_df.sort_index(inplace=True)
    
    # Forward-fill data to handle non-trading hours for some instruments
    price_df.ffill(inplace=True)

    # --- Шаг 2: Загружаем новости так же, по частям ---
    marketaux_api_key = os.getenv("MARKETAUX_API_TOKEN")
    news_fetcher = MarketAuxNewsFetcher(marketaux_api_key)
    
    all_news_articles = []
    for i in range(0, total_days_to_fetch, chunk_days):
        to_date = datetime.now(timezone.utc) - timedelta(days=i)
        from_date = to_date - timedelta(days=chunk_days)
        articles_chunk = news_fetcher.fetch_news(from_date, to_date)
        if articles_chunk:
            all_news_articles.extend(articles_chunk)

    # --- Шаг 3: Обрабатываем и объединяем данные ---
    if not all_news_articles:
        logger.warning("No news found. The resulting dataset will have zero sentiment.")
        final_df = price_df.copy()
        final_df['sentiment'] = 0.0
    else:
        # --- SentimentAnalyzer удален, т.к. MarketAux дает готовый sentiment_score ---
        sentiment_data = []
        logger.info(f"Processing sentiment for {len(all_news_articles)} articles...")
        for article in all_news_articles:
            sentiment_score = article.get('sentiment_score', 0.0)
            sentiment_data.append({
                'Datetime': pd.to_datetime(article['published_at']),
                'sentiment': float(sentiment_score)
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index('Datetime', inplace=True)
        
        logger.info("Merging price data and sentiment data...")
        sentiment_df_resampled = sentiment_df.resample('h').mean()
        
        final_df = price_df.join(sentiment_df_resampled)
        
        final_df['sentiment'].ffill(inplace=True)
        final_df['sentiment'].fillna(0, inplace=True)

    final_df.dropna(subset=['Close'], inplace=True)

    logger.info(f"Final merged DataFrame created. Shape: {final_df.shape}")
    print("\n--- Sample of the final dataset ---")
    print(final_df.head())
    print(final_df.tail())

    output_filename = "gold_data_with_sentiment_hourly.csv"
    final_df.to_csv(output_filename)
    logger.info(f"✅ Successfully created and saved the final dataset to '{output_filename}'")


if __name__ == "__main__":
    main()
