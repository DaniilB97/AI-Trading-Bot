import os
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import time
import sys
from typing import Dict, List, Optional
import numpy as np

# Импорт из core
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core'))
from capital_request import CapitalComAPI

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class EnhancedSentimentAnalyzer:
    """Улучшенный анализатор настроений с несколькими источниками данных"""
    
    def __init__(self):
        self.marketaux_api_key = os.getenv("MARKETAUX_API_TOKEN")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")  # https://newsapi.org/
        #self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")  # https://www.alphavantage.co/
        
    def get_fear_greed_index(self) -> float:
        """Получение индекса страха и жадности от CNN"""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                value = float(data['data'][0]['value'])
                # Нормализуем: 0-25 = очень боязливо (0.0-0.25), 75-100 = очень жадно (0.75-1.0)
                normalized = value / 100.0
                logger.info(f"Fear & Greed Index: {value} (normalized: {normalized:.3f})")
                return normalized
            else:
                logger.warning("Не удалось получить Fear & Greed Index")
                return 0.5
                
        except Exception as e:
            logger.error(f"Ошибка при получении Fear & Greed Index: {e}")
            return 0.5
    
    def get_news_sentiment_newsapi(self, hours_back: int = 24) -> float:
        """Получение sentiment новостей через NewsAPI"""
        try:
            if not self.newsapi_key:
                logger.warning("NewsAPI key не найден")
                return 0.5

            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(hours=hours_back)

            params = {
                'q': 'gold OR DXY OR VIX', # Запрос по ключевым словам
                'language': 'en',
                'sortBy': 'relevancy',
                'from': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'to': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'pageSize': 50, # Максимальное количество статей на запрос
                'apiKey': self.newsapi_key
            }

            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])

                if articles:
                    # Используем нашу внутреннюю функцию для анализа sentiment по ключевым словам
                    sentiment = self._analyze_sentiment_keywords(articles)
                    logger.info(f"NewsAPI Sentiment: {sentiment:.3f} from {len(articles)} articles")
                    return sentiment
                else:
                    logger.warning("Нет статей из NewsAPI по заданным параметрам")
                    return 0.5
            else:
                logger.warning(f"NewsAPI ошибка: {response.status_code} - {response.text}")
                return 0.5

        except Exception as e:
            logger.error(f"Ошибка при получении news sentiment из NewsAPI: {e}")
            return 0.5

    def get_news_sentiment_marketaux(self, hours_back: int = 24) -> float:
        """Получение sentiment новостей через MarketAux"""
        try:
            if not self.marketaux_api_key:
                logger.warning("MarketAux API key не найден")
                return 0.5
                
            # Временной диапазон
            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(hours=hours_back)
            
            params = {
                'api_token': self.marketaux_api_key,
                'symbols': 'XAUUSD,DXY,SPY',  # Золото, доллар, SP500
                'filter_entities': 'true',
                'language': 'en',
                'published_after': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'published_before': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'limit': 50
            }
            
            response = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                
                if articles:
                    all_entity_sentiment_scores = []
                    for article in articles:
                        entities = article.get('entities', [])
                        for entity in entities:
                            score = entity.get('sentiment_score')
                            if score is not None:
                                all_entity_sentiment_scores.append(score)

                    if all_entity_sentiment_scores:
                        avg_sentiment = np.mean(all_entity_sentiment_scores)
                        normalized_sentiment = (avg_sentiment + 1) / 2
                        logger.info(f"MarketAux News Sentiment: {avg_sentiment:.3f} (normalized: {normalized_sentiment:.3f}) from {len(all_entity_sentiment_scores)} entity sentiments")
                        return normalized_sentiment
                    else:
                        logger.warning("Нет статей или сущностей с sentiment scores в MarketAux")
                        return 0.5
                else:
                    logger.warning("Нет статей из MarketAux по заданным параметрам") # Добавлено для ясности
                    return 0.5
                
        except Exception as e:
            logger.error(f"Ошибка при получении news sentiment из MarketAux: {e}")
            return 0.5
    
    def _analyze_sentiment_keywords(self, articles: List[Dict]) -> float:
        """Простой анализ sentiment по ключевым словам"""
        positive_words = [
            'bullish', 'rise', 'gain', 'growth', 'surge', 'rally', 'boom', 'optimistic',
            'strong', 'increase', 'up', 'higher', 'advance', 'recovery', 'positive'
        ]
        
        negative_words = [
            'bearish', 'fall', 'drop', 'decline', 'crash', 'sell', 'pessimistic',
            'weak', 'decrease', 'down', 'lower', 'retreat', 'recession', 'negative'
        ]
        
        total_score = 0
        total_articles = 0
        
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count > 0 or negative_count > 0:
                # Нормализуем счет от -1 до 1, затем к 0-1
                article_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
                total_score += (article_score + 1) / 2  # Нормализуем к 0-1
                total_articles += 1
        
        return total_score / max(total_articles, 1)
    
    def get_vix_based_sentiment(self, capital_api: CapitalComAPI) -> float:
        """Получение sentiment на основе VIX (индекс волатильности)"""
        try:
            # Получаем текущие данные VIX
            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(hours=48)
            
            vix_data = capital_api.get_historical_prices(
                epic="VIX",
                resolution="HOUR",
                max_points=48,
                from_date=from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                to_date=to_date.strftime('%Y-%m-%dT%H:%M:%S')
            )
            
            if vix_data and 'prices' in vix_data and vix_data['prices']:
                # Берем последнее значение VIX
                latest_vix = float(vix_data['prices'][-1]['closePrice']['bid'])
                
                # VIX интерпретация:
                # < 20: низкая волатильность, высокий оптимизм (0.7-1.0)
                # 20-30: умеренная волатильность, нейтральный sentiment (0.4-0.7)
                # > 30: высокая волатильность, страх (0.0-0.4)
                
                if latest_vix < 20:
                    sentiment = 0.7 + (20 - latest_vix) / 20 * 0.3  # 0.7-1.0
                elif latest_vix < 30:
                    sentiment = 0.4 + (30 - latest_vix) / 10 * 0.3  # 0.4-0.7
                else:
                    sentiment = max(0.0, 0.4 - (latest_vix - 30) / 20 * 0.4)  # 0.0-0.4
                
                logger.info(f"VIX-based sentiment: {sentiment:.3f} (VIX: {latest_vix:.2f})")
                return sentiment
            else:
                logger.warning("Не удалось получить данные VIX")
                return 0.5
                
        except Exception as e:
            logger.error(f"Ошибка при получении VIX sentiment: {e}")
            return 0.5
    
    def get_combined_sentiment(self, capital_api: CapitalComAPI = None) -> Dict[str, float]:
        """Получение комбинированного sentiment из всех источников"""
        logger.info("🧠 Получение данных sentiment из всех источников...")
        
        # Получаем данные из разных источников
        fear_greed = self.get_fear_greed_index()
        news_marketaux = self.get_news_sentiment_marketaux()
        news_newsapi = self.get_news_sentiment_newsapi()
        
        # VIX sentiment (если есть Capital API)
        vix_sentiment = 0.5
        if capital_api:
            vix_sentiment = self.get_vix_based_sentiment(capital_api)
        
        # Комбинируем все источники с весами
        weights = {
            'fear_greed': 0.3,
            'news_marketaux': 0.3,
            'news_newsapi': 0.2,
            'vix': 0.2
        }
        
        combined_sentiment = (
            fear_greed * weights['fear_greed'] +
            news_marketaux * weights['news_marketaux'] +
            news_newsapi * weights['news_newsapi'] +
            vix_sentiment * weights['vix']
        )
        
        sentiment_data = {
            'fear_greed': fear_greed,
            'news_marketaux': news_marketaux,
            'news_newsapi': news_newsapi,
            'vix_sentiment': vix_sentiment,
            'combined_sentiment': combined_sentiment,
            'timestamp': datetime.now()
        }
        
        logger.info(f"✅ Combined Sentiment: {combined_sentiment:.3f}")
        logger.info(f"   - Fear & Greed: {fear_greed:.3f}")
        logger.info(f"   - MarketAux News: {news_marketaux:.3f}")
        logger.info(f"   - NewsAPI: {news_newsapi:.3f}")
        logger.info(f"   - VIX-based: {vix_sentiment:.3f}")
        
        return sentiment_data


class GoldTradingDataPipeline:
    """Улучшенный пайплайн для получения данных о торговле золотом"""
    
    def __init__(self):
        self.capital_api = CapitalComAPI(
            os.getenv("CAPITAL_API_KEY"),
            os.getenv("CAPITAL_IDENTIFIER"),
            os.getenv("CAPITAL_PASSWORD")
        )
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        
    def get_market_data_with_sentiment(self, hours_back: int = 48) -> pd.DataFrame:
        """Получение рыночных данных с sentiment анализом"""
        try:
            logger.info("🚀 Запуск улучшенного пайплайна данных...")
            
            # 1. Авторизация
            if not self.capital_api.login_and_get_tokens():
                logger.error("Не удалось авторизоваться в Capital.com")
                return pd.DataFrame()
            
            # 2. Получение данных о ценах
            instruments = ["GOLD", "DXY", "VIX"]
            price_dfs = []
            
            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(hours=hours_back)
            
            for instrument in instruments:
                logger.info(f"📊 Получение данных для {instrument}...")
                
                price_data = self.capital_api.get_historical_prices(
                    epic=instrument,
                    resolution="HOUR",
                    max_points=hours_back,
                    from_date=from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    to_date=to_date.strftime('%Y-%m-%dT%H:%M:%S')
                )
                
                if price_data and 'prices' in price_data:
                    df = self._convert_price_data_to_df(price_data, instrument)
                    if not df.empty:
                        price_dfs.append(df)
                
                time.sleep(1)  # Уважаем API limits
            
            if not price_dfs:
                logger.error("Не удалось получить данные о ценах")
                return pd.DataFrame()
            
            # 3. Объединение ценовых данных
            combined_df = pd.concat(price_dfs, axis=1)
            combined_df.sort_index(inplace=True)
            combined_df.ffill(inplace=True)
            
            # 4. Получение sentiment данных
            sentiment_data = self.sentiment_analyzer.get_combined_sentiment(self.capital_api)
            
            # 5. Добавление sentiment к данным
            for key, value in sentiment_data.items():
                if key != 'timestamp':
                    combined_df[key] = value
            
            # 6. Расчет технических индикаторов
            if 'GOLD_Close' in combined_df.columns:
                combined_df = self._add_technical_indicators(combined_df)
            
            logger.info(f"✅ Пайплайн завершен. Размер данных: {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {e}")
            return pd.DataFrame()
    
    def _convert_price_data_to_df(self, price_data: dict, instrument: str) -> pd.DataFrame:
        """Конвертация данных о ценах в DataFrame"""
        prices_list = []
        
        for price_point in price_data['prices']:
            if 'snapshotTime' in price_point and 'closePrice' in price_point:
                dt_aware = pd.to_datetime(price_point['snapshotTime']).tz_localize('UTC')
                close_price = float(price_point['closePrice']['bid'])
                
                prices_list.append({
                    'Datetime': dt_aware,
                    f'{instrument}_Close': close_price,
                })
        
        if not prices_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(prices_list)
        df.set_index('Datetime', inplace=True)
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        try:
            gold_close = df['GOLD_Close']
            
            # RSI
            df['RSI'] = self._calculate_rsi(gold_close)
            
            # Moving Averages
            df['SMA_20'] = gold_close.rolling(window=20).mean()
            df['EMA_12'] = gold_close.ewm(span=12).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = gold_close.rolling(window=bb_period).std()
            df['BB_Middle'] = gold_close.rolling(window=bb_period).mean()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Price changes
            df['Price_Change_1h'] = gold_close.pct_change()
            df['Price_Change_6h'] = gold_close.pct_change(6)
            
            logger.info("📈 Технические индикаторы добавлены")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при расчете технических индикаторов: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """Сохранение данных"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_gold_data_with_sentiment_{timestamp}.csv"
            
            filepath = os.path.join(os.path.dirname(__file__), filename)
            df.to_csv(filepath)
            logger.info(f"💾 Данные сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении: {e}")


def main():
    """Главная функция"""
    pipeline = GoldTradingDataPipeline()
    
    # Получаем данные за последние 48 часов
    result_df = pipeline.get_market_data_with_sentiment(hours_back=48)
    
    if not result_df.empty:
        print("\n📊 Последние данные:")
        print(result_df.tail(5)[['GOLD_Close', 'RSI', 'combined_sentiment', 'fear_greed']].round(4))
        
        # Сохраняем данные
        pipeline.save_data(result_df, "enhanced_gold_data_with_sentiment_hourly.csv")
        
        print(f"\n✅ Успешно обработано {len(result_df)} записей")
        print(f"Временной диапазон: {result_df.index.min()} - {result_df.index.max()}")
    else:
        print("❌ Не удалось получить данные")


if __name__ == "__main__":
    main()