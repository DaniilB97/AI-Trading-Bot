# extended_data_pipeline.py - Получение 3+ лет данных для обучения

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
import pickle

# Импорт из core
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core'))
from core.capital_request import CapitalComAPI

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class ExtendedDataPipeline:
    """Пайплайн для получения максимально возможного объема данных"""
    
    def __init__(self):
        self.capital_api = CapitalComAPI(
            os.getenv("CAPITAL_API_KEY"),
            os.getenv("CAPITAL_IDENTIFIER"),
            os.getenv("CAPITAL_PASSWORD")
        )
        self.marketaux_api_key = os.getenv("MARKETAUX_API_TOKEN")
        
        # Целевые инструменты для максимального объема данных
        self.instruments = {
            'primary': ['GOLD'],  # Основной актив
            'correlations': ['DXY', 'VIX', 'SPX500', 'EUR/USD', 'GBP/USD'],  # Коррелирующие активы
            'alternatives': ['SILVER', 'OIL', 'NATGAS', 'COPPER']  # Альтернативные товары
        }
        
    def get_maximum_historical_data(self, years_back: int = 3) -> pd.DataFrame:
        """Получение максимального объема исторических данных"""
        logger.info(f"🚀 Запуск сбора {years_back} лет исторических данных...")
        
        # Авторизация
        if not self.capital_api.login_and_get_tokens():
            logger.error("❌ Ошибка авторизации Capital.com")
            return pd.DataFrame()
        
        # Временные диапазоны - максимально возможные
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=years_back * 365)
        
        all_data_frames = []
        
        # Получаем данные по всем инструментам
        all_instruments = []
        for category, instruments in self.instruments.items():
            all_instruments.extend(instruments)
        
        for instrument in all_instruments:
            logger.info(f"📊 Обработка {instrument}...")
            
            try:
                # Получаем данные большими кусками (по 6 месяцев)
                instrument_df = self._get_instrument_data_in_chunks(
                    instrument, start_date, end_date, chunk_months=6
                )
                
                if not instrument_df.empty:
                    all_data_frames.append(instrument_df)
                    logger.info(f"   ✅ {instrument}: {len(instrument_df)} записей")
                else:
                    logger.warning(f"   ⚠️ {instrument}: нет данных")
                    
            except Exception as e:
                logger.error(f"   ❌ Ошибка {instrument}: {e}")
                continue
        
        if not all_data_frames:
            logger.error("❌ Не получено никаких данных")
            return pd.DataFrame()
        
        # Объединяем все данные
        logger.info("🔄 Объединение всех данных...")
        combined_df = pd.concat(all_data_frames, axis=1)
        combined_df.sort_index(inplace=True)
        
        # Forward fill для выходных и праздников
        combined_df = combined_df.ffill()
        
        logger.info(f"✅ Собрано {len(combined_df)} записей за {years_back} лет")
        return combined_df
    
    def _get_instrument_data_in_chunks(self, instrument: str, start_date: datetime, 
                                       end_date: datetime, chunk_months: int = 6) -> pd.DataFrame:
        """Получение данных по инструменту большими кусками"""
        
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_months * 30), end_date)
            
            logger.info(f"   📥 {instrument}: {current_start.date()} → {current_end.date()}")
            
            try:
                # Максимальное количество точек за период
                days_in_chunk = (current_end - current_start).days
                max_points = min(days_in_chunk * 24, 1000)  # Capital.com limit
                
                price_data = self.capital_api.get_historical_prices(
                    epic=instrument,
                    resolution="HOUR",  # Часовые данные для максимальной детализации
                    max_points=max_points,
                    from_date=current_start.strftime('%Y-%m-%dT%H:%M:%S'),
                    to_date=current_end.strftime('%Y-%m-%dT%H:%M:%S')
                )
                
                if price_data and 'prices' in price_data and price_data['prices']:
                    chunk_df = self._convert_price_data_to_df(price_data, instrument)
                    if not chunk_df.empty:
                        chunks.append(chunk_df)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка получения куска {instrument}: {e}")
                time.sleep(5)  # Больше задержка при ошибке
            
            current_start = current_end
        
        if chunks:
            # Объединяем куски и убираем дубликаты
            combined = pd.concat(chunks)
            combined = combined[~combined.index.duplicated(keep='first')]
            combined.sort_index(inplace=True)
            return combined
        else:
            return pd.DataFrame()
    
    def _convert_price_data_to_df(self, price_data: dict, instrument: str) -> pd.DataFrame:
        """Конвертация в DataFrame с расширенными данными"""
        rows = []
        
        for price_point in price_data['prices']:
            try:
                dt = pd.to_datetime(price_point['snapshotTime']).tz_localize('UTC')
                
                # Извлекаем все доступные данные
                row_data = {
                    'Datetime': dt,
                    f'{instrument}_Open': float(price_point.get('openPrice', {}).get('ask', 0)),
                    f'{instrument}_High': float(price_point.get('highPrice', {}).get('ask', 0)),
                    f'{instrument}_Low': float(price_point.get('lowPrice', {}).get('ask', 0)),
                    f'{instrument}_Close': float(price_point.get('closePrice', {}).get('ask', 0)),
                }
                
                # Добавляем bid/ask если доступны
                if 'closePrice' in price_point:
                    if 'bid' in price_point['closePrice']:
                        row_data[f'{instrument}_Bid'] = float(price_point['closePrice']['bid'])
                    if 'ask' in price_point['closePrice']:
                        row_data[f'{instrument}_Ask'] = float(price_point['closePrice']['ask'])
                
                # Объем если доступен
                if 'lastTradedVolume' in price_point:
                    row_data[f'{instrument}_Volume'] = float(price_point['lastTradedVolume'])
                
                rows.append(row_data)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки точки {instrument}: {e}")
                continue
        
        if rows:
            df = pd.DataFrame(rows)
            df.set_index('Datetime', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def get_extended_news_data(self, years_back: int = 3) -> pd.DataFrame:
        """Получение максимального объема новостных данных"""
        logger.info(f"📰 Сбор {years_back} лет новостных данных...")
        
        if not self.marketaux_api_key:
            logger.error("❌ MarketAux API key не найден")
            return pd.DataFrame()
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=years_back * 365)
        
        all_articles = []
        
        # Получаем данные кусками по 1 месяцу (лимиты API)
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=30), end_date)
            
            logger.info(f"📥 Новости: {current_start.date()} → {current_end.date()}")
            
            try:
                params = {
                    'api_token': self.marketaux_api_key,
                    'symbols': 'XAUUSD,DXY,SPY,GLD',  # Золото и связанные
                    'filter_entities': 'true',
                    'language': 'en',
                    'published_after': current_start.strftime('%Y-%m-%dT%H:%M:%S'),
                    'published_before': current_end.strftime('%Y-%m-%dT%H:%M:%S'),
                    'limit': 100
                }
                
                response = requests.get(
                    "https://api.marketaux.com/v1/news/all", 
                    params=params, 
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('data', [])
                    
                    if articles:
                        all_articles.extend(articles)
                        logger.info(f"   ✅ Получено {len(articles)} статей")
                    else:
                        logger.info(f"   ⚪ Нет статей за период")
                        
                elif response.status_code == 429:
                    logger.warning("⚠️ Rate limit - ждем...")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"⚠️ API ошибка: {response.status_code}")
                
                # Rate limiting
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"❌ Ошибка получения новостей: {e}")
                time.sleep(10)
            
            current_start = current_end
        
        # Преобразуем в DataFrame
        if all_articles:
            news_data = []
            for article in all_articles:
                try:
                    news_data.append({
                        'datetime': pd.to_datetime(article['published_at']),
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'sentiment_score': article.get('sentiment_score', 0.0),
                        'source': article.get('source', ''),
                        'entities': str(article.get('entities', [])),
                        'url': article.get('url', '')
                    })
                except Exception as e:
                    logger.warning(f"Ошибка обработки статьи: {e}")
                    continue
            
            if news_data:
                news_df = pd.DataFrame(news_data)
                news_df.set_index('datetime', inplace=True)
                news_df.sort_index(inplace=True)
                
                logger.info(f"✅ Обработано {len(news_df)} новостных записей")
                return news_df
        
        logger.warning("⚠️ Новостные данные не получены")
        return pd.DataFrame()
    
    def create_master_dataset(self, years_back: int = 3) -> pd.DataFrame:
        """Создание мастер-датасета для обучения"""
        logger.info("🔥 Создание МАСТЕР-ДАТАСЕТА для переобучения...")
        
        # 1. Получаем максимум ценовых данных
        price_df = self.get_maximum_historical_data(years_back)
        if price_df.empty:
            logger.error("❌ Не удалось получить ценовые данные")
            return pd.DataFrame()
        
        # 2. Получаем максимум новостных данных
        news_df = self.get_extended_news_data(years_back)
        
        # 3. Создаем технические индикаторы
        enhanced_df = self._add_comprehensive_technical_indicators(price_df)
        
        # 4. Добавляем новостной sentiment
        if not news_df.empty:
            sentiment_df = self._aggregate_news_sentiment(news_df)
            final_df = enhanced_df.join(sentiment_df, how='left')
        else:
            logger.warning("⚠️ Новости не получены, используем только ценовые данные")
            final_df = enhanced_df
            final_df['news_sentiment'] = 0.5  # Нейтральный sentiment
        
        # 5. Финальная обработка
        final_df = self._finalize_master_dataset(final_df)
        
        logger.info(f"🎯 МАСТЕР-ДАТАСЕТ ГОТОВ:")
        logger.info(f"   📊 Записей: {len(final_df):,}")
        logger.info(f"   📈 Колонок: {len(final_df.columns)}")
        logger.info(f"   ⏰ Период: {final_df.index.min()} → {final_df.index.max()}")
        logger.info(f"   💾 Размер: {final_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        return final_df
    
    def _add_comprehensive_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление полного набора технических индикаторов"""
        logger.info("📈 Расчет технических индикаторов...")
        
        if 'GOLD_Close' not in df.columns:
            logger.error("❌ Нет данных по золоту")
            return df
        
        gold = df['GOLD_Close']
        
        # Базовые индикаторы
        df['RSI_14'] = self._calculate_rsi(gold, 14)
        df['RSI_21'] = self._calculate_rsi(gold, 21)
        
        # Скользящие средние разных периодов
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = gold.rolling(window=period, min_periods=1).mean()
            df[f'EMA_{period}'] = gold.ewm(span=period, min_periods=1).mean()
        
        # MACD
        ema_12 = gold.ewm(span=12, min_periods=1).mean()
        ema_26 = gold.ewm(span=26, min_periods=1).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = gold.rolling(window=period, min_periods=1).mean()
            std = gold.rolling(window=period, min_periods=1).std()
            df[f'BB_Upper_{period}'] = sma + (std * 2)
            df[f'BB_Lower_{period}'] = sma - (std * 2)
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            df[f'BB_Position_{period}'] = (gold - df[f'BB_Lower_{period}']) / df[f'BB_Width_{period}']
        
        # Изменения цены
        for period in [1, 4, 24, 168]:  # 1ч, 4ч, 1д, 1нед
            df[f'Price_Change_{period}h'] = gold.pct_change(period)
            df[f'Price_Return_{period}h'] = np.log(gold / gold.shift(period))
        
        # Волатильность
        for period in [24, 168]:  # 1д, 1нед
            df[f'Volatility_{period}h'] = gold.rolling(window=period, min_periods=1).std()
        
        # Корреляции с другими активами
        if 'DXY_Close' in df.columns:
            for period in [24, 168]:
                df[f'GOLD_DXY_Corr_{period}h'] = gold.rolling(period, min_periods=1).corr(df['DXY_Close'])
        
        if 'VIX_Close' in df.columns:
            for period in [24, 168]:
                df[f'GOLD_VIX_Corr_{period}h'] = gold.rolling(period, min_periods=1).corr(df['VIX_Close'])
        
        # Временные признаки
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Лаги
        for lag in [1, 2, 4, 8, 24]:
            df[f'GOLD_Lag_{lag}'] = gold.shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI_14'].shift(lag)
        
        logger.info(f"✅ Добавлено {len([col for col in df.columns if col not in ['GOLD_Close']])} технических индикаторов")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _aggregate_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация новостного sentiment по времени"""
        logger.info("📰 Агрегация новостного sentiment...")
        
        # Ресемплируем новости по часам
        hourly_sentiment = news_df.resample('h').agg({
            'sentiment_score': ['mean', 'std', 'count'],
        }).fillna(0)
        
        # Упрощаем колонки
        hourly_sentiment.columns = ['news_sentiment_mean', 'news_sentiment_std', 'news_count']
        
        # Скользящие средние для sentiment
        for period in [6, 24, 168]:  # 6ч, 1д, 1нед
            hourly_sentiment[f'news_sentiment_{period}h'] = (
                hourly_sentiment['news_sentiment_mean']
                .rolling(window=period, min_periods=1)
                .mean()
            )
        
        # Заполняем пропуски
        hourly_sentiment = hourly_sentiment.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Агрегировано sentiment для {len(hourly_sentiment)} часов")
        return hourly_sentiment
    
    def _finalize_master_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Финальная обработка датасета"""
        logger.info("🔧 Финальная обработка датасета...")
        
        # Убираем строки без основных данных
        df = df.dropna(subset=['GOLD_Close'])
        
        # Заполняем NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Убираем выбросы (3 sigma)
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'GOLD' in col and 'Close' in col:
                mean = df[col].mean()
                std = df[col].std()
                df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        
        # Сортируем по времени
        df.sort_index(inplace=True)
        
        logger.info(f"✅ Финальный датасет: {len(df)} записей, {len(df.columns)} колонок")
        return df
    
    def save_master_dataset(self, df: pd.DataFrame, filename: str = None):
        """Сохранение мастер-датасета"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"master_training_dataset_{timestamp}"
        
        # Сохраняем в нескольких форматах
        csv_path = f"{filename}.csv"
        pickle_path = f"{filename}.pkl"
        
        # CSV для читаемости
        df.to_csv(csv_path)
        logger.info(f"💾 CSV сохранен: {csv_path}")
        
        # Pickle для быстрой загрузки
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"💾 Pickle сохранен: {pickle_path}")
        
        # Статистика
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'start_date': str(df.index.min()),
            'end_date': str(df.index.max()),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_list': list(df.columns)
        }
        
        stats_path = f"{filename}_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"📊 Статистика сохранена: {stats_path}")


def main():
    """Главная функция для создания мастер-датасета"""
    pipeline = ExtendedDataPipeline()
    
    # Создаем датасет за 3 года
    master_df = pipeline.create_master_dataset(years_back=3)
    
    if not master_df.empty:
        # Сохраняем
        pipeline.save_master_dataset(master_df)
        
        print("\n" + "="*60)
        print("🎯 МАСТЕР-ДАТАСЕТ ДЛЯ ПЕРЕОБУЧЕНИЯ ГОТОВ!")
        print("="*60)
        print(f"📊 Записей: {len(master_df):,}")
        print(f"📈 Признаков: {len(master_df.columns)}")
        print(f"⏰ Период: {master_df.index.min()} → {master_df.index.max()}")
        print("\n📋 Топ признаки:")
        for i, col in enumerate(master_df.columns[:10], 1):
            print(f"  {i}. {col}")
        if len(master_df.columns) > 10:
            print(f"  ... и еще {len(master_df.columns) - 10} признаков")
        
        print("\n✅ Готов для переобучения RL модели!")
    else:
        print("❌ Не удалось создать датасет")


if __name__ == "__main__":
    main()