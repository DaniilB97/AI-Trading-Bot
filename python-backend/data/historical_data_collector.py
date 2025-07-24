# historical_data_collector.py - Сбор исторических данных с Golden Cross

import os
import sys
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import time
import numpy as np
import yfinance as yf  # Для получения большого объема исторических данных
import pandas_ta as ta

# Добавляем путь к core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from capital_request import CapitalComAPI

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class HistoricalTradingDataCollector:
    """
    Мощный сборщик исторических данных для обучения RL модели
    
    Особенности:
    - 5+ лет исторических данных
    - Golden Cross MA(50) vs MA(200) на дневных данных
    - Часовые данные для основной логики
    - Реальные sentiment где возможно
    - Подготовка для train_rl_model.py
    """
    
    def __init__(self):
        self.capital_api = CapitalComAPI(
            os.getenv("CAPITAL_API_KEY"),
            os.getenv("CAPITAL_IDENTIFIER"),
            os.getenv("CAPITAL_PASSWORD")
        )
        self.marketaux_key = os.getenv("MARKETAUX_API_TOKEN")
        
        # Выходные файлы
        self.master_training_file = "master_training_dataset.csv"
        self.daily_data_file = "daily_data_with_golden_cross.csv"
        self.hourly_data_file = "hourly_data_with_sentiment.csv"
        
        logger.info("📊 Исторический сборщик данных инициализирован")
    
    def collect_historical_price_data(self, years: int = 5) -> dict:
        """
        Сбор исторических ценовых данных
        
        Args:
            years: Количество лет для сбора данных
            
        Returns:
            dict: Словарь с дневными и часовыми данными
        """
        logger.info(f"📈 Сбор {years} лет исторических данных...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Инструменты для сбора
        instruments = {
            'GOLD': 'GC=F',        # Gold Futures
            'DXY': 'DX=F',         # Dollar Index
            'VIX': '^VIX',         # Volatility Index
            'SPY': 'SPY',          # S&P 500 ETF
            'US10Y': '^TNX'        # 10-Year Treasury
        }
        
        daily_data = {}
        hourly_data = {}
        
        for instrument, yahoo_symbol in instruments.items():
            try:
                logger.info(f"  📊 Получение {instrument} ({yahoo_symbol})...")
                
                # Дневные данные (для Golden Cross)
                daily_ticker = yf.Ticker(yahoo_symbol)
                daily_df = daily_ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if not daily_df.empty:
                    # Переименовываем колонки
                    daily_df.columns = [f"{instrument}_{col}" for col in daily_df.columns]
                    daily_data[instrument] = daily_df
                    logger.info(f"    ✅ Дневные данные {instrument}: {len(daily_df)} записей")
                
                # Часовые данные (последние 2 года для производительности)
                hourly_start = end_date - timedelta(days=365 * 2)
                hourly_df = daily_ticker.history(
                    start=hourly_start.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1h'
                )
                
                if not hourly_df.empty:
                    hourly_df.columns = [f"{instrument}_{col}" for col in hourly_df.columns]
                    hourly_data[instrument] = hourly_df
                    logger.info(f"    ✅ Часовые данные {instrument}: {len(hourly_df)} записей")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"    ⚠️ Ошибка получения {instrument}: {e}")
                continue
        
        return {
            'daily': daily_data,
            'hourly': hourly_data
        }
    
    def calculate_golden_cross_signals(self, daily_df: pd.DataFrame, 
                                     price_column: str = 'GOLD_Close') -> pd.DataFrame:
        """
        Расчет Golden Cross сигналов на дневных данных
        
        Args:
            daily_df: DataFrame с дневными данными
            price_column: Колонка с ценой для расчета MA
            
        Returns:
            DataFrame с Golden Cross сигналами
        """
        logger.info("🎯 Расчет Golden Cross сигналов...")
        
        if price_column not in daily_df.columns:
            logger.error(f"❌ Колонка {price_column} не найдена")
            return daily_df
        
        df = daily_df.copy()
        
        # Рассчитываем скользящие средние
        df['MA_50'] = df[price_column].rolling(window=50, min_periods=1).mean()
        df['MA_200'] = df[price_column].rolling(window=200, min_periods=1).mean()
        
        # Golden Cross логика
        df['MA_50_above_200'] = (df['MA_50'] > df['MA_200']).astype(int)
        df['MA_50_above_200_prev'] = df['MA_50_above_200'].shift(1)
        
        # Определяем пересечения
        df['golden_cross_bullish'] = (
            (df['MA_50_above_200'] == 1) & 
            (df['MA_50_above_200_prev'] == 0)
        ).astype(int)
        
        df['golden_cross_bearish'] = (
            (df['MA_50_above_200'] == 0) & 
            (df['MA_50_above_200_prev'] == 1)
        ).astype(int)
        
        # Текущее состояние тренда
        df['golden_cross_trend'] = np.where(
            df['MA_50'] > df['MA_200'], 1,  # Bullish trend
            np.where(df['MA_50'] < df['MA_200'], -1, 0)  # Bearish trend
        )
        
        # Сила сигнала (расстояние между MA)
        df['ma_spread'] = (df['MA_50'] - df['MA_200']) / df['MA_200']
        df['ma_spread_normalized'] = (df['ma_spread'] - df['ma_spread'].mean()) / (df['ma_spread'].std() + 1e-9)
        
        # Статистика
        bullish_crosses = df['golden_cross_bullish'].sum()
        bearish_crosses = df['golden_cross_bearish'].sum()
        
        logger.info(f"✅ Golden Cross расчеты завершены:")
        logger.info(f"   📈 Bullish crosses: {bullish_crosses}")
        logger.info(f"   📉 Bearish crosses: {bearish_crosses}")
        logger.info(f"   📊 Текущий тренд: {df['golden_cross_trend'].iloc[-1]}")
        
        return df
    
    def add_technical_indicators_comprehensive(self, df: pd.DataFrame, 
                                             price_col: str = 'GOLD_Close') -> pd.DataFrame:
        """
        Добавление всех технических индикаторов для RL модели
        """
        logger.info("📊 Расчет технических индикаторов...")
        
        if price_col not in df.columns:
            logger.error(f"❌ Колонка {price_col} не найдена")
            return df
        
        # Используем pandas_ta для расчета индикаторов
        df.ta.rsi(close=price_col, length=14, append=True)
        df.ta.stoch(high=price_col, low=price_col, close=price_col, 
                   k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(high=price_col, low=price_col, close=price_col, 
                  length=14, append=True)
        df.ta.atr(high=price_col, low=price_col, close=price_col, 
                  length=14, append=True)
        
        # Дополнительные индикаторы
        df['Price_Change_1'] = df[price_col].pct_change(1)
        df['Price_Change_5'] = df[price_col].pct_change(5)
        df['Price_Change_24'] = df[price_col].pct_change(24) if len(df) > 24 else 0
        
        # Волатильность
        df['Volatility_24'] = df['Price_Change_1'].rolling(window=24, min_periods=1).std()
        
        # DXY и VIX изменения
        if 'DXY_Close' in df.columns:
            df['DXY_change'] = df['DXY_Close'].pct_change()
        if 'VIX_Close' in df.columns:
            df['VIX_change'] = df['VIX_Close'].pct_change()
        
        # Убираем NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info("✅ Технические индикаторы добавлены")
        return df
    
    def get_historical_sentiment_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание прокси sentiment для исторических данных
        
        Поскольку у нас нет исторических sentiment данных,
        создаем разумные прокси на основе:
        - VIX (страх/жадность)
        - Price momentum
        - Golden Cross состояние
        """
        logger.info("📰 Создание прокси sentiment для исторических данных...")
        
        # Базовый sentiment на основе VIX
        if 'VIX_Close' in df.columns:
            vix_norm = (df['VIX_Close'] - df['VIX_Close'].rolling(252, min_periods=1).mean()) / \
                      (df['VIX_Close'].rolling(252, min_periods=1).std() + 1e-9)
            vix_sentiment = -np.tanh(vix_norm * 0.5)  # VIX высокий = негативный sentiment
        else:
            vix_sentiment = 0
        
        # Momentum sentiment
        if 'Price_Change_5' in df.columns:
            momentum_sentiment = np.tanh(df['Price_Change_5'] * 10)  # Price growth = positive
        else:
            momentum_sentiment = 0
        
        # Golden Cross sentiment
        if 'golden_cross_trend' in df.columns:
            golden_cross_sentiment = df['golden_cross_trend'] * 0.3  # Долгосрочный тренд
        else:
            golden_cross_sentiment = 0
        
        # Комбинированный sentiment
        df['sentiment_proxy'] = (
            0.4 * vix_sentiment + 
            0.4 * momentum_sentiment + 
            0.2 * golden_cross_sentiment
        )
        
        # Нормализуем в диапазон [-1, 1]
        df['sentiment_proxy'] = np.clip(df['sentiment_proxy'], -1, 1)
        
        # Добавляем шум для реалистичности
        noise = np.random.normal(0, 0.05, len(df))
        df['sentiment'] = np.clip(df['sentiment_proxy'] + noise, -1, 1)
        
        logger.info(f"✅ Sentiment прокси создан (среднее: {df['sentiment'].mean():.3f})")
        return df
    
    def merge_daily_and_hourly_data(self, daily_data: dict, hourly_data: dict) -> pd.DataFrame:
        """
        Объединение дневных (Golden Cross) и часовых данных
        """
        logger.info("🔗 Объединение дневных и часовых данных...")
        
        # Объединяем дневные данные
        daily_combined = None
        for instrument, df in daily_data.items():
            if daily_combined is None:
                daily_combined = df
            else:
                daily_combined = daily_combined.join(df, how='outer')
        
        if daily_combined is None:
            logger.error("❌ Нет дневных данных")
            return pd.DataFrame()
        
        # Рассчитываем Golden Cross на дневных данных
        daily_combined = self.calculate_golden_cross_signals(daily_combined)
        
        # Объединяем часовые данные
        hourly_combined = None
        for instrument, df in hourly_data.items():
            if hourly_combined is None:
                hourly_combined = df
            else:
                hourly_combined = hourly_combined.join(df, how='outer')
        
        if hourly_combined is None:
            logger.error("❌ Нет часовых данных")
            return pd.DataFrame()
        
        # Ресэмплируем дневные Golden Cross сигналы в часовые
        daily_resampled = daily_combined[['golden_cross_trend', 'ma_spread_normalized', 
                                        'golden_cross_bullish', 'golden_cross_bearish']].resample('H').ffill()
        
        # Объединяем часовые данные с дневными сигналами
        final_df = hourly_combined.join(daily_resampled, how='left')
        final_df = final_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"✅ Данные объединены: {len(final_df)} часовых записей")
        return final_df
    
    def create_master_training_dataset(self, years: int = 5) -> pd.DataFrame:
        """
        ГЛАВНАЯ ФУНКЦИЯ: Создание мастер датасета для обучения
        """
        logger.info("🚀 СОЗДАНИЕ МАСТЕР ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ...")
        
        try:
            # 1. Сбор исторических данных
            price_data = self.collect_historical_price_data(years)
            
            if not price_data['daily'] or not price_data['hourly']:
                logger.error("❌ Не удалось получить ценовые данные")
                return pd.DataFrame()
            
            # 2. Объединение данных
            combined_df = self.merge_daily_and_hourly_data(
                price_data['daily'], 
                price_data['hourly']
            )
            
            if combined_df.empty:
                logger.error("❌ Не удалось объединить данные")
                return pd.DataFrame()
            
            # 3. Добавление технических индикаторов
            enriched_df = self.add_technical_indicators_comprehensive(combined_df)
            
            # 4. Создание sentiment прокси
            final_df = self.get_historical_sentiment_proxy(enriched_df)
            
            # 5. Подготовка финального датасета
            feature_columns = [
                'GOLD_Close', 'DXY_Close', 'VIX_Close',  # Основные цены
                'RSI_20', 'STOCHk_14_3_3', 'CCI_14_0.015', 'ATR_14',  # Технические индикаторы
                'Price_Change_1', 'Price_Change_5', 'Volatility_24',  # Momentum и волатильность
                'DXY_change', 'VIX_change',  # Межрыночные сигналы
                'golden_cross_trend', 'ma_spread_normalized',  # Golden Cross сигналы
                'sentiment'  # Sentiment
            ]
            
            # Фильтруем только нужные колонки
            available_columns = [col for col in feature_columns if col in final_df.columns]
            training_df = final_df[available_columns].copy()
            
            # Финальная очистка
            training_df = training_df.dropna()
            training_df = training_df[training_df.index.notna()]
            
            # Сохранение
            training_df.to_csv(self.master_training_file)
            
            logger.info("✅ МАСТЕР ДАТАСЕТ СОЗДАН!")
            logger.info(f"   📊 Записей: {len(training_df)}")
            logger.info(f"   📈 Временной диапазон: {training_df.index.min()} → {training_df.index.max()}")
            logger.info(f"   🎯 Фичи: {list(training_df.columns)}")
            logger.info(f"   💾 Файл: {self.master_training_file}")
            
            return training_df
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания мастер датасета: {e}")
            return pd.DataFrame()
    
    def get_dataset_statistics(self) -> dict:
        """Статистика созданного датасета"""
        try:
            if not os.path.exists(self.master_training_file):
                return {'error': 'Мастер датасет не найден'}
            
            df = pd.read_csv(self.master_training_file, index_col=0, parse_dates=True)
            
            stats = {
                'total_records': len(df),
                'date_range': f"{df.index.min()} → {df.index.max()}",
                'features': list(df.columns),
                'file_size_mb': os.path.getsize(self.master_training_file) / 1024 / 1024,
                'golden_cross_stats': {
                    'bullish_periods': (df['golden_cross_trend'] == 1).sum() if 'golden_cross_trend' in df.columns else 0,
                    'bearish_periods': (df['golden_cross_trend'] == -1).sum() if 'golden_cross_trend' in df.columns else 0,
                    'neutral_periods': (df['golden_cross_trend'] == 0).sum() if 'golden_cross_trend' in df.columns else 0
                },
                'sentiment_stats': {
                    'mean': df['sentiment'].mean() if 'sentiment' in df.columns else None,
                    'std': df['sentiment'].std() if 'sentiment' in df.columns else None,
                    'min': df['sentiment'].min() if 'sentiment' in df.columns else None,
                    'max': df['sentiment'].max() if 'sentiment' in df.columns else None
                }
            }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}


def main():
    """Главная функция"""
    collector = HistoricalTradingDataCollector()
    
    print("📊 ИСТОРИЧЕСКИЙ СБОРЩИК ДАННЫХ ДЛЯ RL ОБУЧЕНИЯ")
    print("=" * 60)
    print("1. 🚀 Создать мастер датасет (5+ лет)")
    print("2. 📊 Статистика существующего датасета")
    print("3. 🧪 Тест Golden Cross логики")
    print("4. 📈 Краткий датасет (2 года) для тестов")
    
    choice = input("\nВыберите действие (1-4): ").strip()
    
    if choice == "1":
        years = int(input("Сколько лет данных собрать? (рекомендуется 5): ") or "5")
        logger.info(f"🚀 Создание мастер датасета за {years} лет...")
        
        dataset = collector.create_master_training_dataset(years)
        
        if not dataset.empty:
            stats = collector.get_dataset_statistics()
            print("\n📊 РЕЗУЛЬТАТ:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
            
            print(f"\n✅ Готово! Файл: {collector.master_training_file}")
            print("📝 Теперь можно запустить train_rl_model.py")
        else:
            print("❌ Ошибка создания датасета")
    
    elif choice == "2":
        stats = collector.get_dataset_statistics()
        print("\n📊 СТАТИСТИКА ДАТАСЕТА:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    elif choice == "3":
        logger.info("🧪 Тест Golden Cross логики...")
        
        # Создаем тестовые данные
        test_data = collector.collect_historical_price_data(1)  # 1 год для теста
        if test_data['daily'].get('GOLD'):
            daily_df = test_data['daily']['GOLD']
            golden_cross_df = collector.calculate_golden_cross_signals(daily_df)
            
            print("\n🎯 GOLDEN CROSS АНАЛИЗ:")
            print(f"📊 Последние 10 дней:")
            print(golden_cross_df[['GOLD_Close', 'MA_50', 'MA_200', 'golden_cross_trend']].tail(10))
            
            recent_crosses = golden_cross_df[golden_cross_df['golden_cross_bullish'] == 1].tail(5)
            print(f"\n📈 Последние bullish crosses:")
            if not recent_crosses.empty:
                for date, row in recent_crosses.iterrows():
                    print(f"  {date.strftime('%Y-%m-%d')}: цена {row['GOLD_Close']:.2f}")
        else:
            print("❌ Не удалось получить данные для теста")
    
    elif choice == "4":
        logger.info("📈 Создание краткого датасета для тестов...")
        dataset = collector.create_master_training_dataset(2)  # 2 года
        
        if not dataset.empty:
            print(f"✅ Краткий датасет создан: {len(dataset)} записей")
            print(f"💾 Файл: {collector.master_training_file}")
        else:
            print("❌ Ошибка создания краткого датасета")
    
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    main()