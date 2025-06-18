#!/usr/bin/env python3
"""
Скрипт для тренировки Random Forest модели для предсказания направления цены ETH.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import argparse
import joblib # For saving scikit-learn models
import yfinance as yf # Import yfinance
import pandas_ta as ta # Import pandas_ta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import MinMaxScaler # RF doesn't strictly need it, can be added if desired

from config_loader import config
from eth_analysis_system import TelegramParser, NewsRelevanceClassifier, TechnicalAnalyzer # Assuming TechnicalAnalyzer is in eth_analysis_system

logger = config.logger

class RandomForestTrainer:
    """Класс для тренировки Random Forest модели"""

    def __init__(self):
        self.config = config
        self.logger = logger
        self.technical_feature_names = [] # Will be populated dynamically
        self.news_feature_names = [
            'message_count', 'avg_sentiment', 'max_impact', 
            'total_views', 'total_reactions'
        ]
        self.setup_directories()

    def setup_directories(self):
        """Создание необходимых директорий"""
        dirs = ['models', 'data/raw', 'data/processed', 'logs/training_rf']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    async def collect_telegram_data(self, days_back=30):
        """Сбор данных из Telegram"""
        self.logger.info("📱 Начинаем сбор данных из Telegram для Random Forest...")
        news_cache_path = Path(f"data/raw/telegram_news_rf_{days_back}d.csv")
        if news_cache_path.exists():
            self.logger.info(f"📂 Найден кэш новостей: {news_cache_path}, загружаем...")
            news_df = pd.read_csv(news_cache_path)
            news_df['date'] = pd.to_datetime(news_df['date'])
            news_df['text'] = news_df['text'].astype(str) # Ensure text is string
            return news_df

        telegram_config = self.config.get_telegram_config()
        channels = self.config.get_telegram_channels()
        parser = TelegramParser(**telegram_config)
        await parser.connect()
        try:
            news_df = await parser.parse_multiple_channels(channels, days_back=days_back, limit=500)
            if not news_df.empty:
                news_df['text'] = news_df['text'].astype(str)
                news_classifier = NewsRelevanceClassifier()
                impacts = [news_classifier.analyze_impact(text_content) for text_content in news_df['text']]
                news_df['sentiment_score'] = [i['sentiment_score'] for i in impacts]
                news_df['impact_level'] = [i['impact_level'] for i in impacts]
                news_df['impact_score'] = news_df['sentiment_score'].abs() 
                news_df.to_csv(news_cache_path, index=False)
                self.logger.info(f"💾 Сохранено {len(news_df)} новостей в кэш: {news_cache_path}")
            return news_df
        finally:
            await parser.close()

    def prepare_technical_data(self, days_back=60, interval="1h"):
        """Подготовка мастер-сета технических данных для Random Forest."""
        self.logger.info(f"📊 Подготовка мастер-сета технических данных ({interval} интервал) для Random Forest...")
        
        master_cache_filename = f"data/raw/eth_master_features_rf_{days_back}d_{interval}.csv"
        master_cache_path = Path(master_cache_filename)

        if master_cache_path.exists():
            self.logger.info(f"📂 Найден мастер-кэш технических данных: {master_cache_path}, загружаем...")
            df_master = pd.read_csv(master_cache_path, index_col=0, parse_dates=True)
            df_master.index = pd.to_datetime(df_master.index).tz_localize(None)
            base_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            self.technical_feature_names = sorted(list(set(
                [col for col in df_master.columns if col not in base_ohlcv + self.news_feature_names + ['target', 'future_close']]
            )))
            self.logger.info(f"Загружен мастер-кэш с {len(self.technical_feature_names)} техническими признаками.")
            return df_master
        
        self.logger.info("Мастер-кэш не найден. Генерируем полный набор признаков...")

        # 1. Fetch base ETH data
        eth_ticker = yf.Ticker("ETH-USD")
        df_eth_base = eth_ticker.history(period=f"{days_back}d", interval=interval)
        if df_eth_base.empty:
            self.logger.error("Не удалось загрузить данные ETH. Прерывание.")
            raise ValueError("ETH data could not be fetched.")
        df_eth_base.index = pd.to_datetime(df_eth_base.index).tz_localize(None)
        self.logger.info(f"💾 Загружены сырые данные ETH: {len(df_eth_base)} записей")
        
        df_master = df_eth_base.copy() # Start with base OHLCV

        # 2. Calculate broad set of indicators using TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        # TechnicalAnalyzer.calculate_indicators is expected to add columns to the df
        df_master = analyzer.calculate_indicators(df_master) 
        self.logger.info(f"💾 Рассчитаны индикаторы из TechnicalAnalyzer.")

        # 3. Calculate specific notebook indicators using pandas_ta
        required_pta_cols = ['High', 'Low', 'Close', 'Volume', 'Open']
        if not all(col in df_master.columns for col in required_pta_cols):
            missing_cols = [col for col in required_pta_cols if col not in df_master.columns]
            raise ValueError(f"Missing base columns for pandas_ta: {missing_cols}")

        df_master["VWAP_pta"] = ta.vwap(df_master['High'], df_master['Low'], df_master['Close'], df_master['Volume'])
        df_master['RSI_16_pta'] = ta.rsi(df_master['Close'], length=16)
        bbands_pta = ta.bbands(df_master['Close'], length=14, std=2.0)
        df_master = df_master.join(bbands_pta[['BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0']]) # These names are unique
        df_master['ATRr_7_pta'] = ta.atr(df_master['High'], df_master['Low'], df_master['Close'], length=7)

        backcandles = 15
        VWAPsignal = np.zeros(len(df_master), dtype=int)
        if 'VWAP_pta' not in df_master.columns or df_master['VWAP_pta'].isnull().all():
            self.logger.warning("VWAP_pta (pandas_ta) не рассчитан или все NaN, VWAPSignal_pta будет 0.")
            df_master['VWAPSignal_pta'] = 0
        else:
            for row in range(backcandles, len(df_master)):
                upt = 1; dnt = 1
                window_slice_vwap = df_master['VWAP_pta'].iloc[row-backcandles:row+1]
                window_slice_open = df_master['Open'].iloc[row-backcandles:row+1]
                window_slice_close = df_master['Close'].iloc[row-backcandles:row+1]
                if window_slice_vwap.isnull().any() or window_slice_open.isnull().any() or window_slice_close.isnull().any(): continue
                for i_idx in range(len(window_slice_open)):
                    if max(window_slice_open.iloc[i_idx], window_slice_close.iloc[i_idx]) >= window_slice_vwap.iloc[i_idx]: dnt=0
                    if min(window_slice_open.iloc[i_idx], window_slice_close.iloc[i_idx]) <= window_slice_vwap.iloc[i_idx]: upt=0
                if upt==1 and dnt==1: VWAPsignal[row]=3
                elif upt==1: VWAPsignal[row]=2
                elif dnt==1: VWAPsignal[row]=1
            df_master['VWAPSignal_pta'] = VWAPsignal

        TotSignal = np.zeros(len(df_master), dtype=int)
        required_ts_cols = ['VWAPSignal_pta', 'Close', 'BBL_14_2.0', 'RSI_16_pta', 'BBU_14_2.0']
        if not all(col in df_master.columns for col in required_ts_cols) or any(df_master[col].isnull().all() for col in required_ts_cols):
            self.logger.warning(f"Одна или несколько колонок для TotalSignal_pta отсутствуют или все NaN: {required_ts_cols}. TotalSignal_pta будет 0.")
            df_master['TotalSignal_pta'] = 0
        else:
            for row_idx in range(backcandles, len(df_master)):
                signal_val = 0
                if df_master[required_ts_cols].iloc[row_idx].isnull().any(): continue
                if (df_master['VWAPSignal_pta'].iloc[row_idx]==2 and df_master['Close'].iloc[row_idx] <= df_master['BBL_14_2.0'].iloc[row_idx] and df_master['RSI_16_pta'].iloc[row_idx] < 45): signal_val = 2
                elif (df_master['VWAPSignal_pta'].iloc[row_idx]==1 and df_master['Close'].iloc[row_idx] >= df_master['BBU_14_2.0'].iloc[row_idx] and df_master['RSI_16_pta'].iloc[row_idx] > 55): signal_val = 1
                TotSignal[row_idx] = signal_val
            df_master['TotalSignal_pta'] = TotSignal
        self.logger.info("💾 Рассчитаны индикаторы и сигналы из ноутбука (pandas_ta) с суффиксом _pta.")

        # 4. Fetch BTC data
        btc_ticker = yf.Ticker("BTC-USD")
        df_btc = btc_ticker.history(period=f"{days_back}d", interval=interval)
        if df_btc.empty: 
            self.logger.warning("Не удалось загрузить данные BTC. Признаки BTC будут нулевыми.")
            df_btc = pd.DataFrame(index=df_master.index)
        else: 
            df_btc.index = pd.to_datetime(df_btc.index).tz_localize(None)
        
        df_btc['BTC_Close'] = df_btc.get('Close', pd.Series(index=df_btc.index, dtype=float))
        df_btc['BTC_Price_change'] = df_btc['BTC_Close'].pct_change()
        
        # 5. Merge master ETH df with BTC data
        df_master = df_master.join(df_btc[['BTC_Close', 'BTC_Price_change']], how='left') # rsuffix='_btc' removed as BTC columns are unique
        
        # 6. Calculate ETH-BTC correlation
        if 'Price_change' not in df_master.columns and 'Close' in df_master.columns: # Price_change for ETH
             df_master['Price_change'] = df_master['Close'].pct_change() 
        
        if 'Price_change' in df_master.columns and 'BTC_Price_change' in df_master.columns:
            df_master['Price_change'] = df_master['Price_change'].fillna(0)
            df_master['BTC_Price_change'] = df_master['BTC_Price_change'].fillna(0)
            df_master['ETH_BTC_corr_3h'] = df_master['Price_change'].rolling(window=3, min_periods=1).corr(df_master['BTC_Price_change'])
            df_master['ETH_BTC_corr_3h'] = df_master['ETH_BTC_corr_3h'].fillna(0)
        else:
            self.logger.warning("Не удалось рассчитать ETH-BTC корреляцию.")
            df_master['ETH_BTC_corr_3h'] = 0

        # 7. Add time-based features
        df_master['hour_of_day'] = df_master.index.hour
        df_master['day_of_week'] = df_master.index.dayofweek
        
        # 8. Fill NaNs robustly
        # Columns that might have leading NaNs due to rolling calculations or pct_change
        cols_to_fill_na = [
            'Price_change', 'VWAP_pta', 'RSI_16_pta', 'BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0', 'ATRr_7_pta',
            'VWAPSignal_pta', 'TotalSignal_pta', 'BTC_Close', 'BTC_Price_change', 'ETH_BTC_corr_3h'
        ]
        # Also include all columns from TechnicalAnalyzer (excluding OHLCV)
        ta_analyzer_cols = [col for col in df_master.columns if col not in df_eth_base.columns and col not in cols_to_fill_na]
        cols_to_fill_na.extend(ta_analyzer_cols)

        for col in cols_to_fill_na:
            if col in df_master.columns:
                df_master[col] = df_master[col].ffill().bfill().fillna(0)
        
        # 9. Dynamically populate self.technical_feature_names
        base_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        df_master = df_master.dropna(axis=1, how='all') # Drop columns that are entirely NaN (e.g. 'Dividends')
        
        self.technical_feature_names = sorted(list(set(
            [col for col in df_master.columns if col not in base_ohlcv + self.news_feature_names + ['target', 'future_close']]
        )))
        self.logger.info(f"Итоговый набор технических признаков ({len(self.technical_feature_names)}): {self.technical_feature_names}")

        df_master.to_csv(master_cache_path)
        self.logger.info(f"💾 Сохранен мастер-сет технических данных: {len(df_master)} записей в {master_cache_path}")
        return df_master

    def aggregate_news_features_by_hour(self, news_df: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
        if news_df is None or news_df.empty:
            return pd.DataFrame(0, index=price_index, columns=self.news_feature_names)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize(None)
        news_df_hourly_agg = []
        for ts_hour_start in price_index:
            ts_hour_end = ts_hour_start + timedelta(hours=1)
            hour_news = news_df[(news_df['date'] >= ts_hour_start) & (news_df['date'] < ts_hour_end)]
            agg_features = {fn: 0 for fn in self.news_feature_names}
            if not hour_news.empty:
                agg_features.update({
                    'message_count': len(hour_news),
                    'avg_sentiment': hour_news['sentiment_score'].mean(),
                    'max_impact': hour_news['impact_score'].abs().max(),
                    'total_views': hour_news['views'].sum(),
                    'total_reactions': hour_news['reactions'].sum()
                })
            agg_features['timestamp_hour_start'] = ts_hour_start
            news_df_hourly_agg.append(agg_features)
        aggregated_df = pd.DataFrame(news_df_hourly_agg)
        if not aggregated_df.empty:
            aggregated_df = aggregated_df.set_index('timestamp_hour_start')
        else:
             return pd.DataFrame(0, index=price_index, columns=self.news_feature_names)
        return aggregated_df.reindex(price_index).fillna(0)[self.news_feature_names]

    def create_features_and_target(self, tech_df: pd.DataFrame, news_df_hourly_agg: pd.DataFrame, prediction_horizon: int = 1):
        self.logger.info(f"Создание признаков и целевой переменной (горизонт {prediction_horizon} час(а))...")
        # self.technical_feature_names should be populated by prepare_technical_data
        if not self.technical_feature_names:
             raise ValueError("self.technical_feature_names is empty. Run prepare_technical_data first or load from cache.")

        existing_tech_features = [col for col in self.technical_feature_names if col in tech_df.columns]
        missing_tech_features = [col for col in self.technical_feature_names if col not in tech_df.columns]
        if missing_tech_features:
            self.logger.warning(f"Следующие технические признаки из self.technical_feature_names отсутствуют в tech_df и будут пропущены: {missing_tech_features}")
        
        # Ensure 'Close' is in tech_df for target calculation, even if not a feature itself
        cols_for_subset = existing_tech_features + ['Close'] if 'Close' not in existing_tech_features else existing_tech_features
        df_tech_subset = tech_df[list(set(cols_for_subset))].copy() # Use set to avoid duplicate 'Close'

        df_tech_subset.index = pd.to_datetime(df_tech_subset.index).tz_localize(None)
        news_df_hourly_agg.index = pd.to_datetime(news_df_hourly_agg.index).tz_localize(None)
        
        combined_df = df_tech_subset.join(news_df_hourly_agg, how='left')
        
        combined_df['future_close'] = combined_df['Close'].shift(-prediction_horizon)
        combined_df['target'] = (combined_df['future_close'] > combined_df['Close']).astype(int)
        
        # Features to use for X are only existing_tech_features + news_feature_names
        feature_columns_for_X = [col for col in (existing_tech_features + self.news_feature_names) if col in combined_df.columns]
        
        # Columns to check for NaNs before dropping: all features used for X plus the target
        dropna_subset_cols = feature_columns_for_X + ['target']
        combined_df.dropna(subset=dropna_subset_cols, inplace=True)
        
        X = combined_df[feature_columns_for_X]
        y = combined_df['target']
        
        self.logger.info(f"Подготовлено {len(X)} образцов для Random Forest. Признаки X: {X.columns.tolist()}")
        return X, y

    def train_evaluate_rf(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("Тренировка Random Forest...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        self.logger.info(f"Размеры данных: Train={len(X_train)}, Test={len(X_test)}")
        param_grid = {
            'n_estimators': [100], 
            'max_depth': [5, 8, 12], 
            'min_samples_split': [20, 50, 100], 
            'min_samples_leaf': [10, 20, 50]
        }
        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=None, verbose=1, scoring='f1_weighted') # n_jobs=None for GridSearchCV itself if sub-jobs are -1
        grid_search.fit(X_train, y_train)
        best_rf_model = grid_search.best_estimator_
        self.logger.info(f"Лучшие параметры Random Forest: {grid_search.best_params_}")
        y_pred_train = best_rf_model.predict(X_train)
        y_pred_test = best_rf_model.predict(X_test)
        self.logger.info("--- Результаты на Тренировочных данных ---")
        self.logger.info(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        self.logger.info(classification_report(y_train, y_pred_train, zero_division=0))
        self.logger.info("--- Результаты на Тестовых данных ---")
        self.logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        self.logger.info(classification_report(y_test, y_pred_test, zero_division=0))
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"\n{confusion_matrix(y_test, y_pred_test)}")
        
        if not X.empty:
            feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            self.logger.info("\n--- Важность признаков ---")
            self.logger.info(f"\n{feature_importances.head(30)}") # Show more features
        else:
            self.logger.warning("DataFrame X пуст, важность признаков не может быть рассчитана.")

        model_filename = f"models/eth_random_forest_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(best_rf_model, model_filename)
        self.logger.info(f"Модель Random Forest сохранена: {model_filename}")
        latest_model_path = "models/eth_random_forest_latest.joblib"
        joblib.dump(best_rf_model, latest_model_path)
        self.logger.info(f"Модель Random Forest сохранена как последняя: {latest_model_path}")
        return best_rf_model

async def main_rf_feature_selection_loop():
    parser = argparse.ArgumentParser(description='Train ETH Random Forest Model with Feature Selection')
    parser.add_argument('--days-back-tech', type=int, default=60, help='Days of historical technical data')
    parser.add_argument('--interval', type=str, default="1h", help='Data interval (e.g., 1h, 5m)')
    parser.add_argument('--skip-news', action='store_true', help='Skip news data collection')
    parser.add_argument('--prediction-horizon', type=int, default=1, help='Prediction horizon for target variable')
    parser.add_argument('--initial-features-to-keep', type=int, default=30, help='Number of top features to start with after initial run')
    parser.add_argument('--feature-pruning-iterations', type=int, default=5, help='Number of pruning iterations')
    parser.add_argument('--features-to_remove_per_iteration', type=int, default=3, help='Number of features to remove per iteration')


    args = parser.parse_args()
    logger.info("🚀 Запуск тренировки Random Forest модели для ETH с итеративным выбором признаков...")
    trainer = RandomForestTrainer()

    try:
        logger.info("Этап 1: Сбор и подготовка полного набора технических данных...")
        # Use asyncio.to_thread for synchronous self.prepare_technical_data
        full_tech_df = await asyncio.to_thread(trainer.prepare_technical_data, days_back=args.days_back_tech, interval=args.interval)
        full_tech_df.index = pd.to_datetime(full_tech_df.index).tz_localize(None)

        logger.info("Фокусируемся только на индикаторах, новостные признаки будут нулевыми.")
        news_data_hourly_agg = pd.DataFrame(0, index=full_tech_df.index, columns=trainer.news_feature_names)
        if not args.skip_news:
            logger.info("Этап 2 (опционально): Сбор и кэширование новостных данных...")
            await trainer.collect_telegram_data(days_back=7) # Using fixed 7 days for news cache for now

        # Initial full feature set for the first run
        current_technical_features = list(trainer.technical_feature_names) # Get all features from prepare_technical_data
        
        best_overall_f1_score = -1
        best_feature_subset = None
        best_model = None

        for iteration in range(args.feature_pruning_iterations + 1): # +1 for initial run with all features
            logger.info(f"\n===== Итерация выбора признаков {iteration + 1}/{args.feature_pruning_iterations + 1} =====")
            if not current_technical_features:
                logger.warning("Нет доступных технических признаков для тренировки.")
                break
            
            logger.info(f"Используется {len(current_technical_features)} признаков: {current_technical_features}")
            
            # Update trainer's list for create_features_and_target to use current subset
            trainer.technical_feature_names = current_technical_features 
            
            X, y = trainer.create_features_and_target(full_tech_df.copy(), news_data_hourly_agg.copy(), prediction_horizon=args.prediction_horizon)

            if X.empty or y.empty or X.shape[1] == 0:
                logger.error("❌ Не удалось создать датасет для обучения или нет признаков в X. Проверьте данные и параметры.")
                break
            
            logger.info(f"Тренировка модели с {X.shape[1]} признаками.")
            trained_model_iteration = trainer.train_evaluate_rf(X, y)
            
            # Evaluate based on test F1-score (macro avg)
            # Need to parse classification_report or get metrics from elsewhere
            # For simplicity, let's assume train_evaluate_rf could return test metrics
            # For now, we'll just track the last model. A proper loop would compare performance.
            # This part needs refinement to properly track the best model based on a validation metric.
            # For this example, we'll just use the features from the last successful run.
            
            # Get feature importances from the trained model for pruning
            if hasattr(trained_model_iteration, 'feature_importances_') and not X.empty :
                importances = pd.Series(trained_model_iteration.feature_importances_, index=X.columns)
                
                if iteration == 0 and args.initial_features_to_keep < len(importances) : # Initial Pruning
                     top_features = importances.sort_values(ascending=False).head(args.initial_features_to_keep).index.tolist()
                     current_technical_features = [f for f in top_features if f not in trainer.news_feature_names] # only prune tech features
                     logger.info(f"Первоначальная обрезка до {len(current_technical_features)} лучших признаков.")
                elif iteration < args.feature_pruning_iterations and len(current_technical_features) > args.features_to_remove_per_iteration : # Subsequent Pruning
                    features_to_remove_count = min(args.features_to_remove_per_iteration, len(current_technical_features) -1 ) # ensure at least 1 feature remains
                    if features_to_remove_count > 0:
                        least_important_features = importances.loc[current_technical_features].sort_values(ascending=True).head(features_to_remove_count).index.tolist()
                        current_technical_features = [f for f in current_technical_features if f not in least_important_features]
                        logger.info(f"Удалено {len(least_important_features)} наименее важных признаков: {least_important_features}")
                    else:
                        logger.info("Недостаточно признаков для дальнейшей обрезки.")
                        break

                else: # Last iteration or too few features
                    logger.info("Достигнуто максимальное количество итераций или минимальное количество признаков.")
                    break 
            else:
                logger.warning("Не удалось получить важность признаков, обрезка невозможна.")
                break
        
        logger.info(f"\n🏁 Итеративный выбор признаков завершен.")
        logger.info(f"Оптимальный набор (последней итерации): {current_technical_features}")
        # Here you would typically save the model trained with this 'best_feature_subset'
        # For now, the last saved model from train_evaluate_rf is considered 'latest'

    except Exception as e:
        logger.error(f"❌ Ошибка при тренировке Random Forest с выбором признаков: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main_rf_feature_selection_loop())
