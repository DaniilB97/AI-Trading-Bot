#!/usr/bin/env python3
"""
Улучшенная GRU стратегия с Optuna оптимизацией и дополнительными фильтрами
"""
import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import argparse
import os
from dotenv import load_dotenv
import optuna
from optuna.samplers import TPESampler
import logging

from backtesting import Backtest, Strategy
from train_gru_scalper import ScalpingGRU, FixedParams
from main_files.config_loader import config

logger = config.logger
load_dotenv()

def fetch_backtest_data_binance(ticker_symbol="ETHUSDT", interval_str="5m", days_back=59):
    """Функция для получения данных из Binance (без изменений)"""
    api_key = os.getenv("BINANCE_API")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError("BINANCE_API or BINANCE_SECRET_KEY not found in .env file.")

    client = Client(api_key, api_secret)
    logger.info(f"Fetching Binance data for {ticker_symbol}, interval {interval_str}, {days_back} days back...")

    interval_map = {
        "1m": Client.KLINE_INTERVAL_1MINUTE, "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE, "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE, "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR, "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR, "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR, "1d": Client.KLINE_INTERVAL_1DAY,
        "3d": Client.KLINE_INTERVAL_3DAY, "1w": Client.KLINE_INTERVAL_1WEEK,
        "1M": Client.KLINE_INTERVAL_1MONTH
    }
    binance_interval = interval_map.get(interval_str, Client.KLINE_INTERVAL_5MINUTE)
    
    start_str = (datetime.utcnow() - timedelta(days=days_back)).strftime("%d %b, %Y %H:%M:%S")
    klines = client.get_historical_klines(ticker_symbol, binance_interval, start_str)
    
    if not klines:
        raise ValueError(f"No data fetched from Binance for {ticker_symbol}.")

    df = pd.DataFrame(klines, columns=[
        'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'CloseTime', 'QuoteAssetVolume', 'NumberofTrades', 
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
    ])

    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df.set_index('Datetime', inplace=True)
    
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[cols_to_numeric]
    df.dropna(inplace=True)

    logger.info(f"Binance data fetched. Shape: {df.shape}")
    return df

class ImprovedGRUScalpingStrategy(Strategy):
    """Улучшенная стратегия с дополнительными фильтрами и оптимизацией"""
    
    # Параметры для оптимизации
    rsi_period = 24
    sequence_length = 35
    
    # Пороги для сигналов
    buy_threshold = 0.55
    sell_threshold = 0.45
    
    # Дополнительные фильтры
    min_confidence_diff = 0.05  # Минимальная разница между порогами
    volatility_filter = True
    trend_filter = True
    volume_filter = True
    
    # Управление рисками
    max_trades_per_day = 50
    min_trade_duration = 2  # минимальная длительность сделки в барах
    max_drawdown_stop = 0.10  # стоп при просадке 10%
    
    # Размер позиции
    trade_size_percent_of_equity = 0.05

    use_sl_tp = True
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 4.0
    
    # Пути к модели
    model_path = "models/gru_scalper_model_ETH-USD_latest.pth"
    scaler_path = "models/gru_scalper_X_scaler_ETH-USD_best_optuna.joblib"

    def init(self):
        self.device = torch.device("cpu")
        
        # Загрузка
        try:
            self.scaler = joblib.load(self.scaler_path)
            best_gru_params = {'hidden_dim': 64, 'num_layers': 3, 'dropout_prob': 0.102}
            self.model = ScalpingGRU(input_dim=2, **best_gru_params).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise

        # --- РАСЧЕТ ИНДИКАТОРОВ ---
        self.data_df = pd.DataFrame(index=self.data.index)
        for col in ['Close', 'High', 'Low', 'Volume']:
            self.data_df[col] = getattr(self.data, col)
        
        # Основные индикаторы (для текущей модели)
        self.rsi_col_name = f'RSI_{self.rsi_period}'
        self.data_df[self.rsi_col_name] = ta.rsi(self.data_df['Close'], length=self.rsi_period)
        self.data_df['VWAP'] = ta.vwap(self.data_df['High'], self.data_df['Low'], self.data_df['Close'], self.data_df['Volume'])
        
        # Индикатор ATR (для SL/TP и фильтра волатильности)
        self.data_df['ATR'] = ta.atr(self.data_df['High'], self.data_df['Low'], self.data_df['Close'], length=14)

        # НОВЫЕ ИНДИКАТОРЫ (для будущего переобучения модели)
        stoch = ta.stoch(self.data_df['High'], self.data_df['Low'], self.data_df['Close'])
        self.data_df['STOCHk'] = stoch['STOCHk_14_3_3']
        self.data_df['CCI'] = ta.cci(self.data_df['High'], self.data_df['Low'], self.data_df['Close'])
        self.data_df['Price_Change'] = self.data_df['Close'].pct_change(periods=5)

        # Фильтры
        if self.trend_filter:
            self.data_df['EMA_21'] = ta.ema(self.data_df['Close'], length=21)
            self.data_df['EMA_50'] = ta.ema(self.data_df['Close'], length=50)
        if self.volatility_filter:
            self.data_df['ATR_MA'] = self.data_df['ATR'].rolling(window=20).mean()
        if self.volume_filter:
            self.data_df['Volume_MA'] = self.data_df['Volume'].rolling(window=20).mean()
        
        self.data_df.dropna(inplace=True)
        
        # ВАЖНО: Сейчас масштабируются только старые признаки, т.к. модель обучена на них.
        # При переобучении модели, сюда нужно будет добавить новые признаки.
        feature_columns_to_scale = [self.rsi_col_name, 'VWAP']
        if not self.data_df.empty and all(col in self.data_df.columns for col in feature_columns_to_scale):
            self.data_df[feature_columns_to_scale] = self.scaler.transform(self.data_df[feature_columns_to_scale])
        
        # Состояние
        self.trades_today = 0
        self.last_trade_bar = -999
        self.current_day = None
        self.initial_equity = self.equity
        self.peak_equity = self.equity

    def apply_filters(self, prediction_prob):
        """Применение дополнительных фильтров"""
        current_bar_time = self.data.index[-1]
        
        if current_bar_time not in self.data_df.index:
            return False
            
        current_data = self.data_df.loc[current_bar_time]
        
        # Фильтр тренда
        if self.trend_filter:
            if 'EMA_21' in current_data and 'EMA_50' in current_data:
                trend_up = current_data['EMA_21'] > current_data['EMA_50']
                # Для покупки нужен восходящий тренд, для продажи - нисходящий
                if prediction_prob > self.buy_threshold and not trend_up:
                    return False
                elif prediction_prob < self.sell_threshold and trend_up:
                    return False
        
        # Фильтр волатильности
        if self.volatility_filter:
            if 'ATR' in current_data and 'ATR_MA' in current_data:
                # Торгуем только при нормальной волатильности
                if current_data['ATR'] > 2 * current_data['ATR_MA']:
                    return False
        
        # Фильтр объема
        if self.volume_filter:
            if 'Volume_MA' in current_data:
                # Торгуем только при достаточном объеме
                if self.data.Volume[-1] < 0.5 * current_data['Volume_MA']:
                    return False
        
        return True

    def check_risk_management(self):
        """Проверка управления рисками"""
        # Обновление пикового капитала
        if self.initial_equity is None:
            self.initial_equity = self.equity
            self.peak_equity = self.equity
        else:
            self.peak_equity = max(self.peak_equity, self.equity)
        
        # Стоп по просадке
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if current_drawdown > self.max_drawdown_stop:
            if self.position:
                self.position.close()
            return False
        
        # Ограничение на количество сделок в день
        current_day = self.data.index[-1].date()
        if self.current_day != current_day:
            self.current_day = current_day
            self.trades_today = 0
        
        if self.trades_today >= self.max_trades_per_day:
            return False
        
        # Минимальная длительность между сделками
        current_bar = len(self.data) - 1
        if current_bar - self.last_trade_bar < self.min_trade_duration:
            return False
        
        return True

    def next(self):
        # Проверка рисков и достаточного кол-ва данных
        if len(self.data.Close) < self.sequence_length or not self.check_risk_management():
            return
        
        # Получаем данные для текущего бара
        current_bar_time = self.data.index[-1]
        if current_bar_time not in self.data_df.index:
            return

        # Получаем предсказание от модели
        try:
            current_loc = self.data_df.index.get_loc(current_bar_time)
            if current_loc < self.sequence_length - 1: return
            
            start_loc = current_loc - self.sequence_length + 1
            sequence_data = self.data_df[[self.rsi_col_name, 'VWAP']].iloc[start_loc:current_loc + 1].values
        except (KeyError, Exception):
            return

        if sequence_data.shape[0] != self.sequence_length: return

        sequence_tensor = torch.from_numpy(sequence_data).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction_prob = self.model(sequence_tensor).item()

        # Применяем фильтры и проверяем уверенность
        if not self.apply_filters(prediction_prob) or abs(prediction_prob - 0.5) < self.min_confidence_diff:
            return

        # --- НОВАЯ ЛОГИКА SL/TP ---
        current_price = self.data.Close[-1]
        atr_value = self.data_df['ATR'].loc[current_bar_time]
        position_size = self.trade_size_percent_of_equity
        
        # --- ЛОГИКА ТОРГОВЛИ ---
        if not self.position: # Если нет открытой позиции
            if prediction_prob > self.buy_threshold:
                if self.use_sl_tp:
                    sl = current_price - self.sl_atr_multiplier * atr_value
                    tp = current_price + self.tp_atr_multiplier * atr_value
                    self.buy(size=position_size, sl=sl, tp=tp)
                else:
                    self.buy(size=position_size)
                
                self.trades_today += 1
                self.last_trade_bar = len(self.data) - 1

            elif prediction_prob < self.sell_threshold:
                if self.use_sl_tp:
                    sl = current_price + self.sl_atr_multiplier * atr_value
                    tp = current_price - self.tp_atr_multiplier * atr_value
                    self.sell(size=position_size, sl=sl, tp=tp)
                else:
                    self.sell(size=position_size)

                self.trades_today += 1
                self.last_trade_bar = len(self.data) - 1
        else: 
            # Если позиция уже есть, закрываем по обратному сигналу
            # (backtesting.py сам закроет по SL/TP, если цена их достигнет)
            if self.position.is_long and prediction_prob < self.sell_threshold:
                self.position.close()
            elif self.position.is_short and prediction_prob > self.buy_threshold:
                self.position.close()

def objective(trial, data, base_cash, base_commission, leverage):
    """Optuna objective function для оптимизации параметров."""
    
    # Оптимизируем старые параметры
    params = {
        'buy_threshold': trial.suggest_float('buy_threshold', 0.51, 0.85, step=0.01),
        'sell_threshold': trial.suggest_float('sell_threshold', 0.15, 0.49, step=0.01),
        'min_confidence_diff': trial.suggest_float('min_confidence_diff', 0.02, 0.20, step=0.01),
        'trade_size_percent_of_equity': trial.suggest_float('trade_size_percent', 0.02, 0.25, step=0.01),
        'max_trades_per_day': trial.suggest_int('max_trades_per_day', 10, 100, step=5),
        'min_trade_duration': trial.suggest_int('min_trade_duration', 1, 10),
        'max_drawdown_stop': trial.suggest_float('max_drawdown_stop', 0.05, 0.30, step=0.01),
        'volatility_filter': trial.suggest_categorical('volatility_filter', [True, False]),
        'trend_filter': trial.suggest_categorical('trend_filter', [True, False]),
        'volume_filter': trial.suggest_categorical('volume_filter', [True, False]),
        # НОВЫЕ ПАРАМЕТРЫ для SL/TP
        'use_sl_tp': trial.suggest_categorical('use_sl_tp', [True, False]),
    }
    
    # Добавляем множители SL/TP только если они используются
    if params['use_sl_tp']:
        params['sl_atr_multiplier'] = trial.suggest_float('sl_atr_multiplier', 0.5, 5.0, step=0.1)
        params['tp_atr_multiplier'] = trial.suggest_float('tp_atr_multiplier', 1.0, 10.0, step=0.1)
    else:
        # Если не используется, ставим значения по умолчанию (они не будут влиять)
        params['sl_atr_multiplier'] = 2.0
        params['tp_atr_multiplier'] = 4.0

    # Применение всех параметров к классу стратегии
    for param, value in params.items():
        setattr(ImprovedGRUScalpingStrategy, param, value)
    
    # Запуск бэктеста
    try:
        bt = Backtest(data, ImprovedGRUScalpingStrategy, cash=base_cash, commission=base_commission, margin=1/leverage, trade_on_close=True)
        stats = bt.run()
        
        # Расчет метрики (остается без изменений)
        return_pct = stats.get('Return [%]', -999)
        sharpe = stats.get('Sharpe Ratio', -999)
        max_drawdown = abs(stats.get('Max. Drawdown [%]', 100))
        win_rate = stats.get('Win Rate [%]', 0)
        num_trades = stats.get('# Trades', 0)
        
        if num_trades < 10 or return_pct < -50 or max_drawdown > 50:
            return -99999
        
        return (return_pct * 0.4) + (sharpe * 10 * 0.3) + ((100 - max_drawdown) * 0.2) + (win_rate * 0.1)
        
    except Exception:
        return -99999

def run_optimization(data, n_trials, n_jobs=-1, base_cash=10000, base_commission=0.00075, leverage=20.0):
    """Запуск оптимизации с Optuna."""
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Используем все доступные ядра процессора
    effective_n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    logger.info(f"Starting optimization with {n_trials} trials, using {effective_n_jobs} jobs...")
    
    # Оборачиваем objective, чтобы передать данные и другие параметры
    objective_wrapper = lambda trial: objective(trial, data, base_cash, base_commission, leverage)
    
    study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs)
    
    return study

def run_final_backtest(data, best_params, cash=10000, commission=0.00075, leverage=20.0):
    """Запуск финального бэктеста с лучшими параметрами"""
    
    # Применение лучших параметров
    for param, value in best_params.items():
        setattr(ImprovedGRUScalpingStrategy, param, value)
    
    margin_ratio = 1 / leverage
    
    bt = Backtest(data, ImprovedGRUScalpingStrategy,
                 cash=cash, commission=commission,
                 margin=margin_ratio, trade_on_close=True)
    
    stats = bt.run()
    
    return bt, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Профессиональный воркфлоу для GRU стратегии")
    parser.add_argument("--ticker", type=str, default="ETHUSDT", help="Ticker symbol")
    parser.add_argument("--days_back", type=int, default=180, help="Общее кол-во дней для всей истории")
    parser.add_argument("--interval", type=str, default="5m", help="Data interval")
    parser.add_argument("--cash", type=float, default=10000, help="Initial cash")
    parser.add_argument("--commission", type=float, default=0.00075, help="Commission per trade")
    parser.add_argument("--leverage", type=float, default=20.0, help="Leverage")
    parser.add_argument("--optimize", action="store_true", help="Запустить полный цикл с оптимизацией")
    parser.add_argument("--n_trials", type=int, default=1000, help="Количество тестов для оптимизации")
    args = parser.parse_args()
    
    try:
        # --- ПОДГОТОВКА: ПОЛУЧЕНИЕ И РАЗДЕЛЕНИЕ ДАННЫХ ---
        full_data = fetch_backtest_data_binance(
            ticker_symbol=args.ticker.replace('-', ''),
            interval_str=args.interval,
            days_back=args.days_back
        )
        
        if len(full_data) < 1000:
            raise ValueError("Not enough data to perform Train/Validation/Test split.")

        # Разделяем данные: 70% на обучение, 15% на валидацию, 15% на тест
        train_size = int(len(full_data) * 0.7)
        validation_size = int(len(full_data) * 0.15)
        
        train_data = full_data.iloc[:train_size]
        validation_data = full_data.iloc[train_size : train_size + validation_size]
        test_data = full_data.iloc[train_size + validation_size:]

        logger.info(f"Data split into three sets:")
        logger.info(f"  Training Set:   {len(train_data)} bars from {train_data.index[0]} to {train_data.index[-1]}")
        logger.info(f"  Validation Set: {len(validation_data)} bars from {validation_data.index[0]} to {validation_data.index[-1]}")
        logger.info(f"  Test Set:       {len(test_data)} bars from {test_data.index[0]} to {test_data.index[-1]}")

        if args.optimize:
            # --- ЭТАП 1: ОПТИМИЗАЦИЯ на тренировочных данных ---
            logger.info("\n--- STAGE 1: Starting Optimization on TRAINING data ---")
            study = run_optimization(
                train_data, 
                n_trials=args.n_trials, 
                n_jobs=-1,  # Используем все ядра
                base_cash=args.cash, 
                base_commission=args.commission, 
                leverage=args.leverage
            )
            
            # Получаем 5 лучших результатов
            top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
            logger.info("Optimization finished. Top 5 trials found:")
            for i, trial in enumerate(top_trials):
                 logger.info(f"  Top {i+1}: Trial #{trial.number} with value {trial.value:.2f}")

            # --- ЭТАП 2: ВАЛИДАЦИЯ на валидационных данных ---
            logger.info("\n--- STAGE 2: Validating top 5 trials on VALIDATION data ---")
            validation_results = []
            for trial in top_trials:
                logger.info(f"  Validating Trial #{trial.number}...")
                bt, stats = run_final_backtest(validation_data, trial.params, args.cash, args.commission, args.leverage)
                
                return_pct = stats.get('Return [%]', -999)
                validation_results.append({'trial': trial, 'return_pct': return_pct})
                logger.info(f"    => Trial #{trial.number} Validation Return: {return_pct:.2f}%")

            if not validation_results:
                raise ValueError("Validation failed, no trials to compare.")
            
            # Выбираем чемпиона по результатам на валидации
            champion_info = max(validation_results, key=lambda x: x['return_pct'])
            champion_trial = champion_info['trial']
            
            logger.info(f"\n--- VALIDATION COMPLETE ---")
            logger.info(f"CHAMPION is Trial #{champion_trial.number} with validation return of {champion_info['return_pct']:.2f}%")
            logger.info(f"Champion parameters: {champion_trial.params}")

            # --- ЭТАП 3: ФИНАЛЬНЫЙ ТЕСТ на тестовых данных ---
            logger.info("\n--- STAGE 3: Final backtest of champion strategy on unseen TEST data ---")
            bt_final, stats_final = run_final_backtest(test_data, champion_trial.params, args.cash, args.commission, args.leverage)

            logger.info("\n--- FINAL HONEST TEST RESULTS ---")
            logger.info(stats_final)
            
            plot_filename = f"reports/final_test_backtest_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path("reports").mkdir(parents=True, exist_ok=True)
            if bt_final:
                bt_final.plot(filename=plot_filename, open_browser=False)
                logger.info(f"Final backtest plot saved to {plot_filename}")

        else:
            logger.warning("Script is run without --optimize flag. No action taken.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)