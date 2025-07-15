#!/usr/bin/env python3
"""
Script to train an IMPROVED GRU model for ETH scalping strategy.
Features: RSI, VWAP, Stochastic, CCI, Price Change.
Target: Price direction in N future bars.
THIS VERSION IS OPTIMIZED TO FETCH DATA ONLY ONCE.
"""
import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, SequentialSampler
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import argparse
import optuna
import os
from dotenv import load_dotenv

from config_loader import config
logger = config.logger
load_dotenv()

def fetch_data_binance(ticker_symbol="ETHUSDT", interval_str="5m", days_back=180):
    """Функция для получения данных из Binance."""
    api_key = os.getenv("BINANCE_API")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError("BINANCE_API or BINANCE_SECRET_KEY not found in .env file.")

    client = Client(api_key, api_secret)
    logger.info(f"Fetching {days_back} days of historical data for {ticker_symbol}...")

    interval_map = {"5m": Client.KLINE_INTERVAL_5MINUTE}
    binance_interval = interval_map.get(interval_str, Client.KLINE_INTERVAL_5MINUTE)
    
    start_str = (datetime.utcnow() - timedelta(days=days_back)).strftime("%d %b, %Y %H:%M:%S")
    klines = client.get_historical_klines(ticker_symbol, binance_interval, start_str)
    
    if not klines: raise ValueError(f"No data fetched from Binance for {ticker_symbol}.")

    df = pd.DataFrame(klines, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df.set_index('Datetime', inplace=True)
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[cols_to_numeric].dropna()
    logger.info(f"Data fetched successfully. Shape: {df.shape}")
    return df

class FixedParams:
    def __init__(self, params):
        self.params = params; self.number = -1
    def suggest_int(self, name, low, high, **kwargs): return self.params[name]
    def suggest_float(self, name, low, high, **kwargs): return self.params[name]
    def suggest_categorical(self, name, choices): return self.params[name]
    def report(self, value, step): pass
    def should_prune(self): return False

class ScalpingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1, dropout_prob=0.2):
        super(ScalpingGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class GRUScalperTrainerV2:
    # --- ИЗМЕНЕНИЕ: Теперь класс принимает данные при инициализации ---
    def __init__(self, trial, data, ticker="ETHUSDT", interval="5m"):
        self.trial = trial
        self.data = data # Используем переданные данные
        self.ticker = ticker
        self.interval = interval
        
        self.rsi_period = self.trial.suggest_int("rsi_period", 10, 30)
        self.sequence_length = self.trial.suggest_int("sequence_length", 12, 48)
        self.prediction_horizon_bars = self.trial.suggest_int("prediction_horizon_bars", 1, 6)
        self.hidden_dim = self.trial.suggest_int("hidden_dim", 32, 128, step=16)
        self.num_layers = self.trial.suggest_int("num_layers", 1, 3)
        self.dropout_prob = self.trial.suggest_float("dropout_prob", 0.1, 0.5)
        self.learning_rate = self.trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        self.batch_size = self.trial.suggest_categorical("batch_size", [64, 128])
        self.epochs = self.trial.suggest_int("epochs", 30, 60, step=10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_cols = [f'RSI_{self.rsi_period}', 'VWAP', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5']
        self.target_col = 'Target'
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))
        Path("models").mkdir(parents=True, exist_ok=True)
        
    def _prepare_features_and_target(self):
        df_features = pd.DataFrame(index=self.data.index)
        df_features[f'RSI_{self.rsi_period}'] = ta.rsi(self.data['Close'], length=self.rsi_period)
        df_features['VWAP'] = ta.vwap(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])
        stoch = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'], k=14, d=3, smooth_k=3)
        df_features['STOCHk_14_3_3'] = stoch['STOCHk_14_3_3']
        df_features['CCI_14_0.015'] = ta.cci(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        df_features['Price_Change_5'] = self.data['Close'].pct_change(periods=5)
        df_features[self.target_col] = (self.data['Close'].shift(-self.prediction_horizon_bars) > self.data['Close']).astype(int)
        
        df_processed = df_features.copy().dropna()
        if df_processed.empty: raise ValueError("DataFrame empty after feature calculation.")
        return df_processed

    def _scale_and_create_sequences(self, processed_data):
        current_feature_cols = [col for col in self.feature_cols if col in processed_data.columns]
        X_data = processed_data[current_feature_cols].values
        y_data = processed_data[self.target_col].values.reshape(-1, 1)
        X_scaled = self.X_scaler.fit_transform(X_data)
        
        X_sequences, y_sequences = [], []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_sequences.append(X_scaled[i : i + self.sequence_length])
            y_sequences.append(y_data[i + self.sequence_length - 1])
        return np.array(X_sequences), np.array(y_sequences)

    # Функции get_dataloaders, train_model, evaluate_model остаются без изменений
    def get_dataloaders(self, X, y):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        indices = list(range(len(dataset)))
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SequentialSampler(indices[train_size : train_size + val_size])
        test_sampler = SequentialSampler(indices[train_size + val_size:])
        return (DataLoader(dataset, batch_size=self.batch_size, sampler=sampler) for sampler in [train_sampler, val_sampler, test_sampler])

    def train_model(self, model, train_loader, val_loader):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
        return model

    def evaluate_model(self, model, test_loader):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred_labels = (model(X_batch) > 0.5).float()
                all_preds.extend(y_pred_labels.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        return f1_score(all_targets, all_preds, zero_division=0)

    def run_training_pipeline(self, is_final_run=False):
        # --- ИЗМЕНЕНИЕ: _fetch_data() больше не вызывается ---
        df_processed = self._prepare_features_and_target()
        X_seq, y_seq = self._scale_and_create_sequences(df_processed)
        
        if X_seq.shape[0] < (self.batch_size * 3):
            logger.warning("Not enough data for trial. Skipping.")
            return 0.0
            
        train_loader, val_loader, test_loader = self.get_dataloaders(X_seq, y_seq)
        
        input_dim = X_seq.shape[2] 
        model_to_train = ScalpingGRU(input_dim, self.hidden_dim, self.num_layers, 1, self.dropout_prob).to(self.device)
        trained_model = self.train_model(model_to_train, train_loader, val_loader)
        f1 = self.evaluate_model(trained_model, test_loader)
        
        if is_final_run:
            joblib.dump(self.X_scaler, f"models/gru_scaler_v2_{self.ticker}.joblib")
            torch.save(trained_model.state_dict(), f"models/gru_model_v2_{self.ticker}.pth")
            logger.info("💾 Final v2 model and scaler saved.")
            
        return f1

# --- ИЗМЕНЕНИЕ: objective теперь принимает данные ---
def objective(trial, data, args_ns):
    try:
        trainer = GRUScalperTrainerV2(trial, data, ticker=args_ns.ticker, interval=args_ns.interval)
        return trainer.run_training_pipeline()
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=False)
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize IMPROVED GRU Model with Optuna")
    parser.add_argument("--ticker", type=str, default="ETHUSDT")
    parser.add_argument("--interval", type=str, default="5m")
    parser.add_argument("--days_back", type=int, default=180) 
    parser.add_argument("--n_trials", type=int, default=100) 
    args = parser.parse_args()

    # --- ИЗМЕНЕНИЕ: Загрузка данных происходит один раз здесь ---
    historical_data = fetch_data_binance(args.ticker, args.interval, args.days_back)

    study = optuna.create_study(direction="maximize")
    # --- ИЗМЕНЕНИЕ: Передаем данные в objective ---
    study.optimize(lambda trial: objective(trial, historical_data, args), n_trials=args.n_trials, n_jobs=-1)
    
    if study.best_trial:
        logger.info("\nRetraining final v2 model with best parameters...")
        best_params_trial_obj = FixedParams(study.best_trial.params)
        final_trainer = GRUScalperTrainerV2(
            best_params_trial_obj, 
            historical_data, 
            ticker=args.ticker, 
            interval=args.interval
        )
        final_trainer.run_training_pipeline(is_final_run=True)
        logger.info("Final v2 model training complete.")
