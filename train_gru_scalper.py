#!/usr/bin/env python3
"""
Script to train a GRU model for ETH scalping strategy with Optuna hyperparameter optimization.
Features: RSI (tunable period), VWAP on 5-minute data.
Target: Price direction in N future bars (tunable horizon).
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, SequentialSampler
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import argparse
import sys
import optuna

from main_files.config_loader import config
logger = config.logger

# Helper class for fixed params run (used if not running Optuna study)
class FixedParams:
    def __init__(self, params):
        self.params = params
        self.number = -1 # Indicates a special/final run

    def suggest_int(self, name, low, high, step=1, log=False):
        return self.params[name]

    def suggest_float(self, name, low, high, step=None, log=False):
        return self.params[name]

    def suggest_categorical(self, name, choices):
        return self.params[name]
    
    def report(self, value, step): pass 
    def should_prune(self): return False

# --- GRU Model Definition ---
class ScalpingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1, dropout_prob=0.2):
        super(ScalpingGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0) 
        out = self.fc(out[:, -1, :]) 
        out = self.sigmoid(out)
        return out

class GRUScalperTrainer: # Renamed class
    def __init__(self, trial, ticker="ETH-USD", interval="5m", days_back=59):
        self.trial = trial 
        self.ticker = ticker
        self.interval = interval
        self.days_back = days_back
        
        if isinstance(trial, optuna.trial.Trial) or isinstance(trial, FixedParams):
            self.rsi_period = self.trial.suggest_int("rsi_period", 10, 30)
            self.sequence_length = self.trial.suggest_int("sequence_length", 12, 48) 
            self.prediction_horizon_bars = self.trial.suggest_int("prediction_horizon_bars", 1, 6)
            self.hidden_dim = self.trial.suggest_int("hidden_dim", 32, 128, step=16)
            self.num_layers = self.trial.suggest_int("num_layers", 1, 3)
            self.dropout_prob = self.trial.suggest_float("dropout_prob", 0.1, 0.5)
            self.learning_rate = self.trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            self.batch_size = self.trial.suggest_categorical("batch_size", [32, 64, 128])
            self.epochs = self.trial.suggest_int("epochs", 20, 50, step=10)
        else: 
            self.rsi_period = trial.get("rsi_period", 20)
            self.sequence_length = trial.get("sequence_length", 24)
            self.prediction_horizon_bars = trial.get("prediction_horizon_bars", 3)
            self.hidden_dim = trial.get("hidden_dim", 50)
            self.num_layers = trial.get("num_layers", 2)
            self.dropout_prob = trial.get("dropout_prob", 0.2)
            self.learning_rate = trial.get("learning_rate", 0.001)
            self.batch_size = trial.get("batch_size", 64)
            self.epochs = trial.get("epochs", 50)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.feature_cols = [f'RSI_{self.rsi_period}', 'VWAP'] 
        self.target_col = 'Target'
        self.data_df = None
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))

        Path("models").mkdir(parents=True, exist_ok=True)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        
        run_type = "Optuna Trial" if isinstance(trial, optuna.trial.Trial) else "Fixed Params Run"
        trial_num_info = f"{self.trial.number}: " if hasattr(self.trial, 'number') and self.trial.number >=0 else ""
        logger.info(f"{run_type} {trial_num_info}Params - RSI: {self.rsi_period}, SeqLen: {self.sequence_length}, Horizon: {self.prediction_horizon_bars} bars, "
                    f"HiddenDim: {self.hidden_dim}, Layers: {self.num_layers}, Dropout: {self.dropout_prob:.2f}, LR: {self.learning_rate:.5f}, Batch: {self.batch_size}, Epochs: {self.epochs}")

    def _fetch_data(self):
        cache_filename = f"data/raw/gru_scalper_raw_data_{self.ticker.replace('=X','')}_{self.days_back}d_{self.interval}.csv" # Changed lstm to gru
        cache_path = Path(cache_filename)
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col='Datetime', parse_dates=['Datetime'])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        data = yf.download(self.ticker, period=f"{self.days_back}d", interval=self.interval)
        if data.empty: raise ValueError("No data fetched.")
        data.index = pd.to_datetime(data.index).tz_localize(None)
        if isinstance(data.columns, pd.MultiIndex) and len(data.columns.levels[1]) == 1:
            data.columns = data.columns.droplevel(1)
        if data.index.name is None: data.index.name = 'Datetime'
        if 'Volume' not in data.columns: data['Volume'] = 0
        elif data['Volume'].sum() == 0: logger.warning(f"Volume data for {self.ticker} is all zeros. VWAP might be NaN.")
        data.to_csv(cache_path)
        return data

    def _prepare_features_and_target(self, df):
        df_features = pd.DataFrame(index=df.index)
        df_features[f'RSI_{self.rsi_period}'] = ta.rsi(df['Close'], length=self.rsi_period)
        df_features['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df_features[self.target_col] = (df['Close'].shift(-self.prediction_horizon_bars) > df['Close']).astype(int)
        df_processed = df_features.copy()
        df_processed.dropna(inplace=True)
        if df_processed.empty: raise ValueError("DataFrame empty after feature calculation and NaN drop.")
        self.data_df = df_processed
        return self.data_df

    def _scale_and_create_sequences(self):
        current_feature_cols = [f'RSI_{self.rsi_period}', 'VWAP']
        X_data = self.data_df[current_feature_cols].values
        y_data = self.data_df[self.target_col].values.reshape(-1, 1)
        X_scaled = self.X_scaler.fit_transform(X_data)
        X_sequences, y_sequences = [], []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_sequences.append(X_scaled[i : i + self.sequence_length])
            y_sequences.append(y_data[i + self.sequence_length - 1])
        if not X_sequences: return np.array([]), np.array([])
        return np.array(X_sequences), np.array(y_sequences)

    def get_dataloaders(self, X, y):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.7 * len(dataset)); val_size = int(0.15 * len(dataset))
        train_indices = list(range(train_size)); val_indices = list(range(train_size, train_size + val_size)); test_indices = list(range(train_size + val_size, len(dataset)))
        train_sampler = SubsetRandomSampler(train_indices); val_sampler = SequentialSampler(val_indices); test_sampler = SequentialSampler(test_indices)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sampler)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler)
        return train_loader, val_loader, test_loader

    def train_model(self, model, train_loader, val_loader):
        criterion = nn.BCELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad(); y_pred = model(X_batch); loss = criterion(y_pred, y_batch); loss.backward(); optimizer.step()
            model.eval(); val_loss_epoch = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader: val_loss_epoch += criterion(model(X_batch), y_batch).item()
            avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0
            if hasattr(self.trial, 'report'): 
                self.trial.report(avg_val_loss, epoch)
                if self.trial.should_prune(): raise optuna.exceptions.TrialPruned()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                 trial_num_str = f"Trial {self.trial.number} - " if hasattr(self.trial, 'number') and self.trial.number >=0 else "Final Run - "
                 logger.debug(f"{trial_num_str}Epoch {epoch+1}/{self.epochs} => Val Loss: {avg_val_loss:.6f}")
        return model

    def evaluate_model(self, model, test_loader):
        model.eval(); all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred_labels = (model(X_batch) > 0.5).float()
                all_preds.extend(y_pred_labels.cpu().numpy()); all_targets.extend(y_batch.cpu().numpy())
        all_preds = np.array(all_preds).flatten(); all_targets = np.array(all_targets).flatten()
        f1 = f1_score(all_targets, all_preds, zero_division=0, average='weighted')
        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)
        trial_num_str = f"Trial {self.trial.number} - " if hasattr(self.trial, 'number') and self.trial.number >=0 else "Final Run - "
        logger.info(f"{trial_num_str}Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        if hasattr(self.trial, 'number') and self.trial.number == -1: 
            logger.info("Classification Report (Final Run):\n" + report); logger.info("Confusion Matrix (Final Run):\n" + str(cm))
        return f1, accuracy, report, cm

    def run_training_pipeline(self, is_final_run=False):
        df_raw = self._fetch_data()
        df_processed = self._prepare_features_and_target(df_raw)
        X_seq, y_seq = self._scale_and_create_sequences()
        if len(X_seq) == 0 or len(y_seq) == 0 or len(X_seq) != len(y_seq) or X_seq.shape[0] < (self.batch_size * 3):
            logger.warning(f"Trial {self.trial.number if hasattr(self.trial, 'number') else 'N/A'}: Not enough data. Skipping.")
            return 0.0 
        train_loader, val_loader, test_loader = self.get_dataloaders(X_seq, y_seq)
        input_dim = X_seq.shape[2]
        # Use ScalpingGRU model
        model_to_train = ScalpingGRU(input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout_prob=self.dropout_prob).to(self.device)
        trained_model = self.train_model(model_to_train, train_loader, val_loader)
        f1, _, _, _ = self.evaluate_model(trained_model, test_loader)
        if is_final_run:
            scaler_filename = f"models/gru_scalper_X_scaler_{self.ticker.replace('=X','')}_best_optuna.joblib" # gru
            joblib.dump(self.X_scaler, scaler_filename)
            logger.info(f"💾 BEST Optuna X_scaler saved to {scaler_filename}")
            model_filename = f"models/gru_scalper_model_{self.ticker.replace('=X','')}_best_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth" # gru
            torch.save(trained_model.state_dict(), model_filename)
            logger.info(f"💾 BEST Optuna GRU Scalper model saved to {model_filename}")
            latest_model_path = f"models/gru_scalper_model_{self.ticker.replace('=X','')}_latest.pth" # gru
            torch.save(trained_model.state_dict(), latest_model_path)
            logger.info(f"💾 BEST Optuna GRU Scalper model also saved as latest: {latest_model_path}")
        return f1

def objective(trial: optuna.trial.Trial, args_ns):
    try:
        trainer = GRUScalperTrainer(trial=trial, ticker=args_ns.ticker, interval=args_ns.interval, days_back=args_ns.days_back) # GRUScalperTrainer
        f1_metric = trainer.run_training_pipeline()
        return f1_metric
    except optuna.exceptions.TrialPruned: raise
    except ValueError as e: logger.error(f"Trial {trial.number} ValueError: {e}"); return 0.0
    except Exception as e: logger.error(f"Trial {trial.number} Exception: {e}", exc_info=True); return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize GRU Scalping Model with Optuna") # GRU
    parser.add_argument("--ticker", type=str, default="ETH-USD", help="Ticker symbol")
    parser.add_argument("--interval", type=str, default="5m", help="Data interval")
    parser.add_argument("--days_back", type=int, default=59, help="Days of historical data")
    parser.add_argument("--n_trials", type=int, default=0, help="Number of Optuna trials (0 to skip Optuna and run with fixed best params)")
    parser.add_argument('--use_best_lstm_params', action='store_true', help='Run with hardcoded best params from previous LSTM Optuna run')
    
    args = parser.parse_args()

    if args.use_best_lstm_params:
        logger.info("Using hardcoded best parameters from a previous LSTM Optuna run for GRU.")
        # Using parameters from the 1000-trial LSTM run (best F1 during study was ~0.67)
        best_lstm_params = {
            'rsi_period': 15, 'sequence_length': 20, 'prediction_horizon_bars': 3,
            'hidden_dim': 96, 'num_layers': 3, 'dropout_prob': 0.18749507903985016,
            'learning_rate': 0.0005862685172654825, 'batch_size': 64, 'epochs': 50
        }
        fixed_trial = FixedParams(best_lstm_params)
        final_trainer = GRUScalperTrainer( # GRUScalperTrainer
            trial=fixed_trial,
            ticker=args.ticker, interval=args.interval, days_back=args.days_back
        )
        logger.info(f"Running GRU training pipeline with best LSTM params: {best_lstm_params}")
        final_trainer.run_training_pipeline(is_final_run=True)
        logger.info("GRU Training with best LSTM parameters completed.")

    elif args.n_trials > 0:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
        logger.info("\nOptuna optimization for GRU finished.")
        if study.best_trial:
            best_trial = study.best_trial
            logger.info("Best GRU trial from this Optuna run:")
            logger.info(f"  Value (F1 Score): {best_trial.value:.4f}")
            logger.info("  Params: "); [logger.info(f"    {key}: {value}") for key, value in best_trial.params.items()]
            logger.info("\nRetraining final GRU model with best parameters from this Optuna run...")
            best_params_trial_obj = FixedParams(best_trial.params)
            final_trainer = GRUScalperTrainer( # GRUScalperTrainer
                trial=best_params_trial_obj, 
                ticker=args.ticker, interval=args.interval, days_back=args.days_back
            )
            logger.info(f"Running final GRU training pipeline with best params from this run: {best_trial.params}")
            final_trainer.run_training_pipeline(is_final_run=True)
            logger.info("Final GRU model training with best Optuna parameters from this run completed.")
        else: logger.warning("No best trial found in this GRU Optuna study.")
    else: logger.info("No Optuna trials requested and --use_best_lstm_params not set for GRU. Exiting.")
