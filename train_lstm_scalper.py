#!/usr/bin/env python3
"""
Script to train an LSTM model for ETH scalping strategy.
Features: RSI(20), VWAP on 5-minute data.
Target: Price direction in N future bars.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Added imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import argparse
import sys

# Add project root to path if not already (adjust if your structure differs)
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_loader import config
logger = config.logger

# --- LSTM Model Definition ---
class ScalpingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1, dropout_prob=0.2):
        super(ScalpingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Get output of the last time step
        out = self.sigmoid(out)
        return out

class LSTMScalperTrainer:
    def __init__(self, ticker="ETH-USD", interval="5m", days_back=59, 
                 rsi_period=20, sequence_length=24, prediction_horizon_bars=3): # 24 bars * 5min = 2hrs lookback; 3 bars * 5min = 15min horizon
        self.ticker = ticker
        self.interval = interval
        self.days_back = days_back
        self.rsi_period = rsi_period
        self.sequence_length = sequence_length
        self.prediction_horizon_bars = prediction_horizon_bars
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.feature_cols = ['RSI_20', 'VWAP'] # Features to be used
        self.target_col = 'Target'
        self.data_df = None
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))
        # Target is binary, no scaler needed for it. If predicting price change, target_scaler would be needed.

        Path("models").mkdir(parents=True, exist_ok=True)
        Path("data/raw").mkdir(parents=True, exist_ok=True)

    def _fetch_data(self):
        logger.info(f"Fetching {self.interval} data for {self.ticker}, {self.days_back} days back...")
        cache_filename = f"data/raw/lstm_scalper_data_{self.ticker.replace('=X','')}_{self.days_back}d_{self.interval}.csv"
        cache_path = Path(cache_filename)

        if cache_path.exists():
            logger.info(f"📂 Loading data from cache: {cache_path}")
            # When loading, explicitly state the index column name if it was set
            df = pd.read_csv(cache_path, index_col='Datetime', parse_dates=['Datetime'])
            df.index = pd.to_datetime(df.index).tz_localize(None) # Ensure tz-naive if not already
            return df

        data = yf.download(self.ticker, period=f"{self.days_back}d", interval=self.interval)
        if data.empty:
            raise ValueError(f"No data fetched for {self.ticker} with interval {self.interval}.")
        data.index = pd.to_datetime(data.index).tz_localize(None)
        
        # Ensure Volume exists for VWAP and handle potential MultiIndex
        volume_col_name = 'Volume' # Base name
        # Determine actual column accessor (flat or tuple for MultiIndex)
        actual_volume_col = volume_col_name if volume_col_name in data.columns else (volume_col_name, self.ticker)

        volume_present_and_positive = False
        if actual_volume_col in data.columns:
            volume_series = data[actual_volume_col]
            volume_sum = volume_series.sum()
            
            # Ensure volume_sum is a scalar for comparison
            if isinstance(volume_sum, pd.Series):
                volume_sum_scalar = volume_sum.iloc[0] if not volume_sum.empty else 0
            else:
                volume_sum_scalar = volume_sum
            
            if volume_sum_scalar > 0:
                volume_present_and_positive = True
            else:
                logger.warning(f"Volume data for {self.ticker} is present but sum is zero. VWAP might be NaN.")
        else:
            logger.warning(f"Volume column ('{volume_col_name}' or {actual_volume_col}) not found for {self.ticker}. Setting Volume to 0 for VWAP calculation.")
            data['Volume'] = 0 # Add a zero volume column if completely missing
            # If we added it, actual_volume_col needs to be the simple string 'Volume' for VWAP calculation later
            # This part is tricky if VWAP expects a specific column name and it wasn't there.
            # For pandas_ta, it usually expects 'Volume'.
            # Let's ensure 'Volume' (flat name) exists if we add it.
            if 'Volume' not in data.columns: # If it was a multi-index and we added 'Volume'
                 data.rename(columns={actual_volume_col: 'Volume'}, inplace=True) # This might be too complex here
                 # Simpler: ensure 'Volume' exists if we had to create it.
                 # The VWAP calculation later will use df['Volume']
            
        if not volume_present_and_positive and 'Volume' not in data.columns:
             data['Volume'] = 0 # Ensure 'Volume' column exists if it was missing and not positive

        # Flatten columns if they are MultiIndex from yfinance (for single ticker)
        if isinstance(data.columns, pd.MultiIndex) and len(data.columns.levels) > 1 and len(data.columns.levels[1]) == 1:
            logger.info("Flattening MultiIndex columns from yfinance data before caching.")
            data.columns = data.columns.droplevel(1) # Drop the ticker level, e.g., ('Close', 'ETH-USD') -> 'Close'
        
        # Ensure the index has a name before saving, helps with reloading
        if data.index.name is None:
            data.index.name = 'Datetime'

        data.to_csv(cache_path)
        logger.info(f"💾 Raw data saved to cache: {cache_path}")
        return data

    def _prepare_features_and_target(self, df):
        logger.info("Preparing features and target...")
        df_features = pd.DataFrame(index=df.index)

        # Correctly access OHLCV columns, handling potential MultiIndex from yfinance if df is raw yf output
        # However, df passed here should be the one from _fetch_data, which might already be processed or cached.
        # For safety, let's assume df might still have MultiIndex if loaded from an old cache or if _fetch_data is bypassed.
        
        close_col = 'Close' if 'Close' in df.columns else ('Close', self.ticker)
        high_col = 'High' if 'High' in df.columns else ('High', self.ticker)
        low_col = 'Low' if 'Low' in df.columns else ('Low', self.ticker)
        volume_col_for_vwap = 'Volume' # pandas_ta.vwap expects 'Volume'
        
        # Ensure 'Volume' column exists with the flat name if it was added due to being missing.
        # If yfinance returned ('Volume', ticker_name), it needs to be accessible as 'Volume'.
        # This is complex if the original df is multi-indexed.
        # A simpler approach is to flatten columns after yf.download if only one ticker.
        # For now, assuming 'Volume' will be present as a flat column name if it was added.
        # If yf.download returns multi-index, and 'Volume' is ('Volume', 'ETH-USD'),
        # ta.vwap might fail if it strictly looks for 'Volume'.
        # Let's assume df passed to _prepare_features_and_target has flat column names for OHLCV.
        # This is true if loaded from our cache, or if yf.download for single ticker flattens.
        # The yf.download for single ticker *does* return multi-index as seen in HMM script.
        # So, we need to handle it here too.

        # Simplification: Flatten columns if multi-indexed and only one ticker
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels[1]) == 1:
            # ticker_name = df.columns.levels[1][0] # Get the ticker name
            df.columns = df.columns.droplevel(1) # Drop the ticker level

        df_features['RSI_20'] = ta.rsi(df['Close'], length=self.rsi_period)
        df_features['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Target: 1 if price is higher N bars ahead, 0 otherwise
        df_features[self.target_col] = (df['Close'].shift(-self.prediction_horizon_bars) > df['Close']).astype(int)
        
        # Combine with original OHLC if needed for other purposes, but LSTM will use scaled features
        # df_processed = df.join(df_features)
        df_processed = df_features.copy() # Only keep generated features and target for now
        df_processed.dropna(inplace=True)
        
        if df_processed.empty:
            raise ValueError("DataFrame is empty after feature calculation and NaN drop. Check data or indicator parameters.")
            
        self.data_df = df_processed
        logger.info(f"Features and target prepared. Shape: {self.data_df.shape}")
        return self.data_df

    def _scale_and_create_sequences(self):
        if self.data_df is None:
            raise ValueError("Data not prepared. Call _prepare_features_and_target first.")

        X_data = self.data_df[self.feature_cols].values
        y_data = self.data_df[self.target_col].values.reshape(-1, 1) # Reshape for consistency

        X_scaled = self.X_scaler.fit_transform(X_data)
        
        # Save the scaler
        scaler_filename = f"models/lstm_scalper_X_scaler_{self.ticker.replace('=X','')}.joblib"
        joblib.dump(self.X_scaler, scaler_filename)
        logger.info(f"💾 X_scaler saved to {scaler_filename}")

        X_sequences, y_sequences = [], []
        # The loop range must account for sequence_length AND prediction_horizon_bars
        # to ensure y_data[index_for_target] is valid.
        # A sequence X_scaled[i : i + self.sequence_length] corresponds to original data points
        # from index i to i + self.sequence_length - 1.
        # The target for this sequence is at original index (i + self.sequence_length - 1),
        # because y_data was created from df_processed where 'Target' was already shifted.
        # The last possible 'i' is such that (i + self.sequence_length - 1) is a valid index in y_data.
        # So, len(X_scaled) - self.sequence_length is the correct upper bound for 'i' if y_data has same length as X_scaled.
        
        # The target 'Target' was created by df['Close'].shift(-self.prediction_horizon_bars)
        # This means y_data[k] is the target for the price at original time k, looking prediction_horizon_bars into the future.
        # When we create a sequence X_scaled[i : i + self.sequence_length], the *last element* of this sequence
        # corresponds to the original data point at index (i + self.sequence_length - 1).
        # We want to predict the target associated with this *last element* of the sequence.
        # So, the target is y_data[i + self.sequence_length - 1].

        # The loop should go up to a point where `i + self.sequence_length - 1` is a valid index in `y_data`.
        # Since `X_scaled` and `y_data` are derived from `self.data_df` (after dropping NaNs from target shift),
        # they should have the same length.
        # The number of sequences will be len(X_scaled) - self.sequence_length + 1.
        # The last valid 'i' is len(X_scaled) - self.sequence_length.
        
        # Corrected loop:
        for i in range(len(X_scaled) - self.sequence_length + 1):
            # Sequence X is from index i to i + sequence_length - 1
            X_sequences.append(X_scaled[i : i + self.sequence_length])
            # Target y corresponds to the end of this sequence
            # y_data is already shifted, so y_data[k] is target for time k
            # The end of current sequence X is at original time index (i + sequence_length - 1)
            y_sequences.append(y_data[i + self.sequence_length - 1])

        if not X_sequences: # Handle case where not enough data for even one sequence
            logger.warning("Not enough data to create any sequences after considering sequence_length.")
            return np.array([]), np.array([])

        return np.array(X_sequences), np.array(y_sequences)


    def get_dataloaders(self, X, y, batch_size=64):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Ensure chronological split if not shuffling (though DataLoader might shuffle train)
        # For time series, explicit splitting before DataLoader is better
        train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size + test_size], generator=torch.Generator().manual_seed(42))
        val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))
        
        # For time series, we should not shuffle. Let's do manual split.
        # Manual chronological split:
        split1 = train_size
        split2 = train_size + val_size
        
        train_indices = list(range(split1))
        val_indices = list(range(split1, split2))
        test_indices = list(range(split2, len(dataset)))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices) # Shuffle for training
        val_sampler = torch.utils.data.SequentialSampler(val_indices)
        test_sampler = torch.utils.data.SequentialSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        return train_loader, val_loader, test_loader

    def train_model(self, model, train_loader, val_loader, epochs=50, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        logger.info("Starting LSTM training...")
        for epoch in range(epochs):
            model.train()
            train_loss_epoch = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            
            model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss_epoch += loss.item()
            
            avg_train_loss = train_loss_epoch / len(train_loader)
            avg_val_loss = val_loss_epoch / len(val_loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        logger.info("LSTM training finished.")
        return model

    def evaluate_model(self, model, test_loader):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred_probs = model(X_batch)
                y_pred_labels = (y_pred_probs > 0.5).float()
                all_preds.extend(y_pred_labels.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(cm))
        return accuracy, report, cm

    def run_training_pipeline(self, epochs=50, lr=0.001, batch_size=64):
        df_raw = self._fetch_data()
        df_processed = self._prepare_features_and_target(df_raw)
        X_seq, y_seq = self._scale_and_create_sequences()

        if len(X_seq) == 0 or len(y_seq) == 0 or len(X_seq) != len(y_seq):
            logger.error(f"Sequence creation resulted in empty or mismatched arrays. X_seq len: {len(X_seq)}, y_seq len: {len(y_seq)}")
            return

        train_loader, val_loader, test_loader = self.get_dataloaders(X_seq, y_seq, batch_size=batch_size)
        
        input_dim = X_seq.shape[2] # Number of features
        lstm_model = ScalpingLSTM(input_dim=input_dim).to(self.device)
        
        trained_model = self.train_model(lstm_model, train_loader, val_loader, epochs=epochs, learning_rate=lr)
        self.evaluate_model(trained_model, test_loader)

        # Save the trained model
        model_filename = f"models/lstm_scalper_model_{self.ticker.replace('=X','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(trained_model.state_dict(), model_filename)
        logger.info(f"💾 LSTM Scalper model saved to {model_filename}")
        
        latest_model_path = f"models/lstm_scalper_model_{self.ticker.replace('=X','')}_latest.pth"
        torch.save(trained_model.state_dict(), latest_model_path)
        logger.info(f"💾 LSTM Scalper model saved as latest: {latest_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Scalping Model")
    parser.add_argument("--ticker", type=str, default="ETH-USD", help="Ticker symbol")
    parser.add_argument("--interval", type=str, default="5m", help="Data interval (e.g., 5m, 1m)")
    parser.add_argument("--days_back", type=int, default=59, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=24, help="Sequence length for LSTM (e.g., 24 for 2 hours of 5m data)")
    parser.add_argument("--prediction_horizon_bars", type=int, default=3, help="Number of bars ahead to predict direction (e.g., 3 for 15 mins on 5m data)")

    args = parser.parse_args()

    trainer = LSTMScalperTrainer(
        ticker=args.ticker,
        interval=args.interval,
        days_back=args.days_back,
        sequence_length=args.sequence_length,
        prediction_horizon_bars=args.prediction_horizon_bars
    )
    trainer.run_training_pipeline(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
