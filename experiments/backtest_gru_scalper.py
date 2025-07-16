#!/usr/bin/env python3
"""
Backtests the trained GRU scalping model for ETH-USD.
"""
import pandas as pd
import numpy as np
# import yfinance as yf # No longer using yfinance
from binance.client import Client # For Binance API
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import argparse
import os # Added import for os.getenv
from dotenv import load_dotenv 


from backtesting import Backtest, Strategy
# Assuming train_gru_scalper.py contains the ScalpingGRU model definition
# and FixedParams (or we redefine ScalpingGRU here if needed)
from train_gru_scalper import ScalpingGRU, FixedParams # To load model and use FixedParams for model init

from main_files.config_loader import config
logger = config.logger
load_dotenv() # Ensure .env variables are loaded

def fetch_backtest_data_binance(ticker_symbol="ETHUSDT", interval_str="5m", days_back=59):
    api_key = os.getenv("BINANCE_API")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError("BINANCE_API or BINANCE_SECRET_KEY not found in .env file.")

    client = Client(api_key, api_secret)
    logger.info(f"Fetching Binance data for {ticker_symbol}, interval {interval_str}, {days_back} days back...")

    # Binance interval mapping
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
    
    # Fetch klines
    klines = client.get_historical_klines(ticker_symbol, binance_interval, start_str)
    
    if not klines:
        raise ValueError(f"No data fetched from Binance for {ticker_symbol}.")

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'CloseTime', 'QuoteAssetVolume', 'NumberofTrades', 
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
    ])

    # Convert timestamp to datetime and set as index
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df.set_index('Datetime', inplace=True)
    
    # Convert relevant columns to numeric
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[cols_to_numeric] # Keep only the necessary columns
    df.dropna(inplace=True) # Drop rows with NaN in OHLCV

    logger.info(f"Binance data fetched. Shape: {df.shape}")
    return df

class GRUScalpingStrategy(Strategy):
    # Parameters for the best GRU model (from Optuna Trial 19 of 50-trial GRU run)
    # These should match the model being loaded.
    rsi_period = 24 
    sequence_length = 35
    # prediction_horizon_bars = 5 # Not directly used by strategy logic if exiting on opposite signal
    
    # Strategy parameters (can be optimized later by backtesting.py's optimizer)
    buy_threshold = 0.55 
    sell_threshold = 0.45
    trade_size_percent_of_equity = 0.05 # Use 10% of equity for sizing

    model_path = "models/gru_scalper_model_ETH-USD_latest.pth" 
    scaler_path = "models/gru_scalper_X_scaler_ETH-USD_best_optuna.joblib" # This should match the scaler saved with the best GRU

    def init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load the scaler
        try:
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
        except FileNotFoundError:
            logger.error(f"Scaler file not found at {self.scaler_path}. Ensure it's saved from training.")
            raise
        
        # Determine model parameters from the best GRU trial (Trial 19 from the 50-trial GRU run)
        # These are needed to instantiate the ScalpingGRU class correctly.
        best_gru_params = {
            'rsi_period': 24, 'sequence_length': 35, 'prediction_horizon_bars': 5,
            'hidden_dim': 64, 'num_layers': 3, 'dropout_prob': 0.10206491971558534, # From Trial 19
            'learning_rate': 0.002359227395136525, 'batch_size': 128, 'epochs': 50
        }
        # The input_dim for GRU is 2 (RSI, VWAP)
        input_dim = 2 
        
        self.model = ScalpingGRU(
            input_dim=input_dim,
            hidden_dim=best_gru_params['hidden_dim'],
            num_layers=best_gru_params['num_layers'],
            dropout_prob=best_gru_params['dropout_prob']
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"GRU model loaded from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. Train the GRU model first.")
            raise
        self.model.eval()

        # Prepare data columns for features
        self.rsi_col_name = f'RSI_{self.rsi_period}'
        self.data_df = pd.DataFrame(index=self.data.index)
        self.data_df['Close'] = self.data.Close
        self.data_df['High'] = self.data.High
        self.data_df['Low'] = self.data.Low
        self.data_df['Volume'] = self.data.Volume
        
        self.data_df[self.rsi_col_name] = ta.rsi(self.data_df['Close'], length=self.rsi_period)
        self.data_df['VWAP'] = ta.vwap(self.data_df['High'], self.data_df['Low'], self.data_df['Close'], self.data_df['Volume'])
        
        # Drop NaNs that arise from indicator calculations at the start
        self.data_df.dropna(inplace=True)
        
        # Scale the features that the model expects
        # Important: Only scale the feature columns, not the whole df
        feature_columns_to_scale = [self.rsi_col_name, 'VWAP']
        if not self.data_df.empty and all(col in self.data_df.columns for col in feature_columns_to_scale):
            self.data_df[feature_columns_to_scale] = self.scaler.transform(self.data_df[feature_columns_to_scale])
        else:
            logger.warning("Not enough data or missing columns for feature scaling in init. Backtest might be unreliable.")
            # This might happen if backtesting data is shorter than rsi_period + sequence_length

    def next(self):
        # Ensure enough historical data for one sequence
        if len(self.data.Close) < self.sequence_length:
            return
        
        # Get the latest sequence_length bars of data from self.data_df
        # Need to align indices carefully if self.data_df was shortened by dropna
        current_bar_time = self.data.index[-1]
        
        if current_bar_time not in self.data_df.index:
            # This bar's features might not be available if it was part of initial NaNs
            return

        # Get the slice of data ending at the current bar's time from our pre-processed df
        # The features are already scaled in self.data_df during init
        try:
            # Get the index location of the current bar in the preprocessed data_df
            current_loc = self.data_df.index.get_loc(current_bar_time)
            if current_loc < self.sequence_length -1 : # Not enough history in preprocessed data
                return

            start_loc = current_loc - self.sequence_length + 1
            
            # Features are already scaled in self.data_df
            sequence_data = self.data_df[[self.rsi_col_name, 'VWAP']].iloc[start_loc : current_loc + 1].values
        except KeyError:
            # logger.warning(f"Timestamp {current_bar_time} not found in preprocessed feature data. Skipping.")
            return
        except Exception as e:
            logger.error(f"Error preparing sequence at {current_bar_time}: {e}")
            return


        if sequence_data.shape[0] != self.sequence_length:
            # Not enough data for a full sequence (e.g. at the very beginning)
            return

        sequence_tensor = torch.from_numpy(sequence_data).float().unsqueeze(0).to(self.device) # Add batch dimension

        with torch.no_grad():
            prediction_prob = self.model(sequence_tensor).item() # Probability of price going up

        # Trading Logic
        # Calculate position size based on 10% of current equity
        # Note: self.equity gives current portfolio value. self.cash is available cash.
        # For leveraged trading, size is often calculated based on a fraction of equity for risk,
        # and the broker applies margin.
        # backtesting.py's `size` parameter in buy/sell is a proportion of equity if < 1.
        # Example: self.buy(size=0.1) would use 10% of equity.
        # The margin setting in Backtest() handles the leverage.
        
        position_size = self.trade_size_percent_of_equity 

        if not self.position: # If no open position
            if prediction_prob > self.buy_threshold:
                self.buy(size=position_size)
                # logger.debug(f"{self.data.index[-1]}: BUY signal. Prob: {prediction_prob:.4f}, Size: {position_size*100:.1f}%")
            elif prediction_prob < self.sell_threshold:
                self.sell(size=position_size)
                # logger.debug(f"{self.data.index[-1]}: SELL signal. Prob: {prediction_prob:.4f}, Size: {position_size*100:.1f}%")
        else: # If position is open
            if self.position.is_long and prediction_prob < self.sell_threshold:
                self.position.close()
                # logger.debug(f"{self.data.index[-1]}: CLOSE LONG signal. Prob: {prediction_prob:.4f}")
                # Optionally, go short immediately if desired (flip position)
                # self.sell(size=position_size) 
            elif self.position.is_short and prediction_prob > self.buy_threshold:
                self.position.close()
                # logger.debug(f"{self.data.index[-1]}: CLOSE SHORT signal. Prob: {prediction_prob:.4f}")
                # Optionally, go long immediately
                # self.buy(size=position_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest GRU Scalping Model")
    parser.add_argument("--ticker", type=str, default="ETHUSDT", help="Ticker symbol (e.g., ETHUSDT, BTCUSDT)") # Changed default for Binance
    # parser.add_argument("--period", type=str, default="6mo", help="Period for historical data (e.g., 3mo, 6mo, 1y)") # Not used for Binance
    parser.add_argument("--days_back", type=int, default=59, help="Number of days of historical data to fetch for Binance")
    parser.add_argument("--interval", type=str, default="5m", help="Data interval (must match model training)")
    parser.add_argument("--cash", type=float, default=10000, help="Initial cash for backtest")
    parser.add_argument("--commission", type=float, default=0.00075, help="Commission per trade (e.g., 0.00075 for 0.075%)")
    parser.add_argument("--buy_thresh", type=float, default=0.55, help="Probability threshold to buy")
    parser.add_argument("--sell_thresh", type=float, default=0.45, help="Probability threshold to sell (close long / open short)")
    parser.add_argument("--trade_size_percent", type=float, default=0.10, help="Percentage of equity to use for trade size (e.g., 0.1 for 10%)")
    parser.add_argument("--leverage", type=float, default=20.0, help="Leverage to apply (e.g., 20 for 20x)")

    args = parser.parse_args()

    try:
        # Use Binance data fetching
        # Note: Binance uses "ETHUSDT" not "ETH-USD". Adjust default or ensure arg is correct.
        binance_ticker = args.ticker.replace('-', '') # Convert "ETH-USD" to "ETHUSDT"
        data = fetch_backtest_data_binance(ticker_symbol=binance_ticker, interval_str=args.interval, days_back=args.days_back)
        
        if data.empty or len(data) < 500: # Need substantial data for backtesting sequences
            logger.error("Not enough data fetched for a meaningful backtest.")
        else:
            GRUScalpingStrategy.buy_threshold = args.buy_thresh
            GRUScalpingStrategy.sell_threshold = args.sell_thresh
            GRUScalpingStrategy.trade_size_percent_of_equity = args.trade_size_percent
            
            # Calculate margin for backtesting.py (1 / leverage)
            margin_ratio = 1 / args.leverage

            # Ensure the model and scaler paths are correct
            # These should point to the artifacts from the best GRU model run
            GRUScalpingStrategy.model_path = "models/gru_scalper_model_ETH-USD_latest.pth"
            GRUScalpingStrategy.scaler_path = "models/gru_scalper_X_scaler_ETH-USD_best_optuna.joblib"
            # The RSI period used by the loaded model is critical.
            # The best GRU model used rsi_period: 24
            GRUScalpingStrategy.rsi_period = 24 # Make sure this matches the loaded model's training
            # The sequence length used by the loaded model
            GRUScalpingStrategy.sequence_length = 35 # Make sure this matches the loaded model's training


            logger.info(f"Running GRU backtest for {args.ticker} from {data.index[0]} to {data.index[-1]}...")
            logger.info(f"Trade size: {args.trade_size_percent*100:.1f}% of equity, Leverage: {args.leverage}x (margin ratio: {margin_ratio:.4f})")
            bt = Backtest(data, GRUScalpingStrategy, cash=args.cash, commissio n=args.commission, margin=margin_ratio, trade_on_close=True)
            stats = bt.run()
            
            logger.info("\n--- GRU Backtest Stats ---") 
            logger.info(stats)
            
            # Print individual stats for clarity
            if '_trades' in stats and not stats['_trades'].empty:
                 logger.info(f"\nReturn [%]: {stats['Return [%]']:.2f}")
                 logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
                 logger.info(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
                 logger.info(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}")
                 logger.info(f"# Trades: {stats['# Trades']}")
            else:
                logger.info("No trades were executed during the backtest.")

            plot_filename = f"reports/backtest_gru_scalper_{args.ticker.replace('=X','')}_{args.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path("reports").mkdir(parents=True, exist_ok=True)
            bt.plot(filename=plot_filename, open_browser=False)
            logger.info(f"📈 Backtest plot saved to {plot_filename}")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}. Ensure model/scaler paths are.  correct and files exist.")
    except ValueError as e:
        logger.error(f"ValueError during backtest setup or run: {e}") 
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
