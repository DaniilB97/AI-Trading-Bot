#!/usr/bin/env python3
"""
Script to train a Hidden Markov Model (HMM) for detecting market regimes in ETH.
"""
import sys # Added for data.info(buf=sys.stdout)
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse

from config_loader import config # Assuming this logger is suitable
logger = config.logger

def fetch_and_prepare_hmm_features(ticker="ETH-USD", days_back=730, interval="1d", 
                                   volatility_window=20 # Standard 20-day vol
                                   # rsi_window and volume_avg_window removed as features are removed
                                   ):
    """
    Fetches daily historical data and prepares features for HMM.
    Features: daily_returns, realized_volatility.
    """
    logger.info(f"Fetching data for {ticker} for HMM feature preparation (daily, returns & vol only)...")
    
    cache_filename = f"data/raw/hmm_features_daily_ret_vol_{ticker.replace('=X', '')}_{days_back}d.csv" # New cache name
    cache_path = Path(cache_filename)

    if cache_path.exists():
        logger.info(f"📂 Loading HMM features from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    data = yf.download(ticker, period=f"{days_back}d", interval=interval)
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}.")
    
    data.index = pd.to_datetime(data.index).tz_localize(None)
    
    df = pd.DataFrame(index=data.index)
    
    # Adjust column access for potential MultiIndex from yfinance
    close_col = 'Close' if 'Close' in data.columns else ('Close', ticker)
    volume_col = 'Volume' if 'Volume' in data.columns else ('Volume', ticker)

    df['daily_returns'] = data[close_col].pct_change().fillna(0)
    df['realized_volatility'] = df['daily_returns'].rolling(window=volatility_window).std() * np.sqrt(volatility_window) # Annualized for window
    
    logger.info("DEBUG: yfinance data info:")
    data.info(buf=sys.stdout) # Print info to stdout/log
    logger.info("DEBUG: yfinance data head:")
    logger.info(data.head())
    
    volume_present_and_positive = False
    if volume_col in data.columns:
        volume_series = data[volume_col]
        logger.info(f"DEBUG: data[volume_col] type: {type(volume_series)}")
        logger.info(f"DEBUG: data[volume_col].head():\n{volume_series.head()}")
        volume_sum = volume_series.sum()
        logger.info(f"DEBUG: data[volume_col].sum() result: {volume_sum}, type: {type(volume_sum)}")
        if isinstance(volume_sum, pd.Series): # Should not happen if volume_col correctly selects a Series
            volume_sum_scalar = volume_sum.iloc[0] if not volume_sum.empty else 0
        else:
            volume_sum_scalar = volume_sum
        
        volume_present_and_positive = volume_sum_scalar > 0
        logger.info(f"DEBUG: volume_sum_scalar > 0 evaluation: {volume_present_and_positive}")

    # Volume ratio and RSI are removed as features for this experiment
    # df['volume_ratio'] = ...
    # df['rsi'] = ...
    
    # VIX_proxy - placeholder for now
    # df['vix_proxy'] = 0 # Replace with actual data if available

    df.dropna(inplace=True)
    
    # Scale features (HMMs often work better with scaled data, especially GaussianHMM)
    # We will scale only the features used by HMM, not 'daily_returns' if it's used differently later.
    # For GaussianHMM, features should ideally be Gaussian-like. Returns often are not.
    # Consider transformations or different HMM types if Gaussian assumption is poor.
    
    feature_cols = ['daily_returns', 'realized_volatility'] # Using only returns and volatility
    # Add 'vix_proxy' if available: feature_cols.append('vix_proxy')
    
    # Note: 'daily_returns' might not be Gaussian. Scaling it with StandardScaler is standard practice,
    # but its distribution should be kept in mind when interpreting HMM results or choosing HMM type.
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Store scaler for later use if needed to transform new data
    # joblib.dump(scaler, f"models/hmm_feature_scaler_{ticker.replace('=X', '')}.joblib")

    df.to_csv(cache_path)
    logger.info(f"💾 HMM features saved to cache: {cache_path}")
    return df

def train_hmm_model(features_df: pd.DataFrame, n_states=2, n_iterations=100, cov_type="diag"):
    """
    Trains a Gaussian HMM model.
    Features in features_df are expected to be already scaled.
    """
    logger.info(f"Training HMM with {n_states} states...")
    
    # Use features that are more likely to be Gaussian or stationary after scaling
    # 'daily_returns' itself might not be ideal for GaussianHMM's direct features
    # but can be an observable. Here, we use only daily_returns and realized_volatility.
    hmm_feature_columns = ['daily_returns', 'realized_volatility']
    # Add 'vix_proxy' if available: hmm_feature_columns.append('vix_proxy')
    feature_matrix = features_df[hmm_feature_columns].values


    if np.any(np.isinf(feature_matrix)) or np.any(np.isnan(feature_matrix)):
        logger.warning("NaN or Inf found in feature matrix before HMM training. Attempting to clean...")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0 or mean/median

    model = hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=n_iterations, tol=1e-3, random_state=42)
    model.fit(feature_matrix)
    
    logger.info(f"HMM training completed. Converged: {model.monitor_.converged}")
    return model

def plot_regimes(df_with_regimes: pd.DataFrame, price_series: pd.Series, n_states: int):
    """Plots price data with colored backgrounds for regimes."""
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    colors = ['#00FF00', '#FF0000', '#FFFF00', '#0000FF', '#FFA500'] # Green, Red, Yellow, Blue, Orange
    regime_labels = [f"Regime {i}" for i in range(n_states)]

    price_series.plot(ax=ax1, color='black', label='ETH Price', alpha=0.7)
    ax1.set_ylabel('ETH Price')
    
    for i in range(n_states):
        regime_data = df_with_regimes[df_with_regimes['regime'] == i]
        if not regime_data.empty:
            ax1.fill_between(regime_data.index, price_series.min(), price_series.max(), 
                             color=colors[i % len(colors)], alpha=0.2, label=f'{regime_labels[i]} (State {i})')
    
    ax1.legend(loc='upper left')
    plt.title(f'ETH Price with Detected Market Regimes ({n_states} States)')
    plt.tight_layout()
    
    plot_filename = f"reports/hmm_regimes_plot_{n_states}_states_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    Path("reports").mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)
    logger.info(f"📈 Regime plot saved to {plot_filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMM for Market Regime Detection")
    parser.add_argument("--ticker", type=str, default="ETH-USD", help="Ticker symbol (e.g., ETH-USD, BTC-USD)")
    parser.add_argument("--days_back", type=int, default=730, help="Number of past days of data to use (e.g., 730 for 2 years)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (changed default to 1d)")
    parser.add_argument("--n_states", type=int, default=3, choices=[2, 3, 4], help="Number of hidden states for HMM")
    parser.add_argument("--n_iterations", type=int, default=100, help="Number of EM iterations for HMM training")
    parser.add_argument("--covariance_type", type=str, default="diag", choices=["spherical", "diag", "full", "tied"], help="HMM covariance type")
    
    args = parser.parse_args()

    logger.info(f"🚀 Starting HMM Regime Model Training for {args.ticker}...")

    # 1. Fetch and prepare data
    features_df = fetch_and_prepare_hmm_features(
        ticker=args.ticker, 
        days_back=args.days_back, 
        interval=args.interval
        # Window parameters will use the new defaults in the function definition
    )
    
    if features_df.empty or len(features_df) < 50: # Need enough data for HMM
        logger.error("Недостаточно данных для тренировки HMM после обработки.")
    else:
        # 2. Train HMM model
        hmm_model = train_hmm_model(features_df, n_states=args.n_states, n_iterations=args.n_iterations, cov_type=args.covariance_type)
        
        # 3. Decode regimes
        # Use the same features for decoding as used for training
        hmm_feature_columns_for_decode = ['daily_returns', 'realized_volatility']
        # Add 'vix_proxy' if available: hmm_feature_columns_for_decode.append('vix_proxy')
        feature_matrix_for_decoding = features_df[hmm_feature_columns_for_decode].values
        feature_matrix_for_decoding = np.nan_to_num(feature_matrix_for_decoding, nan=0.0, posinf=0.0, neginf=0.0)


        hidden_states = hmm_model.predict(feature_matrix_for_decoding)
        
        # Add regimes to the DataFrame for analysis and plotting
        # Ensure features_df index aligns with hidden_states
        df_with_regimes = features_df.copy()
        df_with_regimes['regime'] = hidden_states
        
        logger.info("\n🔍 Regime Counts:")
        logger.info(df_with_regimes['regime'].value_counts(normalize=True).sort_index())
        
        logger.info("\n📊 Characteristics per Regime (based on scaled features):")
        for i in range(args.n_states):
            regime_data = df_with_regimes[df_with_regimes['regime'] == i]
            logger.info(f"\n--- Regime {i} ---")
            logger.info(f"Percentage of time: {len(regime_data) / len(df_with_regimes) * 100:.2f}%")
            # Display means of original (unscaled) values for better interpretation if possible
            # For now, using scaled features for characteristics
            logger.info(regime_data[['daily_returns', 'realized_volatility']].mean()) # Adjusted to current features


        # 4. Save the HMM model
        model_filename = f"models/hmm_regime_model_{args.ticker.replace('=X', '')}_{args.n_states}states_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(hmm_model, model_filename)
        logger.info(f"💾 HMM модель сохранена: {model_filename}")

        # Save latest HMM model
        latest_model_filename = f"models/hmm_regime_model_{args.ticker.replace('=X', '')}_latest.joblib"
        joblib.dump(hmm_model, latest_model_filename)
        logger.info(f"💾 HMM модель сохранена как последняя: {latest_model_filename}")

        # 5. Plot regimes against price
        # Fetch original price data again for plotting, using the specified interval
        price_data_for_plot = yf.download(args.ticker, period=f"{args.days_back}d", interval=args.interval)['Close']
        price_data_for_plot.index = pd.to_datetime(price_data_for_plot.index).tz_localize(None)
        # Align price_data_for_plot with df_with_regimes index
        aligned_price_series = price_data_for_plot.reindex(df_with_regimes.index) 
        
        if not aligned_price_series.empty:
            plot_regimes(df_with_regimes, aligned_price_series, args.n_states)
        else:
            logger.warning("Не удалось построить график режимов из-за проблем с данными цен.")

    logger.info("✅ Тренировка HMM модели завершена.")
