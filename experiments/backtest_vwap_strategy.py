#!/usr/bin/env python3
"""
Backtesting script for the VWAP Scalping strategy based on TotalSignal.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from pathlib import Path
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from main_files.config_loader import config

logger = config.logger

def get_strategy_data(days_back=59, interval="5m"): # Changed for 5-min scalping
    """
    Prepares 5-minute ETH data with VWAP, RSI, BBands, ATR, VWAPSignal, and TotalSignal.
    Similar to prepare_technical_data in train_random_forest.py but without BTC/time features for this specific strategy.
    """
    logger.info(f"📊 Загружаем данные EUR/USD ({interval} интервал) для бэктестинга стратегии...")
    # Cache for this specific backtesting data
    cache_file_name = f"data/raw/eurusd_strategy_data_{days_back}d_{interval}.csv" # Changed cache name
    cache_path = Path(cache_file_name)

    if cache_path.exists():
        logger.info(f"📂 Найден кэш данных для бэктестинга: {cache_path}, загружаем...")
        df_market = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df_market.index = pd.to_datetime(df_market.index).tz_localize(None) # Ensure tz-naive
        return df_market

    logger.info("Кэш не найден. Генерируем данные для бэктестинга...")
    market_ticker = yf.Ticker("EURUSD=X") # Changed to EURUSD
    df_market = market_ticker.history(period=f"{days_back}d", interval=interval)
    if df_market.empty:
        logger.error("Не удалось загрузить данные EUR/USD. Прерывание.")
        raise ValueError("EUR/USD data could not be fetched.")
    df_market.index = pd.to_datetime(df_market.index).tz_localize(None)
    logger.info(f"💾 Загружены сырые данные EUR/USD: {len(df_market)} записей")
    logger.info("DEBUG: Head of raw EUR/USD data from yfinance:")
    logger.info(df_market.head()) # DEBUG PRINT

    # Ensure basic columns are present
    if not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(df_market.columns):
        # Forex data from yfinance might not always have 'Volume'. Handle this.
        if 'Volume' not in df_market.columns:
            logger.warning("Данные по объему (Volume) отсутствуют для EURUSD=X. Устанавливаем в 0.")
            df_market['Volume'] = 0 
        if not {'Open', 'High', 'Low', 'Close'}.issubset(df_market.columns):
             raise ValueError("Missing one or more required OHLC columns in yfinance data for EURUSD=X.")


    # Calculate indicators using pandas_ta as per notebook
    df_market["VWAP"] = ta.vwap(df_market['High'], df_market['Low'], df_market['Close'], df_market['Volume'])
    df_market['RSI_16'] = ta.rsi(df_market['Close'], length=16)
    bbands = ta.bbands(df_market['Close'], length=14, std=2.0)
    df_market = df_market.join(bbands[['BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0']])
    df_market['ATRr_7'] = ta.atr(df_market['High'], df_market['Low'], df_market['Close'], length=7)

    # Implement VWAPSignal from notebook
    backcandles = 15
    VWAPsignal = np.zeros(len(df_market), dtype=int)
    if 'VWAP' not in df_market.columns or df_market['VWAP'].isnull().all():
        logger.warning("VWAP not calculated or all NaN, VWAPSignal will be 0.")
    else:
        for row in range(backcandles, len(df_market)):
            upt = 1
            dnt = 1
            if df_market['VWAP'].iloc[row-backcandles:row+1].isnull().any() or \
               df_market['Open'].iloc[row-backcandles:row+1].isnull().any() or \
               df_market['Close'].iloc[row-backcandles:row+1].isnull().any():
                continue 

            for i in range(row-backcandles, row+1):
                if max(df_market['Open'].iloc[i], df_market['Close'].iloc[i]) >= df_market['VWAP'].iloc[i]:
                    dnt=0
                if min(df_market['Open'].iloc[i], df_market['Close'].iloc[i]) <= df_market['VWAP'].iloc[i]:
                    upt=0
            if upt==1 and dnt==1: VWAPsignal[row]=3 
            elif upt==1: VWAPsignal[row]=2
            elif dnt==1: VWAPsignal[row]=1
    df_market['VWAPSignal'] = VWAPsignal

    # Implement TotalSignal from notebook
    TotSignal = np.zeros(len(df_market), dtype=int)
    required_cols_for_totalsignal = ['VWAPSignal', 'Close', 'BBL_14_2.0', 'RSI_16', 'BBU_14_2.0']
    if not all(col in df_market.columns for col in required_cols_for_totalsignal) or \
       any(df_market[col].isnull().all() for col in required_cols_for_totalsignal):
        logger.warning(f"One or more required columns for TotalSignal are missing or all NaN: {required_cols_for_totalsignal}. TotalSignal will be 0.")
    else:
        for row_idx in range(backcandles, len(df_market)): 
            signal_val = 0
            if df_market[required_cols_for_totalsignal].iloc[row_idx].isnull().any():
                continue

            if (df_market['VWAPSignal'].iloc[row_idx]==2 and
                df_market['Close'].iloc[row_idx] <= df_market['BBL_14_2.0'].iloc[row_idx] and
                df_market['RSI_16'].iloc[row_idx] < 45):
                signal_val = 2 # Buy signal
            elif (df_market['VWAPSignal'].iloc[row_idx]==1 and
                  df_market['Close'].iloc[row_idx] >= df_market['BBU_14_2.0'].iloc[row_idx] and
                  df_market['RSI_16'].iloc[row_idx] > 55):
                signal_val = 1 # Sell signal
            TotSignal[row_idx] = signal_val
    df_market['TotalSignal'] = TotSignal
    
    logger.info("💾 Рассчитаны индикаторы и сигналы из ноутбука для EUR/USD.")
    df_market.to_csv(cache_path)
    logger.info(f"💾 Сохранены данные для бэктестинга: {len(df_market)} записей в {cache_path}")
    return df_market.dropna()

# Define the strategy based on the notebook
class VWAPScalpingStrategy(Strategy):
    # Parameters from notebook (can be optimized later)
    rsi_long_entry = 45
    rsi_short_entry = 55
    atr_multiplier_sl = 1.2
    atr_multiplier_tp_ratio = 1.5 # TP/SL Ratio

    def init(self):
        # Pre-calculate signals or access them as data columns
        # The backtesting library expects data columns to be named as they are in the DataFrame
        # TotalSignal: 2 for Buy, 1 for Sell, 0 for Hold
        # ATRr_7 is the ATR column from pandas_ta
        pass # Signals are already in the data

    def next(self):
        # Ensure ATRr_7 is not NaN
        if np.isnan(self.data.ATRr_7[-1]):
            return

        slatr = self.atr_multiplier_sl * self.data.ATRr_7[-1]
        
        # Close positions based on extreme RSI (from notebook)
        # This logic might need adjustment or removal if it conflicts with TotalSignal
        # For now, let's keep it simple and only use TotalSignal for entries.
        # if len(self.trades) > 0:
        #     if self.trades[-1].is_long and self.data.RSI_16[-1] >= 90: # Using RSI_16
        #         self.trades[-1].close()
        #     elif self.trades[-1].is_short and self.data.RSI_16[-1] <= 10: # Using RSI_16
        #         self.trades[-1].close()

        # Entry logic based on TotalSignal
        if not self.position: # Only enter if no open position
            if self.data.TotalSignal[-1] == 2: # Buy Signal
                sl = self.data.Close[-1] - slatr
                tp = self.data.Close[-1] + slatr * self.atr_multiplier_tp_ratio
                self.buy(sl=sl, tp=tp)
            elif self.data.TotalSignal[-1] == 1: # Sell Signal
                sl = self.data.Close[-1] + slatr
                tp = self.data.Close[-1] - slatr * self.atr_multiplier_tp_ratio
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    logger.info("🚀 Запуск бэктестинга стратегии VWAP Scalping (5-минутный интервал)...")
    
    # Get 5-minute data for a suitable period (e.g., 59 days)
    data_df = get_strategy_data(days_back=59, interval="5m") 

    if data_df.empty or len(data_df) < 50: # Need enough data for backtesting
        logger.error("Недостаточно данных для бэктестинга после обработки.")
    else:
        # Ensure column names match what backtesting.py expects (Open, High, Low, Close, Volume)
        # Our yfinance data should already be in this format.
        # Add any custom signal columns needed by the strategy.
        # 'TotalSignal' and 'ATRr_7' are used by the strategy.
        
        bt = Backtest(data_df, VWAPScalpingStrategy, cash=10000, commission=.001, margin=1.0)
        stats = bt.run()
        
        logger.info("\n📊 --- Результаты Бэктестинга ---")
        print(stats)
        
        # Print individual trades if needed
        # print("\n📈 --- Сделки ---")
        # print(stats._trades)

        try:
            plot_filename = f"reports/backtest_vwap_scalping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path("reports").mkdir(parents=True, exist_ok=True)
            bt.plot(filename=plot_filename, open_browser=False)
            logger.info(f"📈 График бэктестинга сохранен: {plot_filename}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении графика бэктестинга: {e}")

    logger.info("✅ Бэктестинг завершен.")
