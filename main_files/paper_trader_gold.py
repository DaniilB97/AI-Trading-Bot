import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import joblib
from datetime import datetime, timedelta, timezone
import time
import logging
from dotenv import load_dotenv

# Импортируем необходимые классы из ваших других скриптов
from capital_request import CapitalComAPI
from train_gru_scalper_v2 import ScalpingGRU

# Настройка
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- КОНФИГУРАЦИЯ БОТА ---
CHAMPION_PARAMS = {
    'buy_threshold': 0.59,
    'sell_threshold': 0.46,
    'min_confidence_diff': 0.13,
    'trade_size_percent': 0.1,
    'max_trades_per_day': 65,
    'min_trade_duration': 5,
    'max_drawdown_stop': 0.28,
    'volatility_filter': True,
    'trend_filter': True,
    'volume_filter': True,
    'use_sl_tp': False,
    'sl_atr_multiplier': 2.0,
    'tp_atr_multiplier': 4.0
}

EPIC = "GOLD" 
RESOLUTION = "MINUTE_5"
INTERVAL_SECONDS = 5 * 60

# Пути к вашей V2 модели
MODEL_PATH = "gru_model_v2_ETHUSDT.pth"
SCALER_PATH = "gru_scaler_v2_ETHUSDT.joblib"


class LiveTradingBot:
    def __init__(self, params):
        self.params = params
        
        self.device = torch.device("cpu")
        self.scaler = joblib.load(SCALER_PATH)
        self.model = ScalpingGRU(input_dim=5, hidden_dim=128, num_layers=2).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        logger.info("🤖 V2 Model and Scaler loaded successfully.")

        self.api = CapitalComAPI(
            os.getenv("CAPITAL_API_KEY"),
            os.getenv("CAPITAL_IDENTIFIER"),
            os.getenv("CAPITAL_PASSWORD")
        )
        if not self.api.login_and_get_tokens():
            raise ConnectionError("Failed to login to Capital.com API.")
        
        self.data_df = pd.DataFrame()

    def get_market_data(api,sequence_length):
        logger.info("Fetching latest market data...")
        try:
            to_date = datetime.now(timezone.utc)
        
        # --- START OF MODIFIED LOGIC ---
        # We need SEQUENCE_LENGTH bars for the model. Indicators can consume
        # up to 50 bars for their warm-up period. Let's fetch enough for both.
            indicator_warmup_period = 50 
            required_bars = sequence_length + indicator_warmup_period
        
        # Fetching 5-minute data, so multiply by 5
            from_date = to_date - timedelta(minutes=5 * required_bars)
        # --- END OF MODIFIED LOGIC ---

            prices = api.get_historical_prices(
                epic=EPIC,
                resolution="5_MINUTE",
                from_date=from_date.strftime("%Y-%m-%dT%H:%M:%S"),
                to_date=to_date.strftime("%Y-%m-%dT%H:%M:%S")
        )

            if not prices or 'prices' not in prices or not prices['prices']:
                logger.warning("No price data received from API.")
                return None

            price_list = prices['prices']
            df = pd.DataFrame(price_list)

            if 'snapshotTime' not in df.columns:
                logger.error("'snapshotTime' column is missing from price data.")
                return None

            df['Time'] = pd.to_datetime(df['snapshotTime'])
            df.set_index('Time', inplace=True)
        
        # Extracting OHLCV data more safely
            ohlc = {
                'Open': df['openPrice'].apply(lambda x: x.get('bid') if isinstance(x, dict) else x),
                'High': df['highPrice'].apply(lambda x: x.get('bid') if isinstance(x, dict) else x),
                'Low': df['lowPrice'].apply(lambda x: x.get('bid') if isinstance(x, dict) else x),
                'Close': df['closePrice'].apply(lambda x: x.get('bid') if isinstance(x, dict) else x),
                'Volume': df['lastTradedVolume']
            }
            temp_df = pd.DataFrame(ohlc).astype(float)
            temp_df.sort_index(inplace=True)

            temp_df.ta.strategy(ta.Strategy(name="common", ta=STRATEGY_CONFIG))
        
        # Add VWAP separately
            temp_df['VWAP'] = ta.vwap(temp_df['High'], temp_df['Low'], temp_df['Close'], temp_df['Volume'].replace(0, np.nan))
        
            temp_df.dropna(inplace=True)
            return temp_df

        except Exception as e:
            logger.error(f"Error in fetch_market_data: {e}", exc_info=True)
            return None

    
    

    def create_ohlc_df(self, prices_list: list) -> pd.DataFrame:
        """Конвертирует данные из API в pandas DataFrame."""
        data = []
        for price_point in prices_list:
            data.append({
                'Datetime': pd.to_datetime(price_point['snapshotTime']).tz_localize('UTC'),
                'Open': float(price_point['openPrice']['bid']),
                'High': float(price_point['highPrice']['bid']),
                'Low': float(price_point['lowPrice']['bid']),
                'Close': float(price_point['closePrice']['bid']),
                'Volume': 0 
            })
        df = pd.DataFrame(data)
        return df.set_index('Datetime')

    def make_decision(self, open_positions):
        """Анализирует данные и принимает торговое решение."""
        position_for_epic = None
        for pos in open_positions:
            if pos.get('market', {}).get('epic') == EPIC:
                position_for_epic = pos
                break
        
        temp_df = self.data_df.copy()
        temp_df.ta.rsi(length=14, append=True)
        temp_df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        temp_df.ta.cci(length=14, append=True)
        temp_df['Price_Change_5'] = temp_df['Close'].pct_change(periods=5)
        temp_df['VWAP'] = ta.vwap(temp_df['High'], temp_df['Low'], temp_df['Close'], temp_df['Volume'].replace(0, np.nan))
        temp_df.dropna(inplace=True)

        sequence_length = self.params.get('sequence_length', 35)
        if len(temp_df) < sequence_length:
            logger.warning("Not enough data to form a sequence after indicator calculation.")
            return

        feature_cols = ['RSI_14', 'VWAP', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5']
        
        last_sequence = temp_df[feature_cols].tail(sequence_length).values
        scaled_sequence = self.scaler.transform(last_sequence)
        sequence_tensor = torch.from_numpy(scaled_sequence).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(sequence_tensor).item()
        
        logger.info(f"Model Prediction: {prediction:.4f}")

        if position_for_epic is None: 
            if prediction > self.params['buy_threshold']:
                logger.info(f"DECISION: Open new LONG position for {EPIC}.")
                # self.api.create_position(epic=EPIC, direction="BUY", size=1) 
        else:
            direction = position_for_epic.get('position', {}).get('direction')
            if direction == "BUY" and prediction < self.params['sell_threshold']:
                deal_id = position_for_epic.get('position', {}).get('dealId')
                logger.info(f"DECISION: Close LONG position {deal_id} for {EPIC}.")
                # self.api.close_position(deal_id)

    def run(self):
        """Основной цикл работы бота."""
        logger.info(f"📈 Starting Live Trading Bot for {EPIC} on DEMO account...")
        logger.info(f"Using parameters: {self.params}")
        
        while True:
            logger.info("="*50)
            
            # --- ИЗМЕНЕНИЕ: Получаем информацию о счете ---
            account_details = self.api.get_account_details()
            if account_details and account_details.get('accounts'):
                # Предполагаем, что нас интересует первый счет
                acc = account_details['accounts'][0]
                balance = acc.get('balance', {}).get('balance', 'N/A')
                pnl = acc.get('balance', {}).get('pnl', 'N/A')
                logger.info(f"💰 Account Balance: {balance} | P&L: {pnl}")
            else:
                logger.warning("Could not retrieve account details.")
            
            # open_positions = self.api.get_open_positions()
            open_positions = [] # Заглушка, пока функция не реализована

            if open_positions is not None:
                logger.info(f"Found {len(open_positions)} open positions.")
                if self.get_market_data(self.params['sequence_length']):
                    self.make_decision(open_positions)
            else:
                logger.error("Failed to get open positions from the API.")
            
            logger.info(f"Waiting for {INTERVAL_SECONDS // 60} minutes until next check...")
            time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    bot = LiveTradingBot(params=CHAMPION_PARAMS)
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
