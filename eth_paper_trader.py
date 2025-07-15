import os
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import joblib
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager

# Импортируем архитектуру нашей V2 модели
from main_files.train_gru_scalper_v2 import ScalpingGRU

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- КОНФИГУРАЦИЯ БОТА ---
# Лучшие параметры, найденные в ходе оптимизации.
# Мы берем параметры из прогона, который показал хороший и стабильный результат.
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
    'tp_atr_multiplier': 4.0,
    'sequence_length': 37, # из лога обучения
    'rsi_period': 12 # из лога обучения
}

# Настройки инструмента
TICKER = "ETHUSDT"
INTERVAL = "5m" # 5-минутный интервал

# Пути к вашей V2 модели
MODEL_PATH = f"models/gru_model_v2_{TICKER}.pth"
SCALER_PATH = f"models/gru_scaler_v2_{TICKER}.joblib"

class PaperPortfolio:
    """Класс для управления виртуальным портфелем."""
    def __init__(self, initial_cash=10000, commission=0.001):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission = commission
        self.position = 0  # 0: нет, 1: long
        self.position_size = 0.0
        self.entry_price = 0.0
        self.equity = initial_cash
        self.total_trades = 0

    def update_equity(self, current_price):
        """Обновляет общую стоимость портфеля."""
        pnl = 0
        if self.position == 1:
            pnl = (current_price - self.entry_price) * self.position_size
        self.equity = self.cash + pnl
        
    def execute_trade(self, side, price, size_in_percent):
        """Выполняет симуляцию сделки."""
        if side == 'buy' and self.position == 0:
            trade_value = self.equity * size_in_percent
            if price > 0:
                self.position_size = trade_value / price
                commission_cost = trade_value * self.commission
                self.cash -= commission_cost
                self.entry_price = price
                self.position = 1
                self.total_trades += 1
                logger.info(f"+++ LONG POSITION OPENED | Price: {price:.2f}, Size: {self.position_size:.4f} +++")
        elif side == 'sell' and self.position == 1:
            profit = (price - self.entry_price) * self.position_size
            commission_cost = price * self.position_size * self.commission
            net_profit = profit - commission_cost
            self.cash += net_profit
            logger.info(f"--- POSITION CLOSED | Price: {price:.2f}, Net Profit: ${net_profit:.2f} ---")
            self.position = 0
            self.position_size = 0.0
            self.entry_price = 0.0

class LiveBinanceTrader:
    def __init__(self, params):
        self.params = params
        self.portfolio = PaperPortfolio()
        
        self.device = torch.device("cpu")
        self.scaler = joblib.load(SCALER_PATH)
        # Убедитесь, что input_dim=5 для новой модели
        self.model = ScalpingGRU(input_dim=5, hidden_dim=128, num_layers=2).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        logger.info("🤖 V2 Model and Scaler loaded successfully.")
        
        self.historical_data = pd.DataFrame()
        self.is_ready = False

    async def initialize_data(self, client):
        """Загружает начальные исторические данные для расчета индикаторов."""
        logger.info("Fetching initial historical data...")
        # Запрашиваем 500 последних свечей, чтобы хватило для всех индикаторов
        klines = await client.get_klines(symbol=TICKER, interval=INTERVAL, limit=500)
        
        df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Datetime', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
        
        self.historical_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.is_ready = True
        logger.info(f"Initial data loaded. Shape: {self.historical_data.shape}. Bot is ready.")

    def process_kline(self, kline_msg):
        """Обрабатывает новое сообщение от WebSocket и принимает решение."""
        if not self.is_ready or 'k' not in kline_msg:
            return

        kline = kline_msg['k']
        if kline['x']:  # Если свеча закрыта
            logger.info("="*50)
            logger.info(f"New 5m kline closed at price: {kline['c']}")
            
            # Обновляем наши исторические данные
            new_row = pd.DataFrame([{
                'Open': float(kline['o']), 'High': float(kline['h']),
                'Low': float(kline['l']), 'Close': float(kline['c']),
                'Volume': float(kline['v'])
            }], index=[pd.to_datetime(kline['T'], unit='ms')])
            
            self.historical_data = pd.concat([self.historical_data.iloc[1:], new_row])
            
            # Принимаем решение
            self.make_decision()
            
            # Обновляем и выводим состояние портфеля
            self.portfolio.update_equity(float(kline['c']))
            logger.info(f"Current Portfolio Equity: ${self.portfolio.equity:.2f} | Open Position: {'YES' if self.portfolio.position != 0 else 'NO'}")


    def make_decision(self):
        """Анализирует данные и принимает торговое решение."""
        # 1. Рассчитать индикаторы
        temp_df = self.historical_data.copy()
        temp_df.ta.rsi(length=self.params['rsi_period'], append=True)
        temp_df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        temp_df.ta.cci(length=14, append=True)
        temp_df['Price_Change_5'] = temp_df['Close'].pct_change(periods=5)
        temp_df['VWAP'] = ta.vwap(temp_df['High'], temp_df['Low'], temp_df['Close'], temp_df['Volume'].replace(0, np.nan))
        temp_df.dropna(inplace=True)

        sequence_length = self.params['sequence_length']
        if len(temp_df) < sequence_length:
            logger.warning("Not enough data to form a sequence after indicator calculation.")
            return

        # 2. Подготовить данные для модели
        feature_cols = [f'RSI_{self.params["rsi_period"]}', 'VWAP', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5']
        
        last_sequence = temp_df[feature_cols].tail(sequence_length).values
        scaled_sequence = self.scaler.transform(last_sequence)
        sequence_tensor = torch.from_numpy(scaled_sequence).float().unsqueeze(0).to(self.device)

        # 3. Получить предсказание
        with torch.no_grad():
            prediction = self.model(sequence_tensor).item()
        
        logger.info(f"Model Prediction: {prediction:.4f}")

        # 4. Принять торговое решение
        current_price = self.historical_data['Close'].iloc[-1]
        
        if self.portfolio.position == 0:
            if prediction > self.params['buy_threshold']:
                self.portfolio.execute_trade('buy', current_price, self.params['trade_size_percent'])
        elif self.portfolio.position == 1:
            if prediction < self.params['sell_threshold']:
                self.portfolio.execute_trade('sell', current_price, 1.0) # Закрываем всю позицию

async def main():
    """Основная асинхронная функция для запуска бота."""
    client = await AsyncClient.create(os.getenv("BINANCE_API"), os.getenv("BINANCE_SECRET_KEY"))
    bsm = BinanceSocketManager(client)
    
    bot = LiveBinanceTrader(params=CHAMPION_PARAMS)
    await bot.initialize_data(client)
    
    socket = bsm.kline_socket(symbol=TICKER, interval=INTERVAL)
    async with socket as stream:
        while True:
            msg = await stream.recv()
            bot.process_kline(msg)
            
    await client.close_connection()

if __name__ == "__main__":
    logger.info(f"📈 Starting Live Paper Trading Bot for {TICKER} on Binance...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)

