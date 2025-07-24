# simplified_trading_bot.py
# Trading bot with all PyQt GUI removed, keeping only the core logic

import os
import sys
import time
import json
import logging
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timezone
import threading
from pathlib import Path

# --- AI & Data Libraries ---
import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --- Helper Libraries ---
from dotenv import load_dotenv
# Real Capital.com API client
from capital_request import CapitalComAPI
# Real News API client
from real_news_api import RealNewsAPIClient, get_advanced_sentiment

# --- Configuration ---
load_dotenv()

# Check if we have all required environment variables
required_env_vars = ['CAPITAL_API_KEY', 'CAPITAL_IDENTIFIER', 'CAPITAL_PASSWORD']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {missing_vars}")
    print("❌ Please set the following environment variables in your .env file:")
    for var in missing_vars:
        print(f"   {var}=your_value_here")
    sys.exit(1)

# --- RL Model & Trading Configuration ---
TRADING_EPIC = "GOLD"
RESOLUTION = "HOUR"
UPDATE_INTERVAL_SECONDS = 60 * 60  # Check every hour

MODEL_PATH = "rl_gold_trader_model_20250624_1316.pth"
# TRADE_SIZE = 1  # 🔥 УБИРАЕМ - теперь динамический!
LOOKBACK_WINDOW = 30

# Feature columns that the model expects
FEATURE_COLUMNS = ['RSI_14', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5', 'sentiment']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper classes and placeholder functions ---

class NewsAPIClient:
    """
    DEPRECATED: Legacy placeholder class - now replaced with RealNewsAPIClient
    Keeping for backward compatibility only
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.warning("Using deprecated NewsAPIClient. Please use RealNewsAPIClient instead.")

    def get_news(self, query: str, language: str = "en") -> List[Dict[str, str]]:
        logger.warning("Using dummy news data. Real API client should be used instead.")
        return [
            {"title": "Gold prices surge amid economic uncertainty and market volatility."},
            {"title": "Investors flock to gold as a safe-haven asset."},
            {"title": "Analysts predict a bullish trend for gold in the coming weeks."},
            {"title": "Gold slips slightly as dollar strengthens."},
        ]

def get_sentiment(text: str) -> float:
    """
    DEPRECATED: Simple sentiment analysis - now replaced with get_advanced_sentiment
    """
    logger.warning("Using deprecated get_sentiment. Please use get_advanced_sentiment instead.")
    return get_advanced_sentiment(text)
    
def get_market_status():
    """
    Checks the current open/closed status of major financial markets.
    Returns a dictionary with the status of each market.
    """
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).time()
    status = {}
    
    # London (approx 8am - 4:30pm UTC during standard time)
    is_london_open = now_utc.hour >= 8 and (now_utc.hour < 16 or (now_utc.hour == 16 and now_utc.minute < 30))
    status['London'] = "OPEN" if is_london_open else "CLOSED"
    
    # New York & Toronto (approx 2:30pm - 9pm UTC during standard time)
    is_ny_open = (now_utc.hour == 14 and now_utc.minute >= 30) or (now_utc.hour > 14 and now_utc.hour < 21)
    status['New York'] = "OPEN" if is_ny_open else "CLOSED"
    status['Toronto'] = status['New York']
    
    return status

class DynamicPositionSizer:
    """Динамический размер позиции для live торговли"""
    
    def __init__(self, 
                 min_position_pct=0.05,  # 5% минимум
                 max_position_pct=0.20,  # 20% максимум  
                 base_position_pct=0.10): # 10% базовый
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.base_position_pct = base_position_pct
        
    def calculate_trade_size(self, 
                           action_probabilities: torch.Tensor,
                           sentiment_score: float,
                           account_balance: float,
                           current_price: float,
                           epic: str = "GOLD") -> tuple:
        """
        Рассчитывает размер сделки для Capital.com API
        
        Returns: (trade_size, position_pct)
        """
        
        # 1. Уверенность модели (максимальная вероятность)
        max_prob = torch.max(action_probabilities).item()
        model_confidence = max_prob
        
        # 2. Сила sentiment сигнала
        sentiment_strength = abs(sentiment_score)
        
        # 3. Комбинированная уверенность
        combined_confidence = (model_confidence * 0.7 + sentiment_strength * 0.3)
        
        # 4. Рассчитываем процент от депозита
        position_pct = (self.min_position_pct + 
                       (self.max_position_pct - self.min_position_pct) * combined_confidence)
        
        # Ограничиваем пределами
        position_pct = np.clip(position_pct, self.min_position_pct, self.max_position_pct)
        
        # 5. Конвертируем в размер сделки
        position_value = account_balance * position_pct
        #leverage = 100
        trade_size = position_value
        #trade_size = (position_value / current_price) / leverage
        
        # Округляем до разумного количества знаков
        if epic == "GOLD":
            trade_size = round(trade_size, 2)  # Для золота 2 знака после запятой
        else:
            trade_size = round(trade_size, 4)  # Для других активов
        
        # Минимальный размер сделки
        min_trade_size = 0.1 if epic == "GOLD" else 0.01
        trade_size = max(trade_size, min_trade_size)
        
        return trade_size, position_pct

# --- RL Agent and Model Definition ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim), 
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state): 
        return self.actor(state), self.critic(state)

class BalanceAndPnLTracker:
    def __init__(self, data_file_path: str = "balance_history.json"):
        # ❌ БЫЛО:
        # self.data_file_path = Path(data_file_path)
        
        # ✅ СТАЛО (без Path):
        self.data_file_path = data_file_path
        
        self.balance_history: List[Dict] = []
        self.initial_balance: Optional[float] = None
        self.load_history()
    
    def load_history(self):
        """Загружает историю баланса из файла"""
        # ❌ БЫЛО:
        # if self.data_file_path.exists():
        
        # ✅ СТАЛО:
        if os.path.exists(self.data_file_path):
            try:
                with open(self.data_file_path, 'r') as f:
                    data = json.load(f)
                    self.balance_history = data.get('history', [])
                    self.initial_balance = data.get('initial_balance')
                    
                logging.info(f"📊 Loaded {len(self.balance_history)} balance records")
                if self.initial_balance:
                    logging.info(f"💰 Initial balance: ${self.initial_balance:,.2f}")
            except Exception as e:
                logging.error(f"Error loading balance history: {e}")
        else:
            logging.info("📊 No balance history file found, starting fresh")
    
    def save_history(self):
        """Сохраняет историю баланса в файл"""
        try:
            data = {
                'initial_balance': self.initial_balance,
                'history': self.balance_history,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving balance history: {e}")
    
    def record_balance(self, balance: float, equity: float, available: float, 
                      positions_value: float = 0, metadata: Dict = None):
        """
        Записывает текущий баланс в историю
        """
        
        # Устанавливаем начальный баланс при первой записи
        if self.initial_balance is None:
            self.initial_balance = balance
            logging.info(f"🎯 Set initial balance: ${balance:,.2f}")
        
        # Создаем запись
        record = {
            'timestamp': datetime.now().isoformat(),
            'balance': balance,
            'equity': equity,
            'available': available,
            'positions_value': positions_value,
            'pnl_absolute': balance - self.initial_balance,
            'pnl_percentage': ((balance - self.initial_balance) / self.initial_balance) * 100,
            'metadata': metadata or {}
        }
        
        # Добавляем в историю
        self.balance_history.append(record)
        
        # Сохраняем только последние 10000 записей
        if len(self.balance_history) > 10000:
            self.balance_history = self.balance_history[-10000:]
        
        # Сохраняем в файл
        self.save_history()
        
        return record
    
    def get_balance_for_period(self, period: str, custom_start: Optional[datetime] = None, 
                              custom_end: Optional[datetime] = None) -> Dict:
        """
        Получает P&L за указанный период
        """
        
        if not self.balance_history or self.initial_balance is None:
            return {
                'error': 'No balance history available',
                'period': period,
                'pnl_absolute': 0,
                'pnl_percentage': 0
            }
        
        now = datetime.now()
        
        # Определяем временные рамки
        if period == 'custom' and custom_start and custom_end:
            start_time = custom_start
            end_time = custom_end
        else:
            period_map = {
                '1d': timedelta(days=1),
                '1w': timedelta(weeks=1),
                '1m': timedelta(days=30),
                '3m': timedelta(days=90),
                '6m': timedelta(days=180),
                '1y': timedelta(days=365)
            }
            
            if period not in period_map:
                return {'error': f'Invalid period: {period}'}
            
            start_time = now - period_map[period]
            end_time = now
        
        # Фильтруем записи по периоду
        period_records = []
        for record in self.balance_history:
            record_time = datetime.fromisoformat(record['timestamp'])
            if start_time <= record_time <= end_time:
                period_records.append(record)
        
        if not period_records:
            return {
                'error': f'No data available for period {period}',
                'period': period,
                'pnl_absolute': 0,
                'pnl_percentage': 0
            }
        
        # Анализируем данные за период
        first_record = period_records[0]
        last_record = period_records[-1]
        
        start_balance = first_record['balance']
        end_balance = last_record['balance']
        
        pnl_absolute = end_balance - start_balance
        pnl_percentage = ((end_balance - start_balance) / start_balance) * 100 if start_balance > 0 else 0
        
        # Находим максимум и минимум за период
        balances = [r['balance'] for r in period_records]
        max_balance = max(balances)
        min_balance = min(balances)
        
        # Дополнительная статистика
        total_pnl_from_start = last_record['pnl_absolute']
        total_pnl_percentage = last_record['pnl_percentage']
        
        return {
            'period': period,
            'start_date': first_record['timestamp'],
            'end_date': last_record['timestamp'],
            'start_balance': start_balance,
            'end_balance': end_balance,
            'pnl_absolute': pnl_absolute,
            'pnl_percentage': pnl_percentage,
            'max_balance': max_balance,
            'min_balance': min_balance,
            'total_pnl_from_start': total_pnl_from_start,
            'total_pnl_percentage': total_pnl_percentage,
            'records_count': len(period_records),
            'chart_data': {
                'timestamps': [r['timestamp'] for r in period_records],
                'balances': [r['balance'] for r in period_records],
                'pnl_absolute': [r['pnl_absolute'] for r in period_records],
                'pnl_percentage': [r['pnl_percentage'] for r in period_records]
            }
        }
    
    def get_current_balance_info(self) -> Dict:
        """Получает текущую информацию о балансе"""
        if not self.balance_history:
            return {'error': 'No balance data available'}
        
        latest = self.balance_history[-1]
        return {
            'current_balance': latest['balance'],
            'current_equity': latest['equity'],
            'available': latest['available'],
            'positions_value': latest['positions_value'],
            'initial_balance': self.initial_balance,
            'total_pnl_absolute': latest['pnl_absolute'],
            'total_pnl_percentage': latest['pnl_percentage'],
            'last_updated': latest['timestamp']
        }
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        logger.info(f"✅ RL model successfully loaded from {path}")

    def select_action_with_confidence(self, state):
        """Возвращает действие И вероятности для динамического sizing"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs, _ = self.policy(state_tensor.unsqueeze(0))
            action = torch.argmax(action_probs).item()
            
            return action, action_probs.squeeze(0)  # Действие + вероятности

# --- Core Trading Logic ---
class TradingBot:
    def __init__(self, api: CapitalComAPI, news_api: NewsAPIClient, agent: PPOAgent):
        self.api = api
        self.agent = agent
        self.news_api = news_api
        self.scaler = MinMaxScaler()
        self.is_running = True
        self.is_scaler_fitted = False
        self.balance_tracker = BalanceAndPnLTracker("balance_history.json") # for dynamic balance 

        self.position_sizer = DynamicPositionSizer(
            min_position_pct=0.05,  # 5% minimum
            max_position_pct=0.20,  # 20% maximum  
            base_position_pct=0.10  # 10% base
        )

    def get_dynamic_balance(self) -> Tuple[float, Dict]:
        """
        Динамически получает баланс и записывает в историю
        
        Returns:
            (current_balance, account_info_dict)
        """
        
        account_details = self.api.get_account_details()
        positions = self.api.get_open_positions()
        
        if not account_details or 'accounts' not in account_details:
            self.log_message("⚠️ Could not get account details, using cached balance", "error")
            cached_info = self.balance_tracker.get_current_balance_info()
            return cached_info.get('current_balance', 100000), {}
        
        account_info = account_details['accounts'][0]
        balance_info = account_info.get('balance', {})
        
        # Извлекаем все возможные поля баланса
        balance = balance_info.get('balance', 0)
        equity = balance_info.get('equity', balance)
        available = balance_info.get('available', 0)
        deposit = balance_info.get('deposit', 0)
        
        # Выбираем наиболее подходящее значение как основной баланс
        current_balance = max(balance, equity, deposit) if any([balance, equity, deposit]) else 100000
        
        # Рассчитываем стоимость позиций
        positions_value = 0
        position_details = []
        
        if positions:
            for pos in positions:
                position_info = pos.get('position', {})
                market_info = pos.get('market', {})
                
                size = position_info.get('size', 0)
                level = position_info.get('level', 0)
                pnl = position_info.get('pnl', 0)
                
                position_value = size * level
                positions_value += position_value
                
                position_details.append({
                    'epic': market_info.get('epic'),
                    'instrument': market_info.get('instrumentName'),
                    'direction': position_info.get('direction'),
                    'size': size,
                    'level': level,
                    'pnl': pnl,
                    'value': position_value
                })
        
        # Логируем подробную информацию
        self.log_message(f"💰 Account Balance Details:", "info")
        self.log_message(f"   📊 Balance: ${balance:,.2f}", "info")
        self.log_message(f"   📊 Equity: ${equity:,.2f}", "info")
        self.log_message(f"   📊 Available: ${available:,.2f}", "info")
        self.log_message(f"   📊 Deposit: ${deposit:,.2f}", "info")
        self.log_message(f"   📊 Positions Value: ${positions_value:,.2f}", "info")
        self.log_message(f"   ✅ Using for calculations: ${current_balance:,.2f}", "info")
        
        # Записываем в историю баланса
        metadata = {
            'positions_count': len(positions) if positions else 0,
            'position_details': position_details,
            'api_balance_fields': balance_info
        }
        
        self.balance_tracker.record_balance(
            balance=current_balance,
            equity=equity,
            available=available,
            positions_value=positions_value,
            metadata=metadata
        )
        
        return current_balance, {
            'balance': balance,
            'equity': equity,
            'available': available,
            'deposit': deposit,
            'positions_value': positions_value,
            'positions_count': len(positions) if positions else 0
        }
    
    
    def get_pnl_for_period(self, period: str, custom_start: Optional[datetime] = None, 
                          custom_end: Optional[datetime] = None) -> Dict:
        """
        Получает P&L за указанный период
        
        Args:
            period: '1d', '1w', '1m', '3m', '6m', '1y', 'custom'
        """
        return self.balance_tracker.get_balance_for_period(period, custom_start, custom_end)

    def print_pnl_summary(self):
        """Выводит сводку P&L за разные периоды"""
        periods = ['1d', '1w', '1m', '3m', '6m', '1y']
        
        self.log_message("📊 P&L Summary:", "info")
        
        for period in periods:
            pnl_data = self.get_pnl_for_period(period)
            
            if 'error' not in pnl_data:
                pnl_abs = pnl_data['pnl_absolute']
                pnl_pct = pnl_data['pnl_percentage']
                
                period_label = {
                    '1d': 'Day', '1w': 'Week', '1m': 'Month', 
                    '3m': '3 Months', '6m': '6 Months', '1y': 'Year'
                }[period]
                
                status = "📈" if pnl_abs >= 0 else "📉"
                
                self.log_message(f"   {status} {period_label}: {pnl_abs:+.2f} ({pnl_pct:+.2f}%)", "info")
            else:
                self.log_message(f"   ⚠️ {period}: {pnl_data['error']}", "info")


    def log_message(self, message: str, level: str = "info"):
        """Simplified logging without GUI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_map = {
            "info": "INFO",
            "decision": "DECISION", 
            "trade": "TRADE",
            "error": "ERROR"
        }
        log_level = level_map.get(level, "INFO")
        logger.info(f"[{log_level}] {message}")

    def fit_scaler_on_startup(self):
        """
        Fits scaler on historical data for normalizing market features.
        Also validates that we can access real market data.
        """
        self.log_message("Initializing... Getting data to fit scaler.")
        
        # First, let's find the correct EPIC for gold
        self.log_message("🔍 Searching for available gold markets...")
        gold_markets = self.api.get_market_info("gold")
        
        if gold_markets:
            self.log_message(f"Found {len(gold_markets)} gold-related markets:")
            for i, market in enumerate(gold_markets[:3]):  # Show first 3
                epic = market.get('epic', 'N/A')
                name = market.get('instrumentName', 'N/A')
                self.log_message(f"   {i+1}. Epic: {epic} - {name}")
            
            # Use the first gold market found or update TRADING_EPIC
            if gold_markets:
                found_epic = gold_markets[0]['epic']
                global TRADING_EPIC
                TRADING_EPIC = found_epic
                self.log_message(f"✅ Using EPIC: {TRADING_EPIC}")
        else:
            self.log_message("⚠️ No gold markets found, using default EPIC: GOLD")
        
        # Now get historical data
        self.log_message(f"📈 Fetching historical data for {TRADING_EPIC}...")
        market_data = self.api.get_historical_prices(TRADING_EPIC, RESOLUTION, max_points=500)
        
        if not market_data or 'prices' not in market_data:
            self.log_message("Could not get initial data for scaler. Bot will not trade.", "error")
            return False

        prices_count = len(market_data['prices'])
        self.log_message(f"✅ Retrieved {prices_count} historical price points")

        df = self.create_ohlc_df(market_data['prices'])
        df_with_indicators = self.calculate_indicators(df, sentiment_score=0.0)

        if len(df_with_indicators) > 0:
            # Fit scaler ONLY on market feature columns
            self.scaler.fit(df_with_indicators[FEATURE_COLUMNS].values)
            self.is_scaler_fitted = True
            self.log_message("✅ Data scaler successfully fitted.")
            
            # Show some sample data
            latest_price = df['Close'].iloc[-1]
            self.log_message(f"📊 Latest {TRADING_EPIC} price: ${latest_price:.2f}")
            return True
        else:
            self.log_message("Insufficient data to fit scaler.", "error")
            return False

    def run_trading_cycle(self):
        """
        Executes one complete trading cycle: gets data, constructs state,
        gets action from model and executes trade.
        """
        if not self.is_scaler_fitted:
            self.log_message("Scaler not fitted. Stopping cycle.", "error")
            return

        # Get current positions
        open_positions = self.api.get_open_positions()
        if open_positions is None:
            self.log_message("Could not get open positions from API.", "error")
            return
        
        position_for_epic = next((p for p in open_positions if p.get('market',{}).get('epic') == TRADING_EPIC), None)

        current_direction = None
        current_size = 0
        if position_for_epic:
            current_direction = position_for_epic.get('position', {}).get('direction')  # 'BUY' или 'SELL'
            current_size = position_for_epic.get('position', {}).get('size', 0)
            current_pnl = position_for_epic.get('position', {}).get('pnl', 0)
            
            self.log_message(f"📊 Current position: {current_direction} {current_size} units (P&L: {current_pnl:+.2f})", "info")

        # Get market data
        market_data = self.api.get_historical_prices(TRADING_EPIC, RESOLUTION, max_points=LOOKBACK_WINDOW + 50)
        if not market_data or 'prices' not in market_data:
            self.log_message("Could not get market data from API.", "error")
            return
        
        df = self.create_ohlc_df(market_data['prices'])

        # Get sentiment from news - REAL NEWS API with MarketAux sentiments
        self.log_message("📰 Getting REAL sentiment from news APIs (prioritizing MarketAux ready sentiments)...")
        try:
            articles, live_sentiment_score = self.news_api.get_news_with_sentiment(query="gold", language="en")
            
            if articles:
                self.log_message(f"Found {len(articles)} news articles")
                
                # Show breakdown of sentiment analysis
                marketaux_count = sum(1 for a in articles if a.get('has_ready_sentiment', False))
                newsapi_count = len(articles) - marketaux_count
                
                self.log_message(f"   📊 MarketAux articles (ready sentiment): {marketaux_count}")
                self.log_message(f"   📰 NewsAPI articles (analyzed sentiment): {newsapi_count}")
                
                # Log a few examples with their sentiment sources
                for i, article in enumerate(articles[:3]):
                    title = article.get('title', '')[:60] + "..."
                    source = article.get('source', {}).get('name', 'Unknown')
                    
                    if article.get('has_ready_sentiment', False):
                        sentiment = article.get('marketaux_sentiment', 0.0)
                        confidence = article.get('sentiment_confidence', 0.0)
                        self.log_message(f"   📊 {source}: {sentiment:+.3f} (confidence: {confidence:.2f}) - {title}")
                    else:
                        # This would have been analyzed
                        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
                        if text:
                            sentiment = get_advanced_sentiment(text)
                            self.log_message(f"   📰 {source}: {sentiment:+.3f} (analyzed) - {title}")
                
                self.log_message(f"✅ Final weighted sentiment: {live_sentiment_score:.3f}")
                
                # Interpret sentiment for user
                if live_sentiment_score > 0.2:
                    sentiment_label = "😊 BULLISH"
                elif live_sentiment_score < -0.2:
                    sentiment_label = "😞 BEARISH"
                elif live_sentiment_score > 0.05:
                    sentiment_label = "🙂 SLIGHTLY POSITIVE"
                elif live_sentiment_score < -0.05:
                    sentiment_label = "🙁 SLIGHTLY NEGATIVE"
                else:
                    sentiment_label = "😐 NEUTRAL"
                
                self.log_message(f"📈 Market sentiment: {sentiment_label}")
                
            else:
                self.log_message("No fresh news articles found.")
                live_sentiment_score = 0.0
                
        except Exception as e:
            self.log_message(f"Error getting real news: {e}", "error")
            live_sentiment_score = 0.0
        
        # Construct state and get model decision with confidence
        state_vector = self.construct_state(df, position_for_epic, live_sentiment_score)
        if state_vector is None:
            self.log_message("Could not construct state vector from data.", "error")
            return
            
        # 🔥 Получаем действие И вероятности для динамического sizing
        action, action_probabilities = self.agent.select_action_with_confidence(state_vector)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        # Получаем текущий баланс счета
        current_balance, balance_details = self.get_dynamic_balance()    
        current_price = df['Close'].iloc[-1]
        
        is_weekend = datetime.now().weekday() in [5, 6]
        if is_weekend and action in [1, 2]:
            self.log_message("Weekend detected. No new trades will be executed.")
            return

        # 🔥 Рассчитываем динамический размер сделки
        if action == 0:  # HOLD
            self.log_message(f"🤖 Model decision: HOLD", "decision")
            return

        # Рассчитываем размер для новой сделки
        dynamic_trade_size, position_pct = self.position_sizer.calculate_trade_size(
            action_probabilities=action_probabilities,
            sentiment_score=live_sentiment_score,
            account_balance=current_balance,
            current_price=current_price,
            epic=TRADING_EPIC
        )

        max_confidence = torch.max(action_probabilities).item()
        self.log_message(f"🤖 Model decision: {action_map[action]}", "decision")
        self.log_message(f"📊 Model confidence: {max_confidence:.3f}", "decision")
        self.log_message(f"🎯 Suggested position size: {position_pct:.1%} = {dynamic_trade_size:.2f} units", "decision")

        # СТРАТЕГИЯ 1: ПРОСТОЕ РАСШИРЕНИЕ ПОЗИЦИЙ
        if action == 1:  # BUY сигнал
            if position_for_epic is None:
                # Нет позиции - открываем новую BUY
                self.log_message(f"📈 Opening new BUY position: {dynamic_trade_size} units", "trade")
                result = self.api.create_position(epic=TRADING_EPIC, direction="BUY", size=dynamic_trade_size)
                
            elif current_direction == "BUY":
                # Уже есть BUY позиция - можем увеличить её
                if max_confidence > 0.65:  # Только при высокой уверенности
                    self.log_message(f"📈 High confidence ({max_confidence:.3f}) - Adding to BUY position: +{dynamic_trade_size} units", "trade")
                    result = self.api.create_position(epic=TRADING_EPIC, direction="BUY", size=dynamic_trade_size)
                else:
                    self.log_message(f"⏸️ BUY signal but confidence too low ({max_confidence:.3f}) to add to existing position", "decision")
                    return
                    
            elif current_direction == "SELL":
                # Есть SELL позиция, но модель хочет BUY
                if max_confidence > 0.8:  # Очень высокая уверенность для разворота
                    deal_id = position_for_epic.get('position', {}).get('dealId')
                    self.log_message(f"🔄 Very high confidence ({max_confidence:.3f}) - Closing SELL position and opening BUY", "trade")
                    
                    # Сначала закрываем SELL
                    close_result = self.api.close_position(deal_id)
                    if close_result and close_result.get('dealReference'):
                        # Затем открываем BUY
                        result = self.api.create_position(epic=TRADING_EPIC, direction="BUY", size=dynamic_trade_size)
                    else:
                        self.log_message(f"❌ Failed to close SELL position", "error")
                        return
                else:
                    self.log_message(f"⏸️ BUY signal but confidence too low ({max_confidence:.3f}) to reverse SELL position", "decision")
                    return

        elif action == 2:  # SELL сигнал
            if position_for_epic is None:
                # Нет позиции - открываем новую SELL
                self.log_message(f"📉 Opening new SELL position: {dynamic_trade_size} units", "trade")
                result = self.api.create_position(epic=TRADING_EPIC, direction="SELL", size=dynamic_trade_size)
                
            elif current_direction == "SELL":
                # Уже есть SELL позиция - можем увеличить её
                if max_confidence > 0.7:  # Только при высокой уверенности
                    self.log_message(f"📉 High confidence ({max_confidence:.3f}) - Adding to SELL position: +{dynamic_trade_size} units", "trade")
                    result = self.api.create_position(epic=TRADING_EPIC, direction="SELL", size=dynamic_trade_size)
                else:
                    self.log_message(f"⏸️ SELL signal but confidence too low ({max_confidence:.3f}) to add to existing position", "decision")
                    return
                    
            elif current_direction == "BUY":
                # Есть BUY позиция, но модель хочет SELL
                if max_confidence > 0.8:  # Очень высокая уверенность для разворота
                    deal_id = position_for_epic.get('position', {}).get('dealId')
                    self.log_message(f"🔄 Very high confidence ({max_confidence:.3f}) - Closing BUY position and opening SELL", "trade")
                    
                    # Сначала закрываем BUY
                    close_result = self.api.close_position(deal_id)
                    if close_result and close_result.get('dealReference'):
                        # Затем открываем SELL
                        result = self.api.create_position(epic=TRADING_EPIC, direction="SELL", size=dynamic_trade_size)
                    else:
                        self.log_message(f"❌ Failed to close BUY position", "error")
                        return
                else:
                    self.log_message(f"⏸️ SELL signal but confidence too low ({max_confidence:.3f}) to reverse BUY position", "decision")
                    return

        # Проверяем результат выполнения сделки
        if 'result' in locals() and result and result.get('dealReference'):
            self.log_message(f"✅ Trade executed successfully. Reference: {result['dealReference']}", "trade")
            self.log_message(f"💰 Trade value: ${dynamic_trade_size * current_price:.2f}", "trade")
        elif 'result' in locals():
            self.log_message(f"❌ Trade failed. API response: {result}", "error")

    def create_ohlc_df(self, prices_list: list) -> pd.DataFrame:
        """Create OHLC DataFrame from price data"""
        data = []
        for p in prices_list:
            data.append({
                'Datetime': pd.to_datetime(p['snapshotTime']), 
                'Open': float(p['openPrice']['bid']), 
                'High': float(p['highPrice']['bid']), 
                'Low': float(p['lowPrice']['bid']), 
                'Close': float(p['closePrice']['bid']), 
                'Volume': 0
            })
        return pd.DataFrame(data).set_index('Datetime')

    def calculate_indicators(self, df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
        """
        Calculates indicators that the model expects and sets sentiment score.
        """
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=14, append=True)
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['sentiment'] = sentiment_score
        df.bfill(inplace=True)
        df.dropna(inplace=True)
        return df

    def construct_state(self, df: pd.DataFrame, position: Optional[Dict], sentiment_score: float) -> Optional[np.ndarray]:
        """
        Constructs state vector with 154 features to match the model.
        Separates market features for scaling, then adds portfolio features.
        """
        df_with_indicators = self.calculate_indicators(df.copy(), sentiment_score=sentiment_score)
        if len(df_with_indicators) < LOOKBACK_WINDOW:
            return None

        # 1. Get market features part of state
        market_df = df_with_indicators[FEATURE_COLUMNS].tail(LOOKBACK_WINDOW)
        if len(market_df) < LOOKBACK_WINDOW:
            return None

        # 2. Scale ONLY market features
        market_state_scaled = self.scaler.transform(market_df.values)
        market_state = market_state_scaled.flatten()  # This will be 30 * 5 = 150 features

        # 3. Get portfolio features part of state
        balance, pnl, pos_state = 1.0, 0.0, 0
        if position:
            pos_state = 1 if position.get('position',{}).get('direction') == 'BUY' else -1
            pnl = position.get('position',{}).get('pnl', 0.0)

        # Portfolio state is NOT scaled
        portfolio_state = np.array([balance, pos_state, 0, pnl]).flatten()  # This is 4 features

        # 4. Combine both parts to form final state vector
        return np.concatenate([market_state, portfolio_state]).astype(np.float32)  # 150 + 4 = 154 features

    def print_status(self):
        """Print current account status and positions"""
        try:
            details = self.api.get_account_details()
            positions = self.api.get_open_positions()
            market_status = get_market_status()
            
            if details and 'accounts' in details:
                acc = details['accounts'][0]
                balance_info = acc.get('balance', {})
                balance_val = balance_info.get('balance', 0.0)
                pnl_val = balance_info.get('pnl', 0.0)
                print(f"\n💰 Account Balance: ${balance_val:,.2f}")
                print(f"📈 Today's P&L: ${pnl_val:,.2f}")

            print(f"\n🌍 Market Status:")
            for market, status in market_status.items():
                status_icon = "🟢" if status == "OPEN" else "🔴"
                print(f"  {status_icon} {market}: {status}")

            print(f"\n📊 Open Positions:")
            if positions:
                for pos in positions:
                    market_info = pos.get('market', {})
                    position_info = pos.get('position', {})
                    instrument_name = market_info.get('instrumentName', 'Unknown')
                    trade_size = position_info.get('size', 0)
                    open_level = position_info.get('level', 0.0)
                    pnl = position_info.get('pnl', 0.0)
                    pnl_icon = "📈" if pnl >= 0 else "📉"
                    print(f"  {pnl_icon} {instrument_name}: {trade_size} @ ${open_level:.2f} | P&L: ${pnl:.2f}")
            else:
                print("  No open positions.")
            print("-" * 50)
        except Exception as e:
            logger.error(f"Error printing status: {e}")

    def run(self):
        """Main bot loop"""
        self.log_message("🚀 Starting Trading Bot...")
        
        # Initialize scaler
        if not self.fit_scaler_on_startup():
            self.log_message("Failed to initialize bot. Exiting.", "error")
            return

        self.log_message("✅ Bot initialization complete. Starting trading loop...")
        
        cycle_count = 0
        while self.is_running:
            try:
                cycle_count += 1
                self.log_message(f"\n🔄 Trading Cycle #{cycle_count}")
                
                # Print current status
                self.print_status()
                
                # Run trading logic
                self.run_trading_cycle()
                
                self.log_message(f"Sleeping for {UPDATE_INTERVAL_SECONDS} seconds...")
                time.sleep(UPDATE_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                self.log_message("🛑 Bot stopped by user.")
                break
            except Exception as e:
                self.log_message(f"Error in trading cycle: {e}", "error")
                time.sleep(60)  # Wait 1 minute before retrying

    def stop(self):
        """Stop the bot"""
        self.is_running = False


# --- Main Application Entry Point ---
if __name__ == "__main__":
    print("🤖 Simplified Trading Bot - No GUI Version")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Trained model not found. Checked path: {MODEL_PATH}")
        sys.exit(1)

    # --- Initialize API clients ---
    print("🔐 Connecting to Capital.com API...")
    api = CapitalComAPI(
        api_key=os.getenv("CAPITAL_API_KEY"), 
        identifier=os.getenv("CAPITAL_IDENTIFIER"), 
        password=os.getenv("CAPITAL_PASSWORD")
    )
    
    print("🔑 Attempting to login and get authentication tokens...")
    if not api.login_and_get_tokens():
        print("❌ Failed to login to Capital.com API")
        print("Please check your credentials in the .env file:")
        print("   CAPITAL_API_KEY=your_api_key")
        print("   CAPITAL_IDENTIFIER=your_email")
        print("   CAPITAL_PASSWORD=your_password")
        sys.exit(1)
    
    print("✅ Successfully authenticated with Capital.com API")

    # Initialize real news API client
    print("📰 Initializing news API clients...")
    news_api = RealNewsAPIClient(
        news_api_key=os.getenv("NEWS_API_KEY"),
        marketaux_api_key=os.getenv("MARKETAUX_API_TOKEN")
    )

    # --- Initialize RL agent ---
    state_dim = LOOKBACK_WINDOW * len(FEATURE_COLUMNS) + 4  # (30 * 5) + 4 = 154
    agent = PPOAgent(state_dim, action_dim=3)  # 3 actions: HOLD, BUY, SELL
    agent.load_model(MODEL_PATH)

    # --- Create and run bot ---
    bot = TradingBot(api, news_api, agent)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    finally:
        bot.stop()
        print("👋 Bot shutdown complete.")