import os
import sys
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# --- Настройка путей ---
# Добавляем корневую директорию бэкенда в системный путь.
# Это гарантирует, что импорты из других папок (например, core, ml) будут работать корректно.
# SWF4/python-backend/core/main_api.py -> SWF4/python-backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# --- Импорт вашей существующей логики ---
# 🔥 ИСПРАВЛЕНО: TradingWorker -> TradingBot, добавлен RealNewsAPIClient
from core.capital_request import CapitalComAPI
from core.live_rl_trading import (
    PPOAgent, 
    TradingBot,  # ✅ Изменено с TradingWorker на TradingBot
    LOOKBACK_WINDOW, 
    FEATURE_COLUMNS, 
    TRADING_EPIC, 
    RESOLUTION,
    MODEL_PATH,  # ✅ Импортируем MODEL_PATH вместо TRADE_SIZE
    UPDATE_INTERVAL_SECONDS  # ✅ Добавляем импорт UPDATE_INTERVAL_SECONDS
)
from core.real_news_api import RealNewsAPIClient  # ✅ Правильный импорт новостного API

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Загрузка переменных окружения из папки config ---
config_path = BACKEND_ROOT / 'config' / '.env'
if config_path.exists():
    logging.info(f"Loading .env file from: {config_path}")
    load_dotenv(dotenv_path=config_path)
else:
    logging.warning(f".env file not found at {config_path}. Falling back to default search.")
    load_dotenv()

# --- Динамический поиск файла модели ---
# Ищем модель в той же папке, где лежит этот скрипт (т.е. в папке 'core')
MODELS_DIR = Path(__file__).resolve().parent
MODEL_NAME_PATTERN = "rl_gold_trader_model"
ABSOLUTE_MODEL_PATH = None

if MODELS_DIR.exists():
    for f in MODELS_DIR.iterdir():
        # Ищем файл, который начинается с нашего паттерна и заканчивается на .pth
        if f.name.startswith(MODEL_NAME_PATTERN) and f.name.endswith(".pth"):
            ABSOLUTE_MODEL_PATH = f
            logging.info(f"Found model file: {f.name}")
            break # Используем первый найденный файл

# --- Глобальные переменные для доступа к боту ---
trading_bot = None  # ✅ ДОБАВЛЯЕМ глобальную переменную

# --- Глобальное хранилище состояния ---
app_state: Dict[str, Any] = {
    "portfolio_metrics": { "equity": 10000.0, "cash": 10000.0, "total_trades": 0 },
    "open_positions": [],
    "chart_data": { "labels": [], "prices": [] },
    "logs": [],
}
state_lock = threading.Lock()

# --- Модели данных для FastAPI (Pydantic) ---
class PortfolioMetrics(BaseModel):
    equity: float
    cash: float
    total_trades: int

class OpenPosition(BaseModel):
    ticker: str
    size: float
    entry_price: float
    current_price: float
    pnl: float

class ChartData(BaseModel):
    labels: List[str]
    prices: List[float]

class PnLPeriodRequest(BaseModel):
    period: str  # '1d', '1w', '1m', '3m', '6m', '1y', 'custom'
    custom_start: Optional[str] = None  # ISO format datetime
    custom_end: Optional[str] = None    # ISO format datetime

class PnLPeriodResponse(BaseModel):
    period: str
    start_date: str
    end_date: str
    start_balance: float
    end_balance: float
    pnl_absolute: float
    pnl_percentage: float
    max_balance: float
    min_balance: float
    total_pnl_from_start: float
    total_pnl_percentage: float
    records_count: int
    chart_data: Dict[str, List]

class BalanceInfoResponse(BaseModel):
    current_balance: float
    current_equity: float
    available: float
    positions_value: float
    initial_balance: float
    total_pnl_absolute: float
    total_pnl_percentage: float
    last_updated: str

class PnLSummaryResponse(BaseModel):
    day: Dict[str, float]
    week: Dict[str, float] 
    month: Dict[str, float]
    three_months: Dict[str, float]
    six_months: Dict[str, float]
    year: Dict[str, float]

class LogEntry(BaseModel):
    id: str
    time: str
    message: str
    type: str

class AppStateResponse(BaseModel):
    portfolio_metrics: PortfolioMetrics
    open_positions: List[OpenPosition]
    chart_data: ChartData
    logs: List[LogEntry]

# --- Настройка FastAPI ---
app = FastAPI(title="Trading Bot API", description="API для получения данных от торгового бота для веб-интерфейса.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Логика фонового потока ---
def trading_logic_thread(api: CapitalComAPI, bot: TradingBot):  # ✅ worker -> bot
    logging.info("🚀 Background trading logic started.")
    
    # --- ИСПРАВЛЕНИЕ ОШИБКИ 'log_message' ---
    # Класс-заглушка, который имитирует поведение системы логирования бота
    original_log_message = bot.log_message
    
    def enhanced_log_message(message: str, level: str = "info"):
        # Вызываем оригинальный метод логирования
        original_log_message(message, level)
        
        # Добавляем в глобальное состояние для веб-интерфейса
        with state_lock:
            log_entry = {
                "id": f"log_{time.time()}", 
                "time": datetime.now().strftime("%H:%M:%S"), 
                "message": message, 
                "type": level
            }
            app_state["logs"].insert(0, log_entry)
            app_state["logs"] = app_state["logs"][:100]  # Храним только последние 100 записей
    
    # Заменяем метод логирования бота
    bot.log_message = enhanced_log_message
    
    # Инициализация скейлера
    if not bot.fit_scaler_on_startup():
        logging.error("Failed to initialize bot scaler")
        return

    while bot.is_running:
        try:
            logging.info("Starting new trading cycle...")
            bot.run_trading_cycle()
            
            logging.info("Updating global state...")
            details = api.get_account_details()
            positions = api.get_open_positions()
            market_data = api.get_historical_prices(TRADING_EPIC, RESOLUTION, max_points=100)
            
            with state_lock:
                # 🔥 ОБНОВЛЯЕМ СОСТОЯНИЕ ИЗ BALANCE TRACKER
                if hasattr(bot, 'balance_tracker'):  # ✅ ИСПРАВЛЕНО: bot вместо trading_worker
                    current_balance_info = bot.balance_tracker.get_current_balance_info()
                    
                    if 'error' not in current_balance_info:
                        # Обновляем метрики портфеля из трекера
                        app_state["portfolio_metrics"]["equity"] = current_balance_info['current_balance']
                        app_state["portfolio_metrics"]["cash"] = current_balance_info['available']
                        
                        # Добавляем P&L информацию
                        app_state["portfolio_metrics"]["total_pnl"] = current_balance_info['total_pnl_absolute']
                        app_state["portfolio_metrics"]["total_pnl_pct"] = current_balance_info['total_pnl_percentage']
                        app_state["portfolio_metrics"]["initial_balance"] = current_balance_info['initial_balance']
                else:
                    # Fallback к старой логике если трекер недоступен
                    if details and 'accounts' in details:
                        account_info = details['accounts'][0]
                        balance_info = account_info.get('balance', {})
                        app_state["portfolio_metrics"]["equity"] = balance_info.get('balance', 0.0)
                        app_state["portfolio_metrics"]["cash"] = balance_info.get('available', 0.0)
                
                # Обновляем данные графика
                if market_data and 'prices' in market_data:
                    prices = market_data['prices']
                    app_state["chart_data"]["labels"] = [p['snapshotTime'] for p in prices]
                    app_state["chart_data"]["prices"] = [float(p['closePrice']['bid']) for p in prices]
                
                # Обновляем открытые позиции
                formatted_positions = []
                if positions:
                    for pos in positions:
                        position_info = pos.get('position', {})
                        market_info = pos.get('market', {})
                        
                        # Получаем текущую цену
                        current_price_data = api.get_historical_prices(
                            market_info.get('epic'), "MINUTE", max_points=1
                        )
                        current_price = (
                            float(current_price_data['prices'][0]['closePrice']['bid']) 
                            if current_price_data and current_price_data.get('prices') 
                            else position_info.get('level', 0.0)
                        )
                        
                        formatted_positions.append({
                            "ticker": market_info.get('instrumentName', 'N/A'),
                            "size": position_info.get('size', 0),
                            "entry_price": position_info.get('level', 0.0),
                            "current_price": current_price,
                            "pnl": position_info.get('pnl', 0.0)
                        })
                
                app_state["open_positions"] = formatted_positions

            # Спим до следующего цикла
            sleep_duration = UPDATE_INTERVAL_SECONDS
            logging.info(f"State updated. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)

        except Exception as e:
            logging.error(f"Error in trading_logic_thread: {e}", exc_info=True)
            time.sleep(60)  # Ждем минуту перед повторной попыткой

# --- Эндпоинты API ---
@app.get("/api/state", response_model=AppStateResponse)
def get_current_state():
    """Получить текущее состояние торгового бота"""
    with state_lock:
        return app_state

@app.get("/api/health")
def health_check():
    """Проверка работоспособности API"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/balance/current", response_model=BalanceInfoResponse)
def get_current_balance():
    """Получить текущую информацию о балансе"""
    global trading_bot  # ✅ ДОБАВЛЯЕМ global
    if trading_bot and hasattr(trading_bot, 'balance_tracker'):
        balance_info = trading_bot.balance_tracker.get_current_balance_info()
        if 'error' not in balance_info:
            return balance_info
    
    return {
        'current_balance': 0,
        'current_equity': 0,
        'available': 0,  
        'positions_value': 0,
        'initial_balance': 0,
        'total_pnl_absolute': 0,
        'total_pnl_percentage': 0,
        'last_updated': datetime.now().isoformat()
    }

@app.post("/api/pnl/period", response_model=PnLPeriodResponse)
def get_pnl_for_period(request: PnLPeriodRequest):
    """Получить P&L за указанный период"""
    global trading_bot  # ✅ ДОБАВЛЯЕМ global
    
    custom_start = None
    custom_end = None
    
    if request.period == 'custom':
        if request.custom_start:
            custom_start = datetime.fromisoformat(request.custom_start)
        if request.custom_end:
            custom_end = datetime.fromisoformat(request.custom_end)
    
    if trading_bot and hasattr(trading_bot, 'balance_tracker'):
        pnl_data = trading_bot.balance_tracker.get_balance_for_period(
            request.period, 
            custom_start, 
            custom_end
        )
        
        if 'error' not in pnl_data:
            return pnl_data
    
    # Возвращаем пустой ответ при ошибке
    return {
        'period': request.period,
        'start_date': datetime.now().isoformat(),
        'end_date': datetime.now().isoformat(),
        'start_balance': 0,
        'end_balance': 0,
        'pnl_absolute': 0,
        'pnl_percentage': 0,
        'max_balance': 0,
        'min_balance': 0,
        'total_pnl_from_start': 0,
        'total_pnl_percentage': 0,
        'records_count': 0,
        'chart_data': {'timestamps': [], 'balances': [], 'pnl_absolute': [], 'pnl_percentage': []}
    }

@app.get("/api/pnl/summary", response_model=PnLSummaryResponse)
def get_pnl_summary():
    """Получить сводку P&L за все периоды"""
    global trading_bot  # ✅ ДОБАВЛЯЕМ global
    
    periods = ['1d', '1w', '1m', '3m', '6m', '1y']
    period_names = ['day', 'week', 'month', 'three_months', 'six_months', 'year']
    
    summary = {}
    
    if trading_bot and hasattr(trading_bot, 'balance_tracker'):
        for period, name in zip(periods, period_names):
            pnl_data = trading_bot.balance_tracker.get_balance_for_period(period)
            
            if 'error' not in pnl_data:
                summary[name] = {
                    'pnl_absolute': pnl_data['pnl_absolute'],
                    'pnl_percentage': pnl_data['pnl_percentage'],
                    'start_balance': pnl_data['start_balance'],
                    'end_balance': pnl_data['end_balance']
                }
            else:
                summary[name] = {
                    'pnl_absolute': 0,
                    'pnl_percentage': 0,
                    'start_balance': 0,
                    'end_balance': 0
                }
    else:
        # Заполняем нулями если трекер недоступен
        for name in period_names:
            summary[name] = {
                'pnl_absolute': 0,
                'pnl_percentage': 0,
                'start_balance': 0,
                'end_balance': 0
            }
    
    return summary

@app.get("/api/balance/history/{period}")
def get_balance_history(period: str):
    """Получить историю баланса за период для графика"""
    global trading_bot  # ✅ ДОБАВЛЯЕМ global
    
    if trading_bot and hasattr(trading_bot, 'balance_tracker'):
        pnl_data = trading_bot.balance_tracker.get_balance_for_period(period)
        
        if 'error' not in pnl_data:
            return {
                'success': True,
                'data': pnl_data['chart_data']
            }
    
    return {
        'success': False,
        'error': 'No balance history available'
    }

# --- Запуск приложения ---
if __name__ == "__main__":
    # Проверяем наличие необходимых переменных окружения
    required_env = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"]
    missing_vars = [var for var in required_env if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        logging.error(f"Please set them in {config_path}")
        exit(1)

    # Инициализируем API клиент
    api_client = CapitalComAPI(
        api_key=os.getenv("CAPITAL_API_KEY"), 
        identifier=os.getenv("CAPITAL_IDENTIFIER"), 
        password=os.getenv("CAPITAL_PASSWORD")
    )
    
    if not api_client.login_and_get_tokens():
        logging.error("Could not log in to Capital.com API. Exiting.")
        exit(1)

    # Инициализируем новостной API клиент
    news_api_client = RealNewsAPIClient(
        news_api_key=os.getenv("NEWS_API_KEY"),
        marketaux_api_key=os.getenv("MARKETAUX_API_TOKEN")
    )
    
    # Инициализируем RL агента
    state_dim = LOOKBACK_WINDOW * len(FEATURE_COLUMNS) + 4
    rl_agent = PPOAgent(state_dim, action_dim=3)
    
    # Проверяем наличие файла модели
    if not ABSOLUTE_MODEL_PATH or not ABSOLUTE_MODEL_PATH.exists():
        logging.error(f"Could not find any model file with pattern '{MODEL_NAME_PATTERN}' in directory: {MODELS_DIR}")
        logging.info("Available files in models directory:")
        for f in MODELS_DIR.iterdir():
            logging.info(f"  - {f.name}")
        exit(1)
    
    # Загружаем модель
    rl_agent.load_model(str(ABSOLUTE_MODEL_PATH))
    
    # ✅ СОЗДАЕМ ГЛОБАЛЬНУЮ ПЕРЕМЕННУЮ trading_bot
    trading_bot = TradingBot(api_client, news_api_client, rl_agent)
    
    # Запускаем фоновый поток торговли
    background_thread = threading.Thread(
        target=trading_logic_thread, 
        args=(api_client, trading_bot), 
        daemon=True
    )
    background_thread.start()

    logging.info("Starting FastAPI server at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)