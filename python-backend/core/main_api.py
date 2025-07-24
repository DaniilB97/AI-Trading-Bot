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
# Теперь импорты стали более явными и надежными
from core.capital_request import CapitalComAPI
from core.live_rl_trading import PPOAgent, TradingWorker, NewsAPIClient, LOOKBACK_WINDOW, FEATURE_COLUMNS, TRADING_EPIC, RESOLUTION, TRADE_SIZE

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
def trading_logic_thread(api: CapitalComAPI, worker: TradingWorker):
    logging.info("🚀 Background trading logic started.")
    
    # --- ИСПРАВЛЕНИЕ ОШИБКИ 'emit' ---
    # Класс-заглушка, который имитирует поведение PyQt-сигнала
    class DummySignalEmitter:
        def __init__(self, callback_func):
            self.callback = callback_func
        
        def emit(self, message, style):
            # Вместо "излучения" сигнала, мы просто вызываем нашу функцию
            self.callback(message, style)

    # Эта функция будет вызываться нашей заглушкой для обновления логов
    def log_to_state(message, style):
        with state_lock:
            log_entry = { "id": f"log_{time.time()}", "time": datetime.now().strftime("%H:%M:%S"), "message": message, "type": style }
            app_state["logs"].insert(0, log_entry)
            app_state["logs"] = app_state["logs"][:100]

    # Заменяем PyQt-сигнал в экземпляре worker на нашу заглушку
    worker.log_message = DummySignalEmitter(log_to_state)
    
    worker.fit_scaler_on_startup()

    while True:
        try:
            logging.info("Starting new trading cycle...")
            worker.run_trading_cycle()
            logging.info("Updating global state...")
            details = api.get_account_details()
            positions = api.get_open_positions()
            market_data = api.get_historical_prices(TRADING_EPIC, RESOLUTION, max_points=100)
            
            with state_lock:
                if details and 'accounts' in details:
                    account_balance = details['accounts'][0].get('balance', {})
                    app_state["portfolio_metrics"]["equity"] = account_balance.get('deposit', 0.0) + account_balance.get('pnl', 0.0)
                    app_state["portfolio_metrics"]["cash"] = account_balance.get('available', 0.0)
                
                if market_data and 'prices' in market_data:
                    prices = market_data['prices']
                    app_state["chart_data"]["labels"] = [p['snapshotTime'] for p in prices]
                    app_state["chart_data"]["prices"] = [float(p['closePrice']['bid']) for p in prices]
                
                formatted_positions = []
                if positions:
                    for pos in positions:
                        position_info = pos.get('position', {})
                        market_info = pos.get('market', {})
                        current_price_data = api.get_historical_prices(market_info.get('epic'), "MINUTE", max_points=1)
                        current_price = float(current_price_data['prices'][0]['closePrice']['bid']) if current_price_data and current_price_data.get('prices') else position_info.get('level', 0.0)
                        
                        formatted_positions.append({
                            "ticker": market_info.get('instrumentName', 'N/A'),
                            "size": position_info.get('size', 0),
                            "entry_price": position_info.get('level', 0.0),
                            "current_price": current_price,
                            "pnl": position_info.get('pnl', 0.0)
                        })
                app_state["open_positions"] = formatted_positions

            # Используем атрибут из воркера, если он есть, иначе - значение по умолчанию
            sleep_duration = getattr(worker, 'UPDATE_INTERVAL_SECONDS', 3600)
            logging.info(f"State updated. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)

        except Exception as e:
            logging.error(f"Error in trading_logic_thread: {e}", exc_info=True)
            time.sleep(60)

# --- Эндпоинты API ---
@app.get("/api/state", response_model=AppStateResponse)
def get_current_state():
    with state_lock:
        return app_state

# --- Запуск приложения ---
if __name__ == "__main__":
    required_env = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"]
    if any(not os.getenv(key) for key in required_env):
        logging.error(f"Please set required environment variables in {config_path}")
        exit(1)

    api_client = CapitalComAPI(os.getenv("CAPITAL_API_KEY"), os.getenv("CAPITAL_IDENTIFIER"), os.getenv("CAPITAL_PASSWORD"))
    if not api_client.login_and_get_tokens():
        logging.error("Could not log in to Capital.com API. Exiting.")
        exit(1)

    news_api_client = NewsAPIClient(api_key=os.getenv("NEWS_API_KEY", ""))
    
    state_dim = LOOKBACK_WINDOW * len(FEATURE_COLUMNS) + 4
    rl_agent = PPOAgent(state_dim, action_dim=3)
    
    # Проверяем, был ли найден файл модели
    if not ABSOLUTE_MODEL_PATH or not ABSOLUTE_MODEL_PATH.exists():
        logging.error(f"Could not find any model file with pattern '{MODEL_NAME_PATTERN}' in directory: {MODELS_DIR}")
        exit(1)
        
    # Используем найденный абсолютный путь
    rl_agent.load_model(ABSOLUTE_MODEL_PATH)
    
    trading_worker = TradingWorker(api_client, news_api_client, rl_agent)
    
    background_thread = threading.Thread(target=trading_logic_thread, args=(api_client, trading_worker), daemon=True)
    background_thread.start()

    logging.info("Starting FastAPI server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8080)
