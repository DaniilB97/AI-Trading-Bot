# -*- coding: utf-8 -*-
# rl_live_dashboard_final.py
import os
import sys
import time
import json
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
from working_sentiment_api import get_market_sentiment

# --- UI & Threading Libraries ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItemCAPITAL_API_KEY=nTJnhoUFtjcTuN2J
CAPITAL_PASSWORD=zANra3.WW.7JuZ5
CAPITAL_IDENTIFIER=gervafrokit2112@gmail.com,
                             QPlainTextEdit, QGridLayout, QFrame)
from PyQt6.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt6.QtGui import QColor

# --- AI & Data Libraries ---
import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --- Helper Libraries ---
from dotenv import load_dotenv
# Это заполнитель для вашего реального API-клиента.
# Вы должны заменить его своей реализацией.
from capital_request import CapitalComAPI

# --- Configuration ---
load_dotenv()

# --- RL Model & Trading Configuration ---
TRADING_EPIC = "GOLD"
RESOLUTION = "HOUR"
UPDATE_INTERVAL_SECONDS = 60 * 60  # Проверка каждый час

# ВАЖНО: Убедитесь, что путь к модели указан верно
MODEL_PATH = "rl_gold_trader_model_20250624_1316.pth"
TRADE_SIZE = 1  # Например, 1 единица золота
LOOKBACK_WINDOW = 30

# --- ВАЖНО: Этот список признаков должен ТОЧНО соответствовать тем, на которых обучалась модель ---
# Ошибка `size mismatch` возникает, если этот набор признаков отличается от того,
# что был при обучении. Мы не можем использовать все новые индикаторы (MACD, BBands),
# так как модель не знает, что с ними делать. Но мы можем добавить 'sentiment'.
FEATURE_COLUMNS = ['RSI_14', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5', 'sentiment']


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Вспомогательные классы и функции-заполнители ---

class NewsAPIClient:
    """
    Заполнитель для реального клиента новостного API.
    Эта фиктивная реализация возвращает несколько примеров статей.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not api_key:
            logger.warning("Ключ API для новостей не установлен. Используются фиктивные данные.")

    def get_news(self, query: str, language: str = "en") -> List[Dict[str, str]]:
        """
        Получает новостные статьи. В реальной реализации здесь был бы HTTP-запрос.
        """
        logger.info(f"Получение фиктивных новостей для запроса: '{query}'")
        return [
            {"title": "Gold prices surge amid economic uncertainty and market volatility."},
            {"title": "Investors flock to gold as a safe-haven asset."},
            {"title": "Analysts predict a bullish trend for gold in the coming weeks."},
            {"title": "Gold slips slightly as dollar strengthens."},
        ]

def get_sentiment(text: str) -> float:
    """
    Очень простой заполнитель для анализа настроений.
    В реальной системе вы бы использовали библиотеку, такую как NLTK (VADER) или предобученную модель.
    Возвращает оценку от -1 (негативная) до 1 (позитивная).
    """
    text = text.lower()
    if any(word in text for word in ["surge", "bullish", "safe-haven", "strong", "rise"]):
        return 0.7
    elif any(word in text for word in ["slips", "falls", "weakens", "bearish", "uncertainty"]):
        return -0.5
    else:
        return 0.1
    
def get_market_status():
    """
    Checks the current open/closed status of major financial markets.
    Returns a dictionary with the status of each market.
    """
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).time()
    status = {}
    
    # London (approx 8am - 4:30pm UTC during standard time)
    # Note: These times do not account for daylight saving changes.
    is_london_open = now_utc.hour >= 8 and (now_utc.hour < 16 or (now_utc.hour == 16 and now_utc.minute < 30))
    status['London'] = "OPEN" if is_london_open else "CLOSED"
    
    # New York & Toronto (approx 2:30pm - 9pm UTC during standard time)
    is_ny_open = (now_utc.hour == 14 and now_utc.minute >= 30) or (now_utc.hour > 14 and now_utc.hour < 21)
    status['New York'] = "OPEN" if is_ny_open else "CLOSED"
    status['Toronto'] = status['New York'] # Same hours
    
    return status

# --- RL Агент и определение модели ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, state): return self.actor(state), self.critic(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        logger.info(f"✅ Модель RL успешно загружена из {path}")

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs, _ = self.policy(state_tensor.unsqueeze(0))
            action = torch.argmax(action_probs).item()
        return action

# --- Рабочий поток для трейдинга в реальном времени ---
class TradingWorker(QObject):
    log_message = pyqtSignal(str, str)

    def __init__(self, api: CapitalComAPI, news_api: NewsAPIClient, agent: PPOAgent):
        super().__init__()
        self.api = api
        self.agent = agent
        self.news_api = news_api
        self.scaler = MinMaxScaler()
        self.is_running = True
        self.is_scaler_fitted = False

    def run(self):
        while self.is_running:
            try:
                # Получаем данные счета и позиций
                details = self.api.get_account_details()
                positions = self.api.get_open_positions()
                if details is None: details = {}
                if positions is None: positions = []

                # <<< Вот правильное место для вызова функции статуса - ВНУТРИ цикла
                market_status = get_market_status()

                # Передаем ВСЕ данные (включая статус) через сигнал в основной поток
                self.data_updated.emit(details, positions, market_status)

            except Exception as e:
                logger.error(f"Ошибка в UIUpdateWorker: {e}")
            
            # Пауза перед следующей итерацией
            time.sleep(10)

    def fit_scaler_on_startup(self):
        """
        Подгоняет scaler на исторических данных для нормализации рыночных признаков.
        """
        self.log_message.emit("Инициализация... Получение данных для подгонки scaler.", "info")
        market_data = self.api.get_historical_prices(TRADING_EPIC, RESOLUTION, max_points=500)
        if not market_data or 'prices' not in market_data:
            self.log_message.emit("Не удалось получить начальные данные для scaler. Бот не будет торговать.", "error")
            return

        df = self.create_ohlc_df(market_data['prices'])
        df_with_indicators = self.calculate_indicators(df, sentiment_score=0.0)

        if len(df_with_indicators) > 0:
            # Подгоняем scaler ТОЛЬКО на колонках с рыночными признаками
            self.scaler.fit(df_with_indicators[FEATURE_COLUMNS].values)
            self.is_scaler_fitted = True
            self.log_message.emit("✅ Scaler данных успешно подогнан.", "info")
        else:
            self.log_message.emit("Недостаточно данных для подгонки scaler.", "error")

    # В НАЧАЛЕ ФАЙЛА (после других импортов):
from working_sentiment_api import get_market_sentiment

# В КЛАССЕ TradingWorker, МЕТОД run_trading_cycle():

class TradingWorker(QObject):
    # ... остальные методы ...
    
    def run_trading_cycle(self):
        """
        Выполняет один полный торговый цикл: получает данные, конструирует состояние,
        получает действие от модели и выполняет сделку.
        Этот метод включает вашу новую логику получения новостей.
        """
        if not self.is_scaler_fitted:
            self.log_message.emit("Scaler не подогнан. Остановка цикла.", "error")
            return

        open_positions = self.api.get_open_positions()
        if open_positions is None:
            self.log_message.emit("Не удалось получить открытые позиции от API.", "error")
            return
        position_for_epic = next((p for p in open_positions if p.get('market',{}).get('epic') == TRADING_EPIC), None)

        market_data = self.api.get_historical_prices(TRADING_EPIC, RESOLUTION, max_points=LOOKBACK_WINDOW + 50)
        if not market_data or 'prices' not in market_data:
            self.log_message.emit("Не удалось получить рыночные данные от API.", "error")
            return
        
        df = self.create_ohlc_df(market_data['prices'])

        # ====== ЗДЕСЬ ЗАМЕНИТЬ ЭТОТ БЛОК ======
        # СТАРОЕ (УБРАТЬ):
        # --- ВАША НОВАЯ СЕКЦИЯ: Получение и обработка настроений из новостей ---
        self.log_message.emit("📰 Получение настроений из новостей...", "info")
        try:
            articles = self.news_api.get_news(query="gold", language="en")
            live_sentiment_score = 0.0
            if articles:
                sentiments = [get_sentiment(article['title']) for article in articles if article.get('title')]
                if sentiments:
                    live_sentiment_score = sum(sentiments) / len(sentiments)
                    self.log_message.emit(f"Найдена оценка настроений: {live_sentiment_score:.2f}", "info")
            else:
                self.log_message.emit("Свежих новостей для анализа настроений не найдено.", "info")
        except Exception as e:
            self.log_message.emit(f"Ошибка при получении новостей: {e}", "error")
            live_sentiment_score = 0.0 # По умолчанию нейтральное значение при ошибке
        # --- Конец секции получения новостей ---
        
        # НОВОЕ (ДОБАВИТЬ):
        # --- РЕАЛЬНОЕ получение sentiment через MarketAux + Fear&Greed ---
        self.log_message.emit("📊 Получение РЕАЛЬНОГО sentiment...", "info")
        try:
            live_sentiment_score = get_market_sentiment("gold")
            self.log_message.emit(f"✅ Реальный sentiment: {live_sentiment_score:.3f}", "info")
        except Exception as e:
            self.log_message.emit(f"❌ Ошибка sentiment: {e}", "error")
            live_sentiment_score = 0.0
        # --- Конец секции реального sentiment ---
        # ====== КОНЕЦ ЗАМЕНЫ ======
        
        # Передаем оценку настроений в конструктор состояния
        state_vector = self.construct_state(df, position_for_epic, live_sentiment_score)
        if state_vector is None:
            self.log_message.emit("Не удалось сконструировать вектор состояния из данных.", "error")
            return
            
        action = self.agent.select_action(state_vector)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        self.log_message.emit(f"🤖 Решение модели: {action_map[action]}", "decision")

        # ... остальная логика торговли ...

        is_weekend = datetime.now().weekday() in [5, 6]
        if is_weekend and action in [1, 2]: # Block opening/closing on weekends
            self.log_message.emit("Weekend detected. No new trades will be executed.", "info")
            return # Exit the cycle early
        
        if action == 1 and position_for_epic is None:
            self.log_message.emit(f"Выполнение ордера на ПОКУПКУ для {TRADING_EPIC}...", "trade")
            result = self.api.create_position(epic=TRADING_EPIC, direction="BUY", size=TRADE_SIZE)
            if result and result.get('dealReference'):
                self.log_message.emit(f"✅ Запрос на создание позиции успешен. Ref: {result['dealReference']}", "trade")
            else:
                self.log_message.emit(f"❌ НЕ УДАЛОСЬ создать позицию. Ответ API: {result}", "error")

        elif action == 2 and position_for_epic is not None:
            deal_id = position_for_epic.get('position', {}).get('dealId')
            self.log_message.emit(f"Выполнение ордера на ЗАКРЫТИЕ для позиции {deal_id}...", "trade")
            result = self.api.close_position(deal_id)
            if result and result.get('dealReference'):
                self.log_message.emit(f"✅ Запрос на закрытие позиции успешен. Ref: {result['dealReference']}", "trade")
            else:
                self.log_message.emit(f"❌ НЕ УДАЛОСЬ закрыть позицию {deal_id}. Ответ API: {result}", "error")

    def create_ohlc_df(self, prices_list: list) -> pd.DataFrame:
        data = [{'Datetime': pd.to_datetime(p['snapshotTime']), 'Open': float(p['openPrice']['bid']), 'High': float(p['highPrice']['bid']), 'Low': float(p['lowPrice']['bid']), 'Close': float(p['closePrice']['bid']), 'Volume': 0} for p in prices_list]
        return pd.DataFrame(data).set_index('Datetime')

    def calculate_indicators(self, df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
        """
        Рассчитывает индикаторы, которые ожидает модель, и устанавливает оценку настроений.
        """
        df.ta.rsi(length=24, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)2
        df.ta.cci(length=14, append=True)
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['sentiment'] = sentiment_score
        df.bfill(inplace=True)
        df.dropna(inplace=True)
        return df

    def construct_state(self, df: pd.DataFrame, position: Optional[Dict], sentiment_score: float) -> Optional[np.ndarray]:
        """
        ИСПРАВЛЕНО: Эта функция конструирует вектор состояния с 154 признаками, чтобы соответствовать модели.
        Она отделяет рыночные признаки для масштабирования, а затем добавляет признаки портфеля.
        Эта версия отличается от той, что вы предоставили, чтобы избежать ошибки `size mismatch`.
        """
        df_with_indicators = self.calculate_indicators(df.copy(), sentiment_score=sentiment_score)
        if len(df_with_indicators) < LOOKBACK_WINDOW:
            return None

        # 1. Получаем часть состояния с рыночными признаками
        market_df = df_with_indicators[FEATURE_COLUMNS].tail(LOOKBACK_WINDOW)
        if len(market_df) < LOOKBACK_WINDOW:
            return None

        # 2. Масштабируем ТОЛЬКО рыночные признаки
        market_state_scaled = self.scaler.transform(market_df.values)
        market_state = market_state_scaled.flatten() # Это будет 30 * 5 = 150 признаков

        # 3. Получаем часть состояния с признаками портфеля
        balance, pnl, pos_state = 1.0, 0.0, 0
        if position:
            pos_state = 1 if position.get('position',{}).get('direction') == 'BUY' else -1
            pnl = position.get('position',{}).get('pnl', 0.0)

        # Состояние портфеля НЕ масштабируется.
        portfolio_state = np.array([balance, pos_state, 0, pnl]).flatten() # Это 4 признака

        # 4. Объединяем обе части для формирования конечного вектора состояния
        return np.concatenate([market_state, portfolio_state]).astype(np.float32) # 150 + 4 = 154 признака

    def stop(self):
        self.is_running = False

# --- UI Update Worker ---
class UIUpdateWorker(QObject):
    data_updated = pyqtSignal(dict, list, dict)

    def __init__(self, api: CapitalComAPI):
        super().__init__()
        self.api = api
        self.is_running = True

    def run(self):
       
        while self.is_running:
            try:
                details = self.api.get_account_details()
                positions = self.api.get_open_positions()
                if details is None: details = {}
                if positions is None: positions = []
                market_status = get_market_status()
                self.data_updated.emit(details, positions, market_status)
            except Exception as e:
                logger.error(f"Ошибка в UIUpdateWorker: {e}")
            time.sleep(10)

    def stop(self):
        self.is_running = False

# --- Главное окно приложения ---
class MainDashboard(QMainWindow):
    def __init__(self, api_client, news_api_client, agent):
        super().__init__()
        self.api = api_client
        self.news_api = news_api_client
        self.agent = agent
        self.setWindowTitle("Торговый бот RL с анализом настроений (Исправлено)")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QWidget { background-color: #111827; color: #e5e7eb; font-family: Inter, sans-serif; font-size: 14px; }
            QLabel#TitleLabel { font-size: 18px; font-weight: bold; color: white; }
            QLabel#StatLabel { font-size: 28px; font-weight: bold; color: white; }
            QPlainTextEdit, QListWidget { background-color: #1f2937; border-radius: 6px; border: 1px solid #374151;}
            QFrame#StatCard { background-color: rgba(31, 41, 55, 0.5); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.init_ui()
        self.start_trading_worker()
        self.start_ui_updater()

    def init_ui(self):
        self.account_balance_value_label = QLabel("Загрузка...")
        self.todays_pnl_value_label = QLabel("Загрузка...")
        balance_card = self.create_stat_card("Баланс счета", self.account_balance_value_label)
        pnl_card = self.create_stat_card("P&L за сегодня", self.todays_pnl_value_label)
        self.layout.addWidget(balance_card, 0, 0)
        self.layout.addWidget(pnl_card, 0, 1)
        self.market_status_layout = QHBoxLayout()
        self.london_status_label = QLabel("London: ...")
        self.ny_status_label = QLabel("New York: ...")
        self.toronto_status_label = QLabel("Toronto: ...")
        self.market_status_layout.addWidget(self.london_status_label)
        self.market_status_layout.addWidget(self.ny_status_label)
        self.market_status_layout.addWidget(self.toronto_status_label)
        # Add this new layout to your main grid layout, e.g., below the P&L card
        self.layout.addLayout(self.market_status_layout, 0, 2)

        log_title = QLabel("Журнал торгов"); log_title.setObjectName("TitleLabel")
        self.log_area = QPlainTextEdit(); self.log_area.setReadOnly(True)
        positions_title = QLabel("Открытые позиции"); positions_title.setObjectName("TitleLabel")
        self.positions_list = QListWidget()
        self.layout.addWidget(log_title, 1, 0)
        self.layout.addWidget(self.log_area, 2, 0)
        self.layout.addWidget(positions_title, 1, 1)
        self.layout.addWidget(self.positions_list, 2, 1)

    def create_stat_card(self, title: str, value_label: QLabel) -> QFrame:
        card = QFrame(); card.setObjectName("StatCard")
        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        value_label.setObjectName("StatLabel")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card

    def start_trading_worker(self):
        self.worker_thread = QThread()
        self.worker = TradingWorker(self.api, self.news_api, self.agent)
        self.worker.moveToThread(self.worker_thread)
        #self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.add_log)
        self.worker_thread.start()

    def start_ui_updater(self):
        self.ui_worker_thread = QThread()
        self.ui_worker = UIUpdateWorker(self.api)
        self.ui_worker.moveToThread(self.ui_worker_thread)
        self.ui_worker_thread.started.connect(self.ui_worker.run)
        self.ui_worker.data_updated.connect(self.on_data_received)
        self.ui_worker_thread.start()

    def on_data_received(self, details: dict, positions: list, market_status: dict):
        if details and 'accounts' in details:
            acc = details['accounts'][0]
            balance_info = acc.get('balance', {})
            balance_val = balance_info.get('balance', 0.0)
            pnl_val = balance_info.get('pnl', 0.0)
            self.account_balance_value_label.setText(f"${balance_val:,.2f}")
            self.todays_pnl_value_label.setText(f"${pnl_val:,.2f}")

        self.positions_list.clear()
        if positions:
            for pos in positions:
                market_info = pos.get('market', {})
                position_info = pos.get('position', {})
                instrument_name = market_info.get('instrumentName', 'Неизвестный инструмент')
                trade_size = position_info.get('size', 0)
                open_level = position_info.get('level', 0.0)
                pnl = position_info.get('pnl', 0.0)
                item_text = f"{instrument_name}: {trade_size} @ ${open_level:.2f} | P&L: ${pnl:.2f}"
                list_item = QListWidgetItem(item_text)
                list_item.setForeground(QColor("lightgreen") if pnl >= 0 else QColor("lightcoral"))
                self.positions_list.addItem(list_item)
        else:
            if self.positions_list.count() == 0:
                self.positions_list.addItem("Нет открытых позиций.")

        self.london_status_label.setText(f"London: {market_status['London']}")
        self.ny_status_label.setText(f"New York: {market_status['New York']}")
        self.toronto_status_label.setText(f"Toronto: {market_status['Toronto']}")

        london_color = "lightgreen" if market_status['London'] == "OPEN" else "lightcoral"
        ny_color = "lightgreen" if market_status['New York'] == "OPEN" else "lightcoral"
        self.london_status_label.setStyleSheet(f"color: {london_color};")
        self.ny_status_label.setStyleSheet(f"color: {ny_color};")
        self.toronto_status_label.setStyleSheet(f"color: {ny_color};")

    def add_log(self, message, style):
        color_map = {"info": "#9ca3af", "decision": "#facc15", "trade": "#34d399", "error": "#f87171"}
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.appendHtml(f"<span style='color: #6b7280;'>{timestamp}</span>: <span style='color: {color_map.get(style, '#e5e7eb')};'>{message}</span>")

    def closeEvent(self, event):
        self.worker.stop()
        self.ui_worker.stop()
        self.worker_thread.quit()
        self.ui_worker_thread.quit()
        self.worker_thread.wait()
        self.ui_worker_thread.wait()
        event.accept()



# --- Точка входа в приложение ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Обученная модель не найдена. Проверенный путь: {MODEL_PATH}")
        sys.exit(1)

    # --- Инициализация API-клиентов ---
    api = CapitalComAPI(os.getenv("CAPITAL_API_KEY"), os.getenv("CAPITAL_IDENTIFIER"), os.getenv("CAPITAL_PASSWORD"))
    if not api.login_and_get_tokens():
        sys.exit(1)

    news_api = NewsAPIClient(api_key=os.getenv("NEWS_API_KEY", ""))

    # --- Инициализация агента RL ---
    # ИСПРАВЛЕНО: Размерность состояния теперь правильно рассчитана и соответствует модели (154)
    state_dim = LOOKBACK_WINDOW * len(FEATURE_COLUMNS) + 4  # (30 * 5) + 4 = 154
    agent = PPOAgent(state_dim, action_dim=3)  # 3 действия: HOLD, BUY, SELL
    agent.load_model(MODEL_PATH)

    # --- Запуск приложения ---
    app = QApplication(sys.argv)
    window = MainDashboard(api, news_api, agent)
    window.show()
    sys.exit(app.exec())
