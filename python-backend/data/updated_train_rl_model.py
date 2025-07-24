#!/usr/bin/env python3
"""
ОБНОВЛЕННЫЙ скрипт для обучения RL агента под новый формат данных
Поддерживает Golden Cross, готовые технические индикаторы и multi-asset данные
"""
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle
from mathematical_sentiment_model import MathematicalSentimentAnalyzer, get_enhanced_sentiment
from typing import Dict, Tuple 
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ДОБАВИТЬ КЛАСС МЕТРИК
class PerformanceMetrics:
    """Расчет финансовых коэффициентов из таблицы целевых показателей"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def add_trade(self, entry_price: float, exit_price: float, position_size: float):
        """Добавляем сделку"""
        pnl = (exit_price - entry_price) * position_size
        self.trades.append({
            'pnl': pnl,
            'is_profitable': pnl > 0
        })
        
    def add_equity_point(self, equity: float):
        """Добавляем точку equity"""
        self.equity_curve.append(equity)
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            daily_return = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def calculate_sharpe_ratio(self) -> float:
        """Коэффициент Шарпа"""
        if not self.daily_returns:
            return 0.0
        excess_returns = np.array(self.daily_returns) - (self.risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self) -> float:
        """Максимальная просадка"""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        return -max_dd
    
    def calculate_win_rate(self) -> float:
        """Процент прибыльных сделок"""
        if not self.trades:
            return 0.0
        profitable = sum(1 for trade in self.trades if trade['is_profitable'])
        return profitable / len(self.trades)
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Основные метрики"""
        if not self.equity_curve:
            return {}
        
        total_return = (self.equity_curve[-1] / self.equity_curve[0]) - 1
        annual_return = total_return  # Упрощенно
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'total_trades': len(self.trades)
        }

# ДОБАВИТЬ ФУНКЦИЮ РАЗДЕЛЕНИЯ ДАННЫХ
def split_data_by_time(data: pd.DataFrame, train_ratio: float = 0.6, 
                      val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделение данных по времени"""
    data_sorted = data.sort_index()
    total_rows = len(data_sorted)
    
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    train_data = data_sorted.iloc[:train_end].copy()
    val_data = data_sorted.iloc[train_end:val_end].copy()
    test_data = data_sorted.iloc[val_end:].copy()
    
    logger.info(f"📊 Данные разделены: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

# --- Блок 1: ОБНОВЛЕННАЯ Среда для нового формата данных ---
"""
    Обновленная среда для торговли с поддержкой нового формата данных:
    1. GOLD_Close вместо Close
    2. Готовые технические индикаторы (не рассчитываем заново)
    3. Golden Cross сигналы
    4. Multi-asset данные
    5. Расширенный sentiment анализ
"""
class EnhancedGoldTradingEnvironment(gym.Env):
   
    def __init__(self, data, initial_balance=10000, commission=0.001, lookback_window=24):
        super().__init__()
        
        # Проверяем наличие необходимых колонок в новом формате
        required_columns = ['GOLD_Close', 'sentiment']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"❌ Отсутствуют обязательные колонки: {missing_columns}")
            
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window
        # Добавляем совместимость с window_size для новой функции
        self.window_size = lookback_window 
         # Алиас для совместимости
        
        
        # Определяем фичи из ГОТОВОГО датасета (не рассчитываем заново!)
        self.feature_columns = self._define_feature_columns()
        logger.info(f"📊 Используемые фичи ({len(self.feature_columns)}): {self.feature_columns}")
        
        # Проверяем доступность фичей
        self._validate_features()
        
        # Подготавливаем данные
        self._prepare_data()
        logger.info("🔧 Создание единственного экземпляра MathematicalSentimentAnalyzer для среды...")
        custom_weights = getattr(self, 'sentiment_weights', None)
        custom_reliabilities = getattr(self, 'sentiment_reliabilities', None)
        self.sentiment_analyzer = MathematicalSentimentAnalyzer(custom_weights, custom_reliabilities)

        
        # Gym spaces
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Размер observation space
        n_features = len(self.feature_columns)
        n_portfolio_state = 4  # баланс, позиция, размер, нереализованный PnL
        observation_space_size = n_features * lookback_window + n_portfolio_state
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32
        )
        
        logger.info(f"🎯 Environment инициализирован:")
        logger.info(f"   📊 Данных: {len(self.data)} записей")
        logger.info(f"   🧠 Observation space: {observation_space_size}")
        logger.info(f"   ⚡ Action space: {self.action_space.n}")

    def _define_feature_columns(self):
        """Определяем фичи из готового датасета"""
        # Базовые фичи, которые должны быть в датасете
        base_features = [
            'GOLD_Close',           # Основная цена
            'DXY_Close',            # Индекс доллара
            'VIX_Close',            # Индекс страха
            'sentiment'             # Настроения рынка
        ]
        
        # Технические индикаторы (уже рассчитанные)
        technical_features = [
            'STOCHk_14_3_3',        # Stochastic %K
            'CCI_14_0.015',         # Commodity Channel Index
            'Price_Change_1',       # Изменение цены за 1 час
            'Price_Change_5',       # Изменение цены за 5 часов
            'Volatility_24',        # Волатильность
            'DXY_change',           # Изменение доллара
            'VIX_change',           # Изменение VIX
        ]
        
        # Golden Cross фичи (НОВЫЕ!)
        golden_cross_features = [
            'golden_cross_trend',    # Направление тренда (-1, 0, 1)
            'ma_spread_normalized',  # Расстояние между MA
        ]
        
        # Дополнительные фичи (если есть)
        optional_features = [
            'RSI_20',               # RSI (если есть)
            'ATR_20',               # Average True Range (если есть)
            'MA_20',
                                      
        ]
        
        # Собираем все доступные фичи
        all_features = base_features + technical_features + golden_cross_features + optional_features
        
        # Фильтруем только те, что есть в датасете
        available_features = [col for col in all_features if col in self.data.columns]
        
        logger.info(f"✅ Найдено фичей: {len(available_features)} из {len(all_features)} возможных")
        return available_features

    def _validate_features(self):
        """Проверяем доступность всех фичей"""
        missing_features = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_features:
            logger.warning(f"⚠️ Отсутствующие фичи: {missing_features}")
            # Убираем отсутствующие фичи
            self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        
        logger.info(f"✅ Валидация прошла. Финальные фичи: {len(self.feature_columns)}")

    def _prepare_data(self):
        """Подготовка данных без пересчета индикаторов"""
        logger.info("🔧 Подготовка данных...")
        
        # Убираем NaN значения
        initial_length = len(self.data)
        self.data = self.data.dropna(subset=self.feature_columns)
        final_length = len(self.data)
        
        if final_length < initial_length:
            logger.warning(f"⚠️ Удалено {initial_length - final_length} строк с NaN")
        
        # Нормализация фичей (кроме цен и некоторых специальных)
        self._normalize_features()
        
        logger.info(f"✅ Данные подготовлены: {len(self.data)} записей готово к обучению")

    def _normalize_features(self):
        """Нормализация фичей для стабильного обучения"""
        logger.info("📊 Нормализация фичей...")
        
        # Фичи, которые НЕ нормализуем (цены и Golden Cross trend)
        skip_normalization = ['GOLD_Close', 'DXY_Close', 'VIX_Close', 'golden_cross_trend']
        
        for col in self.feature_columns:
            if col not in skip_normalization and col in self.data.columns:
                # Z-score нормализация
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                
                if std_val > 1e-9:  # Избегаем деления на 0
                    self.data[col] = (self.data[col] - mean_val) / std_val
                    logger.debug(f"   ✓ Нормализован {col}: mean={mean_val:.3f}, std={std_val:.3f}")
                else:
                    logger.warning(f"   ⚠️ Пропущена нормализация {col}: std слишком мал")
        
        # Специальная обработка для цен (логарифмическое масштабирование)
        for price_col in ['GOLD_Close', 'DXY_Close', 'VIX_Close']:
            if price_col in self.data.columns:
                # Нормализуем относительно первого значения
                first_value = self.data[price_col].iloc[0]
                self.data[f'{price_col}_normalized'] = self.data[price_col] / first_value
                
                # Добавляем в фичи (заменяем исходную цену)
                if price_col in self.feature_columns:
                    idx = self.feature_columns.index(price_col)
                    self.feature_columns[idx] = f'{price_col}_normalized'
        
        logger.info("✅ Нормализация завершена")

    def reset(self, seed=None):
        """Сброс среды в начальное состояние - КАК В ОРИГИНАЛЕ"""
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        self.metrics = PerformanceMetrics() 
        self.metrics.add_equity_point(self.initial_balance)
        
        return self._get_observation(), {}

    # Inside your GoldTradingEnv class in updated_train_rl_model.py

    def _get_observation(self):
        """
        Получает текущее наблюдение (состояние рынка + состояние портфеля).
        Робастная версия с защитой от ошибок и математическим sentiment.
        """
        # 1. ВСЕГДА определяем базовое состояние рынка в первую очередь
        start = max(0, self.current_step - self.lookback_window + 1)
        end = self.current_step + 1
        
        # Получаем срез данных
        market_data_slice = self.data.iloc[start:end][self.feature_columns]
        market_state = market_data_slice.values.flatten()
        
        # Если в начале эпизода данных меньше, чем размер окна, дополняем нулями
        expected_size = self.lookback_window * len(self.feature_columns)
        if len(market_state) < expected_size:
            padding = np.zeros(expected_size - len(market_state))
            market_state = np.concatenate([padding, market_state])
            logger.debug(f"🔧 Добавлен padding: {len(padding)} нулей")
        
        # 2. ТЕПЕРЬ пытаемся улучшить state с помощью математической модели
    
        try:    
            current_data_row = self.data.iloc[self.current_step]
            enhanced_sentiment, confidence = get_enhanced_sentiment(self.sentiment_analyzer, current_data_row)
            
            # Используем улучшенный sentiment, если модель уверена
            CONFIDENCE_THRESHOLD = 0.5 
            if confidence > CONFIDENCE_THRESHOLD and 'sentiment' in self.feature_columns:
                # Создаем копию, чтобы не изменять оригинальные данные
                enhanced_market_state = market_state.reshape(self.lookback_window, len(self.feature_columns)).copy()
                
                # Находим индекс колонки 'sentiment', чтобы заменить значение
                sentiment_column_index = self.feature_columns.index('sentiment')
                
                # Заменяем sentiment в последнем временном шаге окна
                enhanced_market_state[-1, sentiment_column_index] = enhanced_sentiment
                
                # Обновляем market_state, сглаживая его обратно в 1D массив
                market_state = enhanced_market_state.flatten()
                logger.debug(f"🧠 Sentiment улучшен до {enhanced_sentiment:.3f} с confidence {confidence:.2f}")
            else:
                logger.debug(f"⚠️ Sentiment не улучшен: confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                
        except Exception as e:
            # Если что-то пошло не так, используем базовый market_state
            logger.warning(f"⚠️ Не удалось применить математический sentiment: {e}. Используется оригинальный.")
        
        # 3. Определяем состояние портфеля (ПРАВИЛЬНАЯ ВЕРСИЯ)
        current_price = self.data['GOLD_Close'].iloc[self.current_step]
        unrealized_pnl = 0
        if self.position == 1 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Нормализованный баланс
            float(self.position),                 # Текущая позиция (0 или 1)
            self.position_size / max(1.0, self.balance / current_price) if current_price > 0 else 0,  # Размер позиции
            unrealized_pnl                       # Нереализованный P&L
        ], dtype=np.float32)
        
        # 4. Объединяем состояния. Эта строка теперь всегда будет работать.
        observation = np.concatenate([market_state, portfolio_state]).astype(np.float32)
        
        # 5. Проверяем на NaN/inf и исправляем
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            logger.warning("⚠️ NaN/inf в observation, заменяем на безопасные значения")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 6. Проверка на соответствие размерности
        expected_obs_size = self.observation_space.shape[0]
        if observation.shape[0] != expected_obs_size:
            logger.error(f"❌ Ошибка размерности observation! Ожидалось {expected_obs_size}, получено {observation.shape[0]}")
            # Попытка исправить размерность
            if observation.shape[0] < expected_obs_size:
                # Дополняем нулями
                padding = np.zeros(expected_obs_size - observation.shape[0])
                observation = np.concatenate([observation, padding])
            else:
                # Обрезаем
                observation = observation[:expected_obs_size]
            logger.warning(f"🔧 Размерность исправлена до {observation.shape[0]}")
            
        return observation.astype(np.float32)

    def step(self, action):
        """Выполняет шаг в среде - ОРИГИНАЛЬНАЯ ЛОГИКА + ЛОГИЧНЫЕ УЛУЧШЕНИЯ"""
        current_price = self.data['GOLD_Close'].iloc[self.current_step]
        reward = 0
        
        if action == 1: # Buy
            if self.position <= 0: 
                self.position = 1
                self.position_size = (self.balance * 0.95) / current_price if current_price > 0 else 0
                self.entry_price = current_price
                self.balance -= self.position_size * current_price * self.commission
                self.trade_count += 1
                reward -= 1  # ОРИГИНАЛЬНЫЙ штраф за открытие позиции
                
        elif action == 2: # Sell (в данном случае, только закрытие лонга)
            if self.position == 1:
                profit = (current_price - self.entry_price) * self.position_size
                self.balance += profit - (current_price * self.position_size * self.commission)
                reward += profit / self.initial_balance * 100  # ОРИГИНАЛЬНАЯ награда за прибыль
                self.metrics.add_trade(self.entry_price, current_price, self.position_size)
                self.position = 0
                self.position_size = 0

        # ОРИГИНАЛЬНАЯ ЛОГИКА: награда за unrealized profit пока позиция открыта
        if self.position != 0:
            if (self.current_step + 1) < len(self.data):
                next_price = self.data['GOLD_Close'].iloc[self.current_step + 1]
                price_diff = (next_price - current_price) * self.position
                reward += price_diff / current_price * 100  # ЭТО УЖЕ БЫЛО В ОРИГИНАЛЕ!

        # ✨ ЛОГИЧНОЕ ДОБАВЛЕНИЕ 1: Golden Cross смарт-награды
        if 'golden_cross_trend' in self.data.columns:
            gc_trend = self.data['golden_cross_trend'].iloc[self.current_step]
            
            # Умные награды за торговлю по долгосрочному тренду
            if action == 1 and gc_trend > 0:  # Buy в бычьем тренде
                reward += 2  # Небольшая награда
            elif action == 2 and gc_trend < 0:  # Sell в медвежьем тренде  
                reward += 2  # Небольшая награда
            elif action == 1 and gc_trend < 0:  # Buy против тренда
                reward -= 3  # Штраф за торговлю против тренда

        # ✨ ЛОГИЧНОЕ ДОБАВЛЕНИЕ 2: Штрафы за опасные просадки
        current_equity = self.balance
        if self.position == 1:
            current_equity += self.position_size * current_price
        
        equity_ratio = current_equity / self.initial_balance
        if equity_ratio < 0.7:  # Просадка больше 30%
            reward -= 10  # Штраф за риск
        elif equity_ratio < 0.5:  # Критическая просадка
            reward -= 25  # Большой штраф

        self.current_step += 1
        done = self.current_step >= len(self.data) - 2 or self.balance < self.initial_balance * 0.5  # ОРИГИНАЛЬНЫЕ условия
        
        # ДОБАВИТЬ ТРЕКИНГ EQUITY
        current_equity = self.balance
        if self.position == 1:
            current_equity += self.position_size * current_price
        self.metrics.add_equity_point(current_equity)

        return self._get_observation(), reward, done, False, {}

    def get_portfolio_stats(self):
        """Получение статистики портфеля - УПРОЩЕННАЯ ВЕРСИЯ"""
        current_price = self.data['GOLD_Close'].iloc[self.current_step - 1]
        current_equity = self.balance
        
        if self.position == 1:
            current_equity += self.position_size * current_price
        
        total_return = (current_equity - self.initial_balance) / self.initial_balance
        
        advanced_metrics = self.metrics.get_metrics_summary() if hasattr(self, 'metrics') else {}

        basic_stats = {
            'balance': self.balance,
            'position': self.position,
            'current_equity': current_equity,
            'total_return': total_return,
            'trade_count': self.trade_count
        }
        
        # Объединяем базовые и продвинутые метрики
        return {**basic_stats, **advanced_metrics}

# --- Блок 2: PPO Агент (БЕЗ ИЗМЕНЕНИЙ) ---
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

class Memory:
    def __init__(self): 
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
    
    def clear(self): 
        del self.actions[:]; del self.states[:]; del self.logprobs[:]; del self.rewards[:]; del self.is_terminals[:]

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, hidden_dim=256):
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)

        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        
        logger.info(f"🤖 PPO Agent инициализирован на {self.device}")
        
    def select_action(self, state, memory):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_probs, _ = self.policy_old(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            
        return action.item()

    def update(self, memory):
        # Вычисляем discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: 
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).to(self.device).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).to(self.device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).to(self.device).detach()

        # Оптимизация политики
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)

            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean().item()

# --- Блок 3: Главный цикл обучения ---
def main():
    logger.info("🚀 Запуск обучения RL Gold Trader с train/val/test split...")
    
    # Загружаем новый датасет
    try:
        # Пробуем разные возможные названия файлов
        possible_files = [
            'master_training_dataset.csv',
            'gold_data_with_sentiment_hourly.csv', 
            'automated_gold_data_with_sentiment.csv'
        ]
        
        data_df = None
        for filename in possible_files:
            try:
                data_df = pd.read_csv(filename, index_col=0, parse_dates=True)
                logger.info(f"✅ Загружен датасет: {filename}")
                break
            except FileNotFoundError:
                continue
        
        if data_df is None:
            raise FileNotFoundError("❌ Не найден ни один из ожидаемых файлов датасета")
            
        logger.info(f"📊 Датасет загружен: {data_df.shape}")
        logger.info(f"📋 Колонки: {list(data_df.columns)}")
        logger.info(f"📅 Временной диапазон: {data_df.index.min()} → {data_df.index.max()}")
        
        train_data, val_data, test_data = split_data_by_time(data_df, train_ratio=0.6, val_ratio=0.2)

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных: {e}")
        return

    # Создаем среду
    try:
        env = EnhancedGoldTradingEnvironment(train_data, lookback_window=24)  # ТОЛЬКО TRAIN!
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        logger.info(f"🎯 Обучение на {len(train_data)} записях (training set)")

        logger.info(f"🎯 Среда создана:")
        logger.info(f"   State dimension: {state_dim}")
        logger.info(f"   Action dimension: {action_dim}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания среды: {e}")
        return

    # Создаем агента
    agent = PPOAgent(state_dim, action_dim, lr=1e-4, gamma=0.99)  # Уменьшили learning rate
    memory = Memory()

    # Параметры обучения
    num_episodes = 200  # Уменьшили для начала
    update_timestep = 1000  # Обновляем чаще
    max_timesteps = 10000
    
    logger.info(f"🎓 Начинаем обучение:")
    logger.info(f"   Episodes: {num_episodes}")
    logger.info(f"   Update frequency: {update_timestep}")
    logger.info(f"   Max timesteps per episode: {max_timesteps}")
    
    # Основной цикл обучения
    timestep = 0
    best_reward = -float('inf')
    best_val_return = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        try:
            while episode_steps < max_timesteps:
                timestep += 1
                episode_steps += 1
                
                # Выбираем действие
                action = agent.select_action(state, memory)
                
                # Выполняем шаг
                state, reward, done, _, _ = env.step(action)
                
                # Сохраняем в память
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                episode_reward += reward
                
                # Обновляем политику
                if timestep % update_timestep == 0:
                    loss = agent.update(memory)
                    memory.clear()
                    logger.info(f"🔄 Policy updated at timestep {timestep}. Loss: {loss:.4f}")
                
                if done:
                    break
                    
        except Exception as e:
            logger.error(f"❌ Ошибка в эпизоде {episode}: {e}")
            break
        
        # Статистика эпизода
        stats = env.get_portfolio_stats()
        
        logger.info(f"Episode {episode:3d} | "
                   f"Reward: {episode_reward:8.2f} | "
                   f"Return: {stats['total_return']*100:6.2f}% | "
                   f"Trades: {stats['trade_count']:3d} | "
                   f"Sharpe: {stats.get('sharpe_ratio', 0):.2f} | "
                   f"MaxDD: {stats.get('max_drawdown', 0)*100:.1f}% | "
                   f"WinRate: {stats.get('win_rate', 0)*100:.1f}%")
        
        # Сохраняем лучшую модель
        if episode % 10 == 0:
            logger.info(f"🔍 Валидация после эпизода {episode}...")
            try:
                # Создаем среду для валидации
                val_env = EnhancedGoldTradingEnvironment(val_data, lookback_window=24)
                val_state, _ = val_env.reset()
                val_reward = 0
                val_steps = 0
                
                # Прогоняем валидацию без обучения
                while val_steps < len(val_data) - 24 - 1:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(val_state).to(agent.device).unsqueeze(0)
                        action_probs, _ = agent.policy_old(state_tensor)
                        action = torch.argmax(action_probs, dim=1).item()  # Greedy action
                    
                    val_state, reward, done, _, _ = val_env.step(action)
                    val_reward += reward
                    val_steps += 1
                    
                    if done:
                        break
                
                val_stats = val_env.get_portfolio_stats()
                val_return = val_stats['total_return']
                
                logger.info(f"📊 Validation: Return={val_return*100:.2f}%, "
                           f"Sharpe={val_stats.get('sharpe_ratio', 0):.2f}, "
                           f"Trades={val_stats['trade_count']}")
                
                # Сохраняем лучшую модель по validation
                if val_return > best_val_return:
                    best_val_return = val_return
                    best_model_path = f"best_val_model_{val_return:.3f}_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
                    torch.save(agent.policy_old.state_dict(), best_model_path)
                    logger.info(f"💾 Новая лучшая модель: {best_model_path}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Ошибка валидации: {e}")
        
        # Сохраняем лучшую модель по training reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            train_best_path = f"best_train_model_{episode_reward:.1f}_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            torch.save(agent.policy_old.state_dict(), train_best_path)

    # Финальное сохранение
    final_model_path = f"final_rl_gold_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(agent.policy_old.state_dict(), final_model_path)
    
    # ДОБАВИТЬ ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ
    logger.info("📝 Финальное тестирование на test set...")
    try:
        # Загружаем лучшую модель если есть
        if 'best_model_path' in locals():
            agent.policy_old.load_state_dict(torch.load(best_model_path))
            logger.info(f"✅ Загружена лучшая модель для тестирования")
        
        # Тестирование на test set
        test_env = EnhancedGoldTradingEnvironment(test_data, lookback_window=24)
        test_state, _ = test_env.reset()
        test_reward = 0
        test_steps = 0
        
        while test_steps < len(test_data) - 24 - 1:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(test_state).to(agent.device).unsqueeze(0)
                action_probs, _ = agent.policy_old(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
            
            test_state, reward, done, _, _ = test_env.step(action)
            test_reward += reward
            test_steps += 1
            
            if done:
                break
        
        test_stats = test_env.get_portfolio_stats()
        
        # ФИНАЛЬНЫЙ ОТЧЕТ
        print("\n" + "="*60)
        print("📊 ФИНАЛЬНЫЙ ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*60)
        print(f"📈 Test Return:        {test_stats['total_return']*100:6.2f}% | Цель: 15.00%")
        print(f"📊 Sharpe Ratio:       {test_stats.get('sharpe_ratio', 0):6.2f} | Цель:  2.50")
        print(f"📉 Max Drawdown:       {test_stats.get('max_drawdown', 0)*100:6.1f}% | Цель: -8.0%")
        print(f"🎯 Win Rate:           {test_stats.get('win_rate', 0)*100:6.1f}% | Цель: 65.0%")
        print(f"🔄 Total Trades:       {test_stats['trade_count']:6d}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Ошибка финального тестирования: {e}")
    
    logger.info("✅ Обучение завершено!")
    logger.info(f"💾 Финальная модель: {final_model_path}")
    logger.info(f"🏆 Лучшая validation модель: {locals().get('best_model_path', 'N/A')}")
    logger.info(f"🏆 Лучший validation return: {best_val_return:.2%}")

if __name__ == '__main__':
    main()