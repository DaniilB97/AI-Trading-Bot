#!/usr/bin/env python3
"""
ИНТЕГРИРОВАННЫЙ скрипт для обучения RL агента с улучшенной reward функцией
Включает математический sentiment анализ, Sharpe ratio оптимизацию и расширенные награды
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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Блок 1: ИНТЕГРИРОВАННАЯ Среда с улучшенной reward функцией ---
class EnhancedGoldTradingEnvironment(gym.Env):
    """
    Интегрированная среда для торговли с поддержкой:
    1. Нового формата данных (GOLD_Close, готовые индикаторы)
    2. Математического sentiment анализа
    3. Sharpe ratio оптимизации
    4. Расширенной reward функции
    5. Golden Cross сигналов
    """
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
        self.window_size = lookback_window  # Алиас для совместимости
        
        # 🔥 ДОБАВЛЯЕМ для Sharpe оптимизации
        self.returns_history = []
        self.trades_history = []
        self.portfolio_values = []
        self.risk_free_rate = 0.02 / (365 * 24)  # Часовая безрисковая ставка
        
        # Определяем фичи из ГОТОВОГО датасета (не рассчитываем заново!)
        self.feature_columns = self._define_feature_columns()
        logger.info(f"📊 Используемые фичи ({len(self.feature_columns)}): {self.feature_columns}")
        
        # Проверяем доступность фичей
        self._validate_features()
        
        # Подготавливаем данные
        self._prepare_data()
        
        # СОЗДАЕМ МАТЕМАТИЧЕСКИЙ АНАЛИЗАТОР SENTIMENT ОДИН РАЗ!
        try:
            from mathematical_sentiment_model import MathematicalSentimentAnalyzer
            self.sentiment_analyzer = MathematicalSentimentAnalyzer()
            logger.info("🧮 Математический анализатор sentiment создан")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось создать анализатор sentiment: {e}")
            self.sentiment_analyzer = None
        
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
            'RSI_14',               # RSI (если есть)
            'ATR_14',               # Average True Range (если есть)
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
        """Сброс среды в начальное состояние"""
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        
        # Сброс истории для Sharpe ratio
        self.returns_history = []
        self.trades_history = []
        self.portfolio_values = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Получает текущее наблюдение с математическим sentiment анализом
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
            # Используем УЖЕ СОЗДАННЫЙ анализатор (не создаем новый!)
            if self.sentiment_analyzer is not None:
                from mathematical_sentiment_model import get_enhanced_sentiment
                
                current_data_row = self.data.iloc[self.current_step]
                enhanced_sentiment, confidence = get_enhanced_sentiment(self.sentiment_analyzer, current_data_row)
                
                # Используем улучшенный sentiment, если модель уверена
                CONFIDENCE_THRESHOLD = 0.6
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
            else:
                logger.debug("⚠️ Анализатор sentiment недоступен, используем оригинальные данные")
                
        except Exception as e:
            # Если что-то пошло не так, используем базовый market_state
            logger.warning(f"⚠️ Не удалось применить математический sentiment: {e}. Используется оригинальный.")
        
        # 3. Определяем состояние портфеля
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
        
        # 4. Объединяем состояния
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
        """Улучшенный step с математическим sentiment анализатором и оптимизацией Sharpe ratio"""
        current_price = self.data['GOLD_Close'].iloc[self.current_step]
        
        # Запоминаем предыдущий portfolio value
        prev_portfolio_value = self.balance
        if self.position == 1:
            prev_portfolio_value += self.position_size * current_price
            
        # === ОРИГИНАЛЬНАЯ ТОРГОВАЯ ЛОГИКА ===
        base_reward = 0
        
        if action == 1: # Buy
            if self.position <= 0: 
                self.position = 1
                self.position_size = (self.balance * 0.30) / current_price if current_price > 0 else 0
                self.entry_price = current_price
                self.balance -= self.position_size * current_price * self.commission
                self.trade_count += 1
                base_reward -= 1  # Штраф за открытие позиции
                
        elif action == 2: # Sell
            if self.position == 1:
                profit = (current_price - self.entry_price) * self.position_size
                self.balance += profit - (current_price * self.position_size * self.commission)
                
                # Отмечаем сделку как прибыльную/убыточную
                is_profitable = profit > 0
                self.trades_history.append(is_profitable)
                if len(self.trades_history) > 20:  # Храним последние 20 сделок
                    self.trades_history.pop(0)
                
                base_reward += profit / self.initial_balance * 100
                self.position = 0
                self.position_size = 0

        # Unrealized profit пока позиция открыта
        if self.position != 0:
            if (self.current_step + 1) < len(self.data):
                next_price = self.data['GOLD_Close'].iloc[self.current_step + 1]
                price_diff = (next_price - current_price) * self.position
                base_reward += price_diff / current_price * 100

        # === НОВАЯ SHARPE-ОПТИМИЗИРОВАННАЯ REWARD ===
        
        # 1. Рассчитываем текущий portfolio value
        current_portfolio_value = self.balance
        if self.position == 1:
            current_portfolio_value += self.position_size * current_price
            
        self.portfolio_values.append(current_portfolio_value)
        
        # 2. Рассчитываем return за период
        period_return = 0
        if len(self.portfolio_values) >= 2:
            period_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns_history.append(period_return)
            
            # Ограничиваем историю
            if len(self.returns_history) > 168:  # 1 неделя часовых данных
                self.returns_history.pop(0)
        
        # 3. Рассчитываем Sharpe ratio
        sharpe_reward = 0
        current_sharpe = 0
        if len(self.returns_history) >= 24:  # Минимум 24 часа данных
            returns_array = np.array(self.returns_history)
            excess_returns = returns_array - self.risk_free_rate
            
            if np.std(excess_returns) > 0:
                current_sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(24 * 365)
                sharpe_reward = current_sharpe * 12  # Увеличили с 10 до 12
            
        # 4. 🧮 МАТЕМАТИЧЕСКИЙ SENTIMENT АНАЛИЗ
        mathematical_sentiment_reward = 0
        enhanced_sentiment = 0
        sentiment_confidence = 0
        
        try:
            if self.sentiment_analyzer is not None:
                from mathematical_sentiment_model import get_enhanced_sentiment
                
                current_data_row = self.data.iloc[self.current_step]
                enhanced_sentiment, sentiment_confidence = get_enhanced_sentiment(
                    self.sentiment_analyzer, current_data_row
                )
                
                # Награды на основе математического sentiment
                if action == 1 and enhanced_sentiment > 0.2:  # Buy при сильном позитивном
                    mathematical_sentiment_reward += enhanced_sentiment * sentiment_confidence * 8
                elif action == 2 and enhanced_sentiment < -0.2:  # Sell при сильном негативном
                    mathematical_sentiment_reward += abs(enhanced_sentiment) * sentiment_confidence * 8
                elif action == 0 and abs(enhanced_sentiment) < 0.15:  # Hold при неопределенности
                    mathematical_sentiment_reward += sentiment_confidence * 2
                elif action == 1 and enhanced_sentiment < -0.3:  # Buy против сильного негатива
                    mathematical_sentiment_reward -= abs(enhanced_sentiment) * sentiment_confidence * 10
                
        except Exception as e:
            logger.debug(f"⚠️ Ошибка математического sentiment: {e}")
            # Fallback к базовому sentiment
            if 'sentiment' in self.data.columns:
                basic_sentiment = self.data['sentiment'].iloc[self.current_step]
                enhanced_sentiment = basic_sentiment
                sentiment_confidence = 0.5
                
                if action == 1 and basic_sentiment > 0.15:
                    mathematical_sentiment_reward += basic_sentiment * 4
                elif action == 2 and basic_sentiment < -0.15:
                    mathematical_sentiment_reward += abs(basic_sentiment) * 4
        
        # 5. Golden Cross умная награда (усиленная)
        golden_cross_reward = 0
        if 'golden_cross_trend' in self.data.columns:
            gc_trend = self.data['golden_cross_trend'].iloc[self.current_step]
            
            # Учитываем также ma_spread для силы сигнала
            ma_spread = 0
            if 'ma_spread_normalized' in self.data.columns:
                ma_spread = abs(self.data['ma_spread_normalized'].iloc[self.current_step])
            
            spread_multiplier = 1 + (ma_spread * 2)  # Усиливаем при сильном спреде
            
            if action == 1 and gc_trend > 0:  # Buy в бычьем тренде
                golden_cross_reward += 4 * spread_multiplier
            elif action == 2 and gc_trend < 0:  # Sell в медвежьем тренде  
                golden_cross_reward += 4 * spread_multiplier
            elif action == 1 and gc_trend < 0:  # Buy против тренда
                golden_cross_reward -= 6 * spread_multiplier
            elif action == 0 and gc_trend == 0:  # Hold в неопределенности
                golden_cross_reward += 1
        
        # 6. Win rate бонус (улучшенный)
        
        
        # 7. Penalty за высокую волатильность (ужесточенный)
        volatility_penalty = 1
        if len(self.returns_history) >= 24:
            hourly_vol = np.std(self.returns_history)
            annual_vol = hourly_vol * np.sqrt(24 * 365)
            if annual_vol > 0.6:  # Годовая волатильность > 60%
                volatility_penalty = -(annual_vol - 0.6) * 20
        
        # 8. Penalty за критические просадки (ужесточенный)
        drawdown_penalty = 0
        current_equity = current_portfolio_value
        equity_ratio = current_equity / self.initial_balance
        
        if equity_ratio < 0.8:  # Просадка > 20%
            drawdown_penalty = -8
        elif equity_ratio < 0.7:  # Просадка > 30%
            drawdown_penalty = -20
        elif equity_ratio < 0.6:  # Критическая просадка > 40%
            drawdown_penalty = -40
        
        # 9. 🔥 НОВЫЙ: Momentum consistency reward
        momentum_consistency_reward = 0
        if len(self.returns_history) >= 5:
            recent_returns = self.returns_history[-5:]
            if action == 1 and all(r > 0 for r in recent_returns[-3:]):  # Buy на восходящем momentum
                momentum_consistency_reward += 3
            elif action == 2 and all(r < 0 for r in recent_returns[-3:]):  # Sell на нисходящем momentum
                momentum_consistency_reward += 3
        
        # === ИТОГОВАЯ НАГРАДА С НОВЫМИ ВЕСАМИ ===
        total_reward = (base_reward * 0.25 +                    # Базовая награда
                       sharpe_reward * 0.35 +                   # Основной вес на Sharpe
                       mathematical_sentiment_reward * 0.15 +   # 🧮 Математический sentiment
                       golden_cross_reward * 0.10 +             # Golden Cross
                            # Win rate бонус
                       volatility_penalty * 0.04 +              # Волатильность штраф
                       drawdown_penalty * 0.04 +                # Просадка штраф
                       momentum_consistency_reward * 0.02)      # 🔥 Momentum consistency
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 2 or equity_ratio < 0.4  # Стоп при 60% просадке
        
        return self._get_observation(), total_reward, done, False, {
            'sharpe_component': sharpe_reward,
            'mathematical_sentiment_component': mathematical_sentiment_reward,
            'enhanced_sentiment': enhanced_sentiment,
            'sentiment_confidence': sentiment_confidence,
            'golden_cross_component': golden_cross_reward,  
            'base_component': base_reward,
            'current_sharpe': current_sharpe,
            'momentum_consistency': momentum_consistency_reward
        }

    def get_portfolio_stats(self):
        """Расширенная статистика с Sharpe ratio"""
        current_price = self.data['GOLD_Close'].iloc[self.current_step - 1]
        current_equity = self.balance
        
        if self.position == 1:
            current_equity += self.position_size * current_price
        
        total_return = (current_equity - self.initial_balance) / self.initial_balance
        
        # Рассчитываем Sharpe ratio
        sharpe_ratio = 0
        if len(self.returns_history) >= 10:
            returns_array = np.array(self.returns_history)
            excess_returns = returns_array - self.risk_free_rate
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(24 * 365)
        
        # Win rate
        win_rate = sum(self.trades_history) / len(self.trades_history) if self.trades_history else 0
        
        return {
            'balance': self.balance,
            'position': self.position,
            'current_equity': current_equity,
            'total_return': total_return,
            'trade_count': self.trade_count,
            'sharpe_ratio': sharpe_ratio,  # 🔥 НОВОЕ
            'win_rate': win_rate,          # 🔥 НОВОЕ
            'max_drawdown': 1 - (current_equity / max(self.portfolio_values)) if self.portfolio_values else 0  # 🔥 НОВОЕ
        }

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
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        
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
    logger.info("🚀 Запуск обучения RL Gold Trader с улучшенной reward функцией...")
    
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
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных: {e}")
        return

    # Создаем среду
    try:
        env = EnhancedGoldTradingEnvironment(data_df, lookback_window=24)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
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
    best_sharpe = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Компоненты reward для анализа
        total_sharpe_component = 0
        total_sentiment_component = 0
        total_golden_cross_component = 0
        total_base_component = 0
        
        try:
            while episode_steps < max_timesteps:
                timestep += 1
                episode_steps += 1
                
                # Выбираем действие
                action = agent.select_action(state, memory)
                
                # Выполняем шаг
                state, reward, done, _, info = env.step(action)
                
                # Сохраняем в память
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                episode_reward += reward
                
                # Собираем статистику по компонентам reward
                if info:
                    total_sharpe_component += info.get('sharpe_component', 0)
                    total_sentiment_component += info.get('mathematical_sentiment_component', 0)
                    total_golden_cross_component += info.get('golden_cross_component', 0)
                    total_base_component += info.get('base_component', 0)
                
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
        
        # Расширенный лог с компонентами reward
        logger.info(f"Episode {episode:3d} | "
                   f"Reward: {episode_reward:8.2f} | "
                   f"Return: {stats['total_return']*100:6.2f}% | "
                   f"Sharpe: {stats['sharpe_ratio']:6.3f} | "
                   f"WinRate: {stats['win_rate']*100:5.1f}% | "
                   f"Trades: {stats['trade_count']:3d} | "
                   f"Steps: {episode_steps:4d}")
        
        # Детальная статистика reward компонентов каждые 10 эпизодов
        if episode % 10 == 0:
            logger.info(f"📊 Reward Components Analysis (Episode {episode}):")
            logger.info(f"   🎯 Base: {total_base_component:8.2f}")
            logger.info(f"   📈 Sharpe: {total_sharpe_component:8.2f}")
            logger.info(f"   🧠 Sentiment: {total_sentiment_component:8.2f}")
            logger.info(f"   🌟 Golden Cross: {total_golden_cross_component:8.2f}")
            logger.info(f"   💰 Max Drawdown: {stats['max_drawdown']*100:5.1f}%")
        
        # Сохраняем лучшую модель по общей награде
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = f"best_rl_gold_model_reward_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            torch.save(agent.policy_old.state_dict(), best_model_path)
            logger.info(f"💾 Новая лучшая модель по reward сохранена: {best_model_path}")
        
        # Сохраняем лучшую модель по Sharpe ratio
        if stats['sharpe_ratio'] > best_sharpe and stats['trade_count'] > 5:
            best_sharpe = stats['sharpe_ratio']
            best_sharpe_model_path = f"best_rl_gold_model_sharpe_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            torch.save(agent.policy_old.state_dict(), best_sharpe_model_path)
            logger.info(f"📈 Новая лучшая модель по Sharpe сохранена: {best_sharpe_model_path}")

    # Финальное сохранение
    final_model_path = f"final_rl_gold_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(agent.policy_old.state_dict(), final_model_path)
    
    logger.info("✅ Обучение завершено!")
    logger.info(f"💾 Финальная модель: {final_model_path}")
    logger.info(f"🏆 Лучшая награда: {best_reward:.2f}")
    logger.info(f"📈 Лучший Sharpe ratio: {best_sharpe:.3f}")

if __name__ == '__main__':
    main()