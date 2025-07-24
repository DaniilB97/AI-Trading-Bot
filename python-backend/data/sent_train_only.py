#!/usr/bin/env python3
"""
SENTIMENT-ONLY RL Trading Agent
Торгует ТОЛЬКО на основе sentiment анализа без других фичей
Цель: изолировать влияние sentiment на торговые решения
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
from mathematical_sentiment_model import MathematicalSentimentAnalyzer, get_enhanced_sentiment
from typing import Dict, Tuple

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentOnlyTradingEnvironment(gym.Env):
    """
    Упрощенная среда для торговли ТОЛЬКО на sentiment
    Фичи: только sentiment + базовое состояние портфеля
    """
    
    def __init__(self, data, initial_balance=10000, commission=0.001, lookback_window=12):
        super().__init__()
        
        # Проверяем обязательные колонки
        required_columns = ['GOLD_Close', 'sentiment']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"❌ Отсутствуют обязательные колонки: {missing_columns}")
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window
        
        # 🔥 НОВЫЕ ПАРАМЕТРЫ ДЛЯ SL И ДИНАМИЧЕСКИХ ПОЗИЦИЙ
        self.stop_loss_pct = 0.04  # 2% Stop Loss
        self.max_position_size = 0.6  # Максимум 90% от баланса
        self.min_position_size = 0.1  # Минимум 10% от баланса
        
        # ТОЛЬКО sentiment фичи!
        self.feature_columns = ['sentiment']
        
        logger.info(f"📊 Sentiment-Only фичи: {self.feature_columns}")
        
        # Создаем sentiment анализатор с оптимизированными параметрами
        self.sentiment_weights = {
            'marketaux_news': 0.45,      # MarketAux
            'fear_greed_index': 0.25,    # Fear & Greed  
            'golden_cross_trend': 0.13,  # Golden Cross
           # 'vix_momentum': 0.07,        # VIX
            #'price_momentum': 0.10       # Price momentum
        }
        
        self.sentiment_reliabilities = {
            'marketaux_news': 0.83,
            'fear_greed_index': 0.86,
            'golden_cross_trend': 0.54,
            #'vix_momentum': 0.69,
            #'price_momentum': 0.51
        }
        
        self.sentiment_analyzer = MathematicalSentimentAnalyzer(
            custom_weights=self.sentiment_weights,
            custom_reliabilities=self.sentiment_reliabilities
        )
        
        logger.info("🧠 Sentiment анализатор создан с оптимизированными параметрами")
        
        # Подготовка данных
        self._prepare_data()
        
        # Gym spaces
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Observation: sentiment lookback + портфель (5 значений)
        observation_space_size = lookback_window + 5  # sentiment history + portfolio state + SL info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32
        )
        
        logger.info(f"🎯 Sentiment-Only Environment:")
        logger.info(f"   📊 Данных: {len(self.data)} записей")
        logger.info(f"   🧠 Observation space: {observation_space_size}")
        logger.info(f"   📈 Только sentiment + портфель")

    def _prepare_data(self):
        """Простая подготовка данных"""
        logger.info("🔧 Подготовка sentiment данных...")
        
        # Убираем NaN
        initial_length = len(self.data)
        self.data = self.data.dropna(subset=['GOLD_Close', 'sentiment'])
        final_length = len(self.data)
        
        if final_length < initial_length:
            logger.warning(f"⚠️ Удалено {initial_length - final_length} строк с NaN")
        
        # НЕ нормализуем sentiment - он уже в диапазоне [-1, 1]
        logger.info(f"✅ Данные готовы: {len(self.data)} записей")

    def reset(self, seed=None):
        """Сброс среды"""
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # 0: нет позиции, 1: лонг
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0  # 🔥 Stop Loss цена
        self.trade_count = 0
        self.total_profit = 0.0
        self.sl_triggered_count = 0  # 🔥 Счетчик сработавших SL
        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Получает наблюдение: sentiment history + состояние портфеля
        """
        # 1. Sentiment history за lookback_window
        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step
        
        sentiment_history = []
        
        # Собираем sentiment за последние lookback_window шагов
        for i in range(start, end):
            if i < len(self.data):
                try:
                    # Используем enhanced sentiment
                    data_row = self.data.iloc[i]
                    enhanced_sentiment, confidence = get_enhanced_sentiment(
                        self.sentiment_analyzer, data_row
                    )
                    
                    # Если confidence высокий, используем enhanced, иначе базовый
                    if confidence > 0.4:
                        sentiment_history.append(enhanced_sentiment)
                    else:
                        sentiment_history.append(data_row['sentiment'])
                        
                except Exception as e:
                    # Fallback к базовому sentiment
                    sentiment_history.append(self.data.iloc[i]['sentiment'])
            else:
                sentiment_history.append(0.0)  # Padding
        
        # Дополняем до нужного размера
        while len(sentiment_history) < self.lookback_window:
            sentiment_history.insert(0, 0.0)
        
        # 2. Состояние портфеля с SL информацией
        current_price = self.data['GOLD_Close'].iloc[self.current_step]
        current_equity = self.balance
        if self.position == 1:
            current_equity += self.position_size * current_price
        
        unrealized_pnl = 0
        sl_distance = 0  # Расстояние до Stop Loss
        
        if self.position == 1 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            if self.stop_loss_price > 0:
                sl_distance = (current_price - self.stop_loss_price) / current_price
        
        portfolio_state = [
            self.balance / self.initial_balance,  # Нормализованный баланс
            float(self.position),                 # Позиция (0 или 1)
            self.position_size / max(1.0, self.balance / current_price) if current_price > 0 else 0,
            unrealized_pnl,                      # Нереализованный P&L
            sl_distance                          # 🔥 Расстояние до SL
        ]
        
        # 3. Объединяем
        observation = sentiment_history + portfolio_state
        observation = np.array(observation, dtype=np.float32)
        
        # Защита от NaN
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation

    def step(self, action):
        """
        Шаг в среде с STOP LOSS и ДИНАМИЧЕСКИМИ ПОЗИЦИЯМИ
        """
        current_price = self.data['GOLD_Close'].iloc[self.current_step]
        reward = 0
        sl_triggered = False
        
        # 🔥 ПРОВЕРКА STOP LOSS В ПЕРВУЮ ОЧЕРЕДЬ
        if self.position == 1 and self.stop_loss_price > 0:
            if current_price <= self.stop_loss_price:
                # Stop Loss сработал!
                sl_triggered = True
                
                # Принудительное закрытие позиции
                commission_cost = current_price * self.position_size * self.commission
                self.balance += current_price * self.position_size - commission_cost
                
                # Убыток от SL
                loss = (current_price - self.entry_price) * self.position_size - commission_cost
                self.total_profit += loss
                
                # Штраф за сработавший SL
                reward -= 10  # Штраф за неправильное управление рисками
                
                self.position = 0
                self.position_size = 0
                self.entry_price = 0
                self.stop_loss_price = 0
                self.sl_triggered_count += 1
                
                logger.debug(f"🛑 Stop Loss triggered at {current_price:.2f}, Loss: ${loss:.2f}")
        
        # Получаем текущий sentiment для торговых решений
        try:
            current_data_row = self.data.iloc[self.current_step]
            enhanced_sentiment, confidence = get_enhanced_sentiment(
                self.sentiment_analyzer, current_data_row
            )
            
            if confidence > 0.4:
                current_sentiment = enhanced_sentiment
            else:
                current_sentiment = current_data_row['sentiment']
                
        except Exception:
            current_sentiment = self.data.iloc[self.current_step]['sentiment']
        
        # ТОРГОВАЯ ЛОГИКА (если SL не сработал)
        if not sl_triggered:
            
            if action == 1:  # Buy
                if self.position == 0:  # Открываем лонг только если нет позиции
                    
                    # 🔥 ДИНАМИЧЕСКИЙ размер позиции на основе sentiment
                    sentiment_strength = abs(current_sentiment)
                    position_multiplier = self.min_position_size + (sentiment_strength * (self.max_position_size - self.min_position_size))
                    position_multiplier = np.clip(position_multiplier, self.min_position_size, self.max_position_size)
                    
                    self.position = 1
                    self.position_size = (self.balance * position_multiplier) / current_price
                    self.entry_price = current_price
                    
                    # 🔥 Устанавливаем Stop Loss
                    self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    
                    self.balance -= self.position_size * current_price * (1 + self.commission)
                    self.trade_count += 1
                    
                    # Награда за покупку при позитивном sentiment
                    if current_sentiment > 0.1:
                        reward += current_sentiment * 3 + sentiment_strength * 2  # Бонус за силу сигнала
                    else:
                        reward -= 1  # Мягкий штраф за покупку в негативе
                    
                    logger.debug(f"📈 BUY: Size={position_multiplier:.1%}, SL={self.stop_loss_price:.2f}")
                        
            elif action == 2:  # Sell
                if self.position == 1:  # Закрываем лонг
                    commission_cost = current_price * self.position_size * self.commission
                    self.balance += current_price * self.position_size - commission_cost
                    
                    # Прибыль/убыток
                    profit = (current_price - self.entry_price) * self.position_size - commission_cost
                    self.total_profit += profit
                    
                    # Награда пропорциональна прибыли
                    profit_ratio = profit / (self.entry_price * self.position_size)
                    reward += profit_ratio * 50  # Основная награда
                    
                    # Бонус за продажу при негативном sentiment
                    if current_sentiment < -0.1:
                        reward += abs(current_sentiment) * 2
                    
                    # Бонус за избежание SL
                    if profit > 0:
                        reward += 2  # Бонус за прибыльную сделку
                    
                    self.position = 0
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss_price = 0
                    
                    logger.debug(f"📉 SELL: Profit=${profit:.2f}, P&L={profit_ratio:.1%}")
            
            # Hold (action == 0) - адаптивный штраф
            elif action == 0:
                if abs(current_sentiment) < 0.05:
                    reward += 0.1  # Бонус за hold при неопределенности
                else:
                    reward -= 0.05  # Мягкий штраф за упущенную возможность
        
        # Дополнительные rewards
        
        # 1. Reward за удержание прибыльной позиции
        if self.position == 1 and not sl_triggered:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            
            if unrealized_pnl > 0:
                # Бонус за удержание прибыльной позиции при позитивном sentiment
                if current_sentiment > 0:
                    reward += unrealized_pnl * current_sentiment * 5
            else:
                # Штраф за удержание убыточной позиции при негативном sentiment
                if current_sentiment < 0:
                    reward += unrealized_pnl * abs(current_sentiment) * 3  # Отрицательный reward
        
        # 2. Проверка на критические просадки
        current_equity = self.balance
        if self.position == 1:
            current_equity += self.position_size * current_price
        
        equity_ratio = current_equity / self.initial_balance
        
        if equity_ratio < 0.7:  # Просадка > 30%
            reward -= 20
            done = True  # Принудительное завершение
        else:
            done = False
        
        self.current_step += 1
        
        # Обычные условия завершения
        if self.current_step >= len(self.data) - 1:
            done = True
        
        return self._get_observation(), reward, done, False, {
            'sl_triggered': sl_triggered,
            'current_sentiment': current_sentiment
        }

    def get_portfolio_stats(self):
        """Получение статистики портфеля с SL информацией"""
        current_price = self.data['GOLD_Close'].iloc[self.current_step - 1] if self.current_step > 0 else self.data['GOLD_Close'].iloc[0]
        current_equity = self.balance
        
        if self.position == 1:
            current_equity += self.position_size * current_price
        
        total_return = (current_equity - self.initial_balance) / self.initial_balance
        
        return {
            'balance': self.balance,
            'position': self.position,
            'current_equity': current_equity,
            'total_return': total_return,
            'trade_count': self.trade_count,
            'total_profit': self.total_profit,
            'sl_triggered_count': self.sl_triggered_count,  # 🔥 Количество сработавших SL
            'sl_rate': self.sl_triggered_count / max(1, self.trade_count) * 100  # 🔥 Процент SL
        }

# ПРОСТОЙ PPO АГЕНТ (из лучшего)
class SimpleActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SimpleActorCritic, self).__init__()
        
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

class SimpleMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class SentimentPPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, hidden_dim=128):
        self.gamma = gamma
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = SimpleActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old = SimpleActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        
        logger.info(f"🤖 Sentiment PPO Agent: LR={lr}, Hidden={hidden_dim}, Device={self.device}")
    
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
        if len(memory.rewards) <= 1:
            return 0.0
        
        # Discounted rewards
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
        
        # PPO update
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)
            
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean().item()

def main():
    logger.info("🚀 SENTIMENT-ONLY RL TRADING AGENT")
    logger.info("🎯 Торгует ТОЛЬКО на sentiment, игнорируя все остальные фичи")
    logger.info("=" * 60)
    
    # Загрузка данных
    try:
        possible_files = [
            'master_training_dataset.csv',
            'gold_data_with_sentiment_hourly.csv',
            'automated_gold_data_with_sentiment.csv'
        ]
        
        data_df = None
        for filename in possible_files:
            try:
                data_df = pd.read_csv(filename, index_col=0, parse_dates=True)
                logger.info(f"✅ Загружен: {filename}")
                break
            except FileNotFoundError:
                continue
        
        if data_df is None:
            raise FileNotFoundError("❌ Датасет не найден")
        
        logger.info(f"📊 Данные: {data_df.shape}")
        logger.info(f"📅 Период: {data_df.index.min()} → {data_df.index.max()}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки: {e}")
        return
    
    # Разделение данных
    train_size = int(len(data_df) * 0.8)
    train_data = data_df.iloc[:train_size].copy()
    test_data = data_df.iloc[train_size:].copy()
    
    logger.info(f"🔄 Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Создание среды
    try:
        env = SentimentOnlyTradingEnvironment(train_data, lookback_window=12)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        logger.info(f"🎯 Среда создана: State={state_dim}, Actions={action_dim}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания среды: {e}")
        return
    
    # Создание агента
    agent = SentimentPPOAgent(state_dim, action_dim, lr=2e-4, hidden_dim=128)
    memory = SimpleMemory()
    
    # Параметры обучения
    num_episodes = 150
    update_timestep = 500
    
    logger.info(f"🎓 Параметры обучения:")
    logger.info(f"   Episodes: {num_episodes}")
    logger.info(f"   Update frequency: {update_timestep}")
    logger.info("=" * 60)
    
    # Обучение
    timestep = 0
    best_return = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        try:
            while True:
                timestep += 1
                action = agent.select_action(state, memory)
                state, reward, done, _, _ = env.step(action)
                
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                episode_reward += reward
                
                if timestep % update_timestep == 0:
                    loss = agent.update(memory)
                    memory.clear()
                
                if done:
                    break
        
        except Exception as e:
            logger.error(f"❌ Ошибка в эпизоде {episode}: {e}")
            break
        
        # Статистика
        stats = env.get_portfolio_stats()
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode:3d} | "
                       f"Reward: {episode_reward:8.2f} | "
                       f"Return: {stats['total_return']*100:6.2f}% | "
                       f"Trades: {stats['trade_count']:3d} | "
                       f"SL Rate: {stats['sl_rate']:4.1f}% | "
                       f"Profit: ${stats['total_profit']:8.2f}")
        
        # Отслеживаем лучший результат БЕЗ сохранения
        if stats['total_return'] > best_return:
            best_return = stats['total_return']
            # Убрали сохранение модели для чистоты эксперимента
    
    # Тестирование
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТИРОВАНИЕ НА ОТЛОЖЕННЫХ ДАННЫХ")
    logger.info("="*60)
    
    try:
        test_env = SentimentOnlyTradingEnvironment(test_data, lookback_window=12)
        test_state, _ = test_env.reset()
        
        steps = 0
        max_steps = len(test_data) - 15
        
        while steps < max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(test_state).to(agent.device).unsqueeze(0)
                action_probs, _ = agent.policy_old(state_tensor)
                action = torch.argmax(action_probs).item()
            
            test_state, _, done, _, _ = test_env.step(action)
            steps += 1
            
            if done:
                break
        
        test_stats = test_env.get_portfolio_stats()
        
        print(f"\n🏆 РЕЗУЛЬТАТЫ SENTIMENT + SL + DYNAMIC TRADING:")
        print(f"📈 Test Return:     {test_stats['total_return']*100:8.2f}%")
        print(f"💰 Final Equity:    ${test_stats['current_equity']:8.2f}")
        print(f"🔄 Total Trades:    {test_stats['trade_count']:8d}")
        print(f"💵 Total Profit:    ${test_stats['total_profit']:8.2f}")
        print(f"🛑 SL Triggered:    {test_stats['sl_triggered_count']:8d} ({test_stats['sl_rate']:.1f}%)")
        print("="*60)
        
        if test_stats['total_return'] > 0.05:
            print("✅ УСПЕХ! Sentiment торговля прибыльна!")
        else:
            print("⚠️ Sentiment торговля требует доработки")
    
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
    
    logger.info("✅ Sentiment-Only обучение завершено!")

if __name__ == '__main__':
    main()