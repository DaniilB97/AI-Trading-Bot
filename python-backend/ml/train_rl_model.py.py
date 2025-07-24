#!/usr/bin/env python3 # самый работающий пока что скрипт
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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Блок 1: Определение Среды (Environment) ---
class NewsTradingEnvironment(gym.Env):
    """
    Кастомная среда для торговли, где состояние включает:
    1. Технические индикаторы.
    2. Новостной сентимент.
    3. Состояние портфеля.
    """
    def __init__(self, data, initial_balance=10000, commission=0.001, lookback_window=30):
        super().__init__()
        if 'sentiment' not in data.columns:
            raise ValueError("Input data must contain a 'sentiment' column.")
            
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window
        
        self._add_technical_indicators()
        
        self.feature_columns = ['RSI_14', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5', 'sentiment',
        'ATR_14', 'DXY_change', 'VIX_change']
        
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        n_features = len(self.feature_columns)
        n_portfolio_state = 4 # баланс, позиция, размер, нереализованный PnL
        observation_space_size = n_features * lookback_window + n_portfolio_state
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32
        )
        for col in ['RSI_14', 'STOCHk_14_3_3', 'CCI_14_0.015', 'Price_Change_5']:
            if col in self.data.columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-9)

    def _add_technical_indicators(self):
        """Расчет и добавление технических индикаторов."""
        self.data.ta.rsi(length=14, append=True)
        self.data.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        self.data.ta.cci(length=14, append=True)
        self.data['Price_Change_5'] = self.data['Close'].pct_change(periods=5)
        self.data.ta.atr(length=14, append=True) # Let's name it 'ATR_14'
        self.data['DXY_change'] = self.data['DXY_Close'].pct_change()
        self.data['VIX_change'] = self.data['VIX_Close'].pct_change()
        
        self.data.bfill(inplace=True)
        self.data.dropna(inplace=True)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        return self._get_observation(), {}

    def _get_observation(self):
        """Формирует вектор состояния для агента."""
        start = self.current_step - self.lookback_window
        end = self.current_step
        
        market_state = self.data.iloc[start:end][self.feature_columns].values.flatten()
        
        current_price = self.data['Close'].iloc[end - 1]
        unrealized_pnl = 0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price if self.entry_price > 0 else 0

        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.position,
            self.position_size / (self.balance / current_price if current_price > 0 else 1),
            unrealized_pnl
        ]).flatten()
        
        return np.concatenate([market_state, portfolio_state]).astype(np.float32)

    def step(self, action):
        """Выполняет шаг в среде."""
        current_price = self.data['Close'].iloc[self.current_step]
        reward = 0
        
        if action == 1: # Buy
            if self.position <= 0: 
                self.position = 1
                self.position_size = (self.balance * 0.95) / current_price if current_price > 0 else 0
                self.entry_price = current_price
                self.balance -= self.position_size * current_price * self.commission
                self.trade_count += 1
                reward -= 1 
        elif action == 2: # Sell (в данном случае, только закрытие лонга)
            if self.position == 1:
                profit = (current_price - self.entry_price) * self.position_size
                self.balance += profit - (current_price * self.position_size * self.commission)
                reward += profit / self.initial_balance * 100
                self.position = 0
                self.position_size = 0

        if self.position != 0:
            if (self.current_step + 1) < len(self.data):
                next_price = self.data['Close'].iloc[self.current_step + 1]
                price_diff = (next_price - current_price) * self.position
                reward += price_diff / current_price * 100

        self.current_step += 1
        done = self.current_step >= len(self.data) - 2 or self.balance < self.initial_balance * 0.5
        
        return self._get_observation(), reward, done, False, {}

# --- Блок 2: Определение Агента (PPO) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, state):
        return self.actor(state), self.critic(state)

class Memory:
    def __init__(self): self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
    def clear(self): del self.actions[:]; del self.states[:]; del self.logprobs[:]; del self.rewards[:]; del self.is_terminals[:]

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy, self.policy_old = ActorCritic(state_dim, action_dim).to(self.device), ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        
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
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).to(self.device).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).to(self.device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).to(self.device).detach()

        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            
            # --- ИЗМЕНЕНИЕ: Убираем лишнее измерение у state_values ---
            state_values = torch.squeeze(state_values)

            dist = Categorical(action_probs)
            logprobs, dist_entropy = dist.log_prob(old_actions), dist.entropy()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1, surr2 = ratios * advantages, torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean().item()

# --- Блок 3: Главный цикл обучения ---
def main():
    logger.info("🚀 Starting RL Gold Trader Training...")
    try:
        data_df = pd.read_csv('gold_data_with_sentiment_hourly.csv', index_col='Datetime', parse_dates=True)
        logger.info(f"Successfully loaded dataset with shape: {data_df.shape}")
    except FileNotFoundError:
        logger.error("❌ 'gold_data_with_sentiment_hourly.csv' not found. Please run data_pipeline.py first.")
        return

    env = NewsTradingEnvironment(data_df)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    agent, memory = PPOAgent(state_dim, action_dim), Memory()

    num_episodes, update_timestep, timestep = 500, 2000, 0
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        try:
            for t in range(1, len(env.data) - env.lookback_window):
                timestep += 1
                action = agent.select_action(state, memory)
                state, reward, done, _, _ = env.step(action)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                episode_reward += reward
                if timestep % update_timestep == 0:
                    loss = agent.update(memory); memory.clear()
                    logger.info(f"Policy updated at timestep {timestep}. Loss: {loss:.4f}")
                if done: break
        except IndexError as e:
            logger.error(f"IndexError during episode {episode}: {e}"); break
        
        logger.info(f"Episode {episode} | Reward: {episode_reward:.2f} | Final Balance: {env.balance:.2f} | Trades: {env.trade_count}")

    model_path = f"rl_gold_trader_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(agent.policy_old.state_dict(), model_path)
    logger.info(f"✅ Training complete. Model saved to {model_path}")

if __name__ == '__main__':
    main()
