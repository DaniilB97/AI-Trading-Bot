#!/usr/bin/env python3
"""
ПОЛНАЯ Optuna оптимизация: RL параметры + Sentiment веса + Reliability
"""
import optuna
import torch
import pandas as pd
import logging
from updated_train_rl_model import *

def objective(trial):
    """Оптимизация ВСЕХ параметров"""
    
    # 1. RL ПАРАМЕТРЫ
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    lookback_window = trial.suggest_categorical('lookback_window', [12, 24, 48])
    
    # 2. SENTIMENT ВЕСА (должны суммироваться к 1.0)
    marketaux_weight = trial.suggest_float('marketaux_weight', 0.15, 0.50)
    fear_greed_weight = trial.suggest_float('fear_greed_weight', 0.10, 0.40)
    golden_cross_weight = trial.suggest_float('golden_cross_weight', 0.05, 0.35)
    vix_weight = trial.suggest_float('vix_weight', 0.05, 0.20)
    # Price weight = оставшееся до 1.0
    
    # Нормализация весов
    total_weight = marketaux_weight + fear_greed_weight + golden_cross_weight + vix_weight
    remaining_weight = max(0.05, 1.0 - total_weight)  # Минимум 5% для price
    
    sentiment_weights = {
        'marketaux_news': marketaux_weight / (total_weight + remaining_weight),
        'fear_greed_index': fear_greed_weight / (total_weight + remaining_weight),
        'golden_cross_trend': golden_cross_weight / (total_weight + remaining_weight),
        'vix_momentum': vix_weight / (total_weight + remaining_weight),
        'price_momentum': remaining_weight / (total_weight + remaining_weight)
    }
    
    # 3. SENTIMENT RELIABILITY
    sentiment_reliabilities = {
        'marketaux_news': trial.suggest_float('marketaux_reliability', 0.6, 0.95),
        'fear_greed_index': trial.suggest_float('fear_greed_reliability', 0.7, 0.99),
        'golden_cross_trend': trial.suggest_float('golden_cross_reliability', 0.5, 0.85),
        'vix_momentum': trial.suggest_float('vix_reliability', 0.4, 0.75),
        'price_momentum': trial.suggest_float('price_reliability', 0.3, 0.70)
    }
    
    # 4. REWARD ПАРАМЕТРЫ
    golden_cross_reward_bonus = trial.suggest_float('gc_reward_bonus', 1.0, 5.0)
    drawdown_penalty = trial.suggest_float('drawdown_penalty', 5.0, 30.0)
    
    try:
        # Загрузка данных
        data_df = pd.read_csv('master_training_dataset.csv', index_col=0, parse_dates=True)
        train_data, val_data, test_data = split_data_by_time(data_df, 0.6, 0.2)
        
        # Создание среды с оптимизируемыми параметрами
        env = create_optimized_environment(
            train_data, lookback_window, sentiment_weights, sentiment_reliabilities,
            golden_cross_reward_bonus, drawdown_penalty
        )
        
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, 
                        lr=lr, gamma=gamma, hidden_dim=hidden_dim)
        
        # Быстрое обучение для optuna
        episodes = trial.suggest_categorical('episodes', [30, 50, 80])
        val_performance = train_and_validate(env, agent, train_data, val_data, episodes)
        
        return val_performance
        
    except Exception as e:
        logging.error(f"Trial failed: {e}")
        return -1.0

def create_optimized_environment(data, lookback_window, sentiment_weights, 
                               sentiment_reliabilities, gc_bonus, dd_penalty):
    """Создание среды с оптимизированными параметрами"""
    
    # Создаем кастомную среду
    class OptimizedEnvironment(EnhancedGoldTradingEnvironment):
        def __init__(self, data, lookback_window):
            # Сохраняем параметры для sentiment анализатора
            self.sentiment_weights = sentiment_weights
            self.sentiment_reliabilities = sentiment_reliabilities
            self.gc_bonus = gc_bonus
            self.dd_penalty = dd_penalty
            
            super().__init__(data, lookback_window=lookback_window)
        
        def step(self, action):
            # Вызываем оригинальный step
            state, reward, done, truncated, info = super().step(action)
            
            # Модифицируем reward с оптимизированными параметрами
            current_price = self.data['GOLD_Close'].iloc[self.current_step]
            
            # Golden Cross бонус (оптимизируемый)
            if 'golden_cross_trend' in self.data.columns:
                gc_trend = self.data['golden_cross_trend'].iloc[self.current_step]
                if action == 1 and gc_trend > 0:
                    reward += self.gc_bonus
                elif action == 1 and gc_trend < 0:
                    reward -= self.gc_bonus
            
            # Штраф за просадки (оптимизируемый)
            current_equity = self.balance
            if self.position == 1:
                current_equity += self.position_size * current_price
            
            equity_ratio = current_equity / self.initial_balance
            if equity_ratio < 0.8:
                reward -= self.dd_penalty
            
            return state, reward, done, truncated, info
    
    return OptimizedEnvironment(data, lookback_window)

def train_and_validate(env, agent, train_data, val_data, episodes):
    """Быстрое обучение и валидация"""
    memory = Memory()
    
    # Обучение
    for episode in range(episodes):
        state, _ = env.reset()
        
        for step in range(800):  # Короткие эпизоды для скорости
            action = agent.select_action(state, memory)
            state, reward, done, _, _ = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if step % 100 == 0:
                agent.update(memory)
                memory.clear()
            
            if done:
                break
    
    # Валидация
    val_env = create_optimized_environment(
        val_data, env.lookback_window, env.sentiment_weights, 
        env.sentiment_reliabilities, env.gc_bonus, env.dd_penalty
    )
    
    state, _ = val_env.reset()
    
    for step in range(min(400, len(val_data)-50)):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = agent.policy_old(state_tensor)
            action = torch.argmax(action_probs).item()
        
        state, _, done, _, _ = val_env.step(action)
        if done:
            break
    
    # Возвращаем комбинированную метрику
    stats = val_env.get_portfolio_stats()
    return_score = stats['total_return']
    sharpe_score = stats.get('sharpe_ratio', 0) * 0.1  # Бонус за Sharpe
    win_rate_score = stats.get('win_rate', 0) * 0.1    # Бонус за Win Rate
    
    combined_score = return_score + sharpe_score + win_rate_score
    return combined_score

def run_comprehensive_optimization():
    """Запуск полной оптимизации"""
    print("🔬 КОМПЛЕКСНАЯ OPTUNA ОПТИМИЗАЦИЯ")
    print("=" * 50)
    print("🎯 Оптимизируем:")
    print("  • RL параметры (lr, gamma, hidden_dim, lookback)")
    print("  • Sentiment веса (5 источников)")
    print("  • Sentiment reliability (5 значений)")
    print("  • Reward модификаторы")
    print("=" * 50)
    
    study = optuna.create_study(direction='maximize')
    
    # Разные стадии оптимизации
    print("🚀 Стадия 1: Быстрый поиск (50 trials)")
    study.optimize(objective, n_trials=50)
    
    print("🎯 Стадия 2: Средний поиск (100 trials)")
    study.optimize(objective, n_trials=100)
    
    print("🏆 Стадия 3: Финальная настройка (50 trials)")
    study.optimize(objective, n_trials=50)
    
    # Результаты
    print("\n🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
    print("=" * 50)
    best_params = study.best_trial.params
    
    print(f"📈 Best Score: {study.best_trial.value:.4f}")
    print("\n🤖 RL Параметры:")
    for param in ['learning_rate', 'gamma', 'hidden_dim', 'lookback_window', 'episodes']:
        if param in best_params:
            print(f"  {param}: {best_params[param]}")
    
    print("\n🧠 Sentiment Веса:")
    for param in ['marketaux_weight', 'fear_greed_weight', 'golden_cross_weight', 'vix_weight']:
        if param in best_params:
            print(f"  {param}: {best_params[param]:.3f}")
    
    print("\n🔍 Sentiment Reliability:")
    for param in ['marketaux_reliability', 'fear_greed_reliability', 'golden_cross_reliability']:
        if param in best_params:
            print(f"  {param}: {best_params[param]:.3f}")
    
    print("\n🎁 Reward Модификаторы:")
    for param in ['gc_reward_bonus', 'drawdown_penalty']:
        if param in best_params:
            print(f"  {param}: {best_params[param]:.1f}")
    
    return study.best_trial.params

if __name__ == "__main__":
    best_params = run_comprehensive_optimization()
    
    # Сохраняем лучшие параметры
    import json
    with open('best_optuna_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n💾 Лучшие параметры сохранены в best_optuna_params.json")