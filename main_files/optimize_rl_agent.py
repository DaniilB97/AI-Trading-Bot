import optuna
import logging
import pandas as pd
import torch
import os
import json

# Импортируем все необходимые компоненты из вашего рабочего скрипта
from rl_gold_traderv0_01 import NewsTradingEnvironment, PPOAgent, Memory, ActorCritic

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ОБЪЕКТИВНАЯ ФУНКЦИЯ ДЛЯ OPTUNA ---
# Эта функция будет запускать полный цикл обучения и возвращать результат,
# который Optuna будет пытаться максимизировать.
def objective(trial, data_df):
    """
    Запускает один полный цикл обучения RL-агента и возвращает его финальный баланс.
    """
    logger.info(f"\n===== Starting Optuna Trial #{trial.number} =====")
    
    # 1. Optuna предлагает новые гиперпараметры для этого теста
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "K_epochs": trial.suggest_int("K_epochs", 4, 10),
        "update_timestep": trial.suggest_int("update_timestep", 1000, 5000, step=500),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512])
    }
    logger.info(f"Trial Parameters: {params}")

    # 2. Инициализация среды и агента с новыми параметрами
    # Используем фиксированный lookback_window и commission для консистентности
    env = NewsTradingEnvironment(data_df, lookback_window=30, commission=0.001)
    state_dim = env.observation_space.shape[0]
    action_dim = 3 # Hold, Buy, Sell
    
    agent = PPOAgent(
        state_dim, 
        action_dim, 
        lr=params["lr"], 
        gamma=params["gamma"], 
        K_epochs=params["K_epochs"]
    )
    # Используем ActorCritic с новым hidden_dim
    agent.policy = ActorCritic(state_dim, action_dim, hidden_dim=params["hidden_dim"]).to(agent.device)
    agent.policy_old = ActorCritic(state_dim, action_dim, hidden_dim=params["hidden_dim"]).to(agent.device)
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=params["lr"])


    memory = Memory()
    
    # 3. Запуск укороченного цикла обучения для быстрой оценки
    # Мы используем меньше эпизодов, чтобы один тест не занимал 10 часов
    num_episodes = 100 # Уменьшенное количество для быстрой оценки
    timestep = 0
    final_balance = 0
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        while not done:
            timestep += 1
            action = agent.select_action(state, memory)
            state, reward, done, _, _ = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if timestep % params["update_timestep"] == 0:
                agent.update(memory)
                memory.clear()

        final_balance = env.balance
        # Логируем только каждый 10-й эпизод, чтобы не засорять вывод
        if episode % 10 == 0:
            logger.info(f"Trial #{trial.number}, Episode {episode} | Final Balance: {final_balance:.2f}")

    logger.info(f"===== Finished Optuna Trial #{trial.number} | Final Balance: {final_balance:.2f} =====\n")
    
    # 4. Возвращаем финальный баланс. Optuna будет пытаться его максимизировать.
    return final_balance


# --- ГЛАВНЫЙ БЛОК ЗАПУСКА ---
if __name__ == '__main__':
    # 1. Загружаем данные один раз
    try:
        data_df = pd.read_csv('gold_data_with_sentiment_hourly.csv', index_col='Datetime', parse_dates=True)
        logger.info(f"Successfully loaded dataset with shape: {data_df.shape}")
    except FileNotFoundError:
        logger.error("❌ 'gold_data_with_sentiment_hourly.csv' not found. Please run data_pipeline.py first.")
        exit()

    # 2. Создаем и запускаем исследование Optuna
    # n_jobs=-1 задействует все ядра вашего процессора для параллельного выполнения тестов
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_df), n_trials=50, n_jobs=-1)
    
    # 3. Выводим результаты
    logger.info("\n🎉 Optuna Optimization Finished! 🎉")
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info(f"Best value (Final Balance): ${study.best_trial.value:.2f}")
    logger.info("Best hyperparameters found:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  - {key}: {value}")
        
    # 4. Сохраняем лучшие параметры в файл
    with open("best_rl_hyperparams.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    logger.info("✅ Best hyperparameters saved to best_rl_hyperparams.json")
