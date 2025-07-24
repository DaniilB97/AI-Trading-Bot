# dynamic_position_sizing.py
"""
Динамический размер позиции на основе уверенности модели и sentiment
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

class ConfidenceBasedPositionSizer:
    def __init__(self, 
                 min_position_pct=0.05,  # 5% минимум
                 max_position_pct=0.20,  # 20% максимум
                 base_position_pct=0.10): # 10% базовый размер
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.base_position_pct = base_position_pct
        
    def calculate_position_size(self, 
                              action_probabilities: torch.Tensor,
                              sentiment_score: float,
                              current_balance: float,
                              current_price: float,
                              market_conditions: Dict = None) -> float:
        """
        Вычисляет размер позиции на основе:
        1. Уверенности модели (action probabilities)
        2. Силы sentiment сигнала
        3. Текущих рыночных условий
        
        Returns: количество единиц для покупки
        """
        
        # 1. Уверенность модели (0 до 1)
        max_prob = torch.max(action_probabilities).item()
        model_confidence = max_prob  # Чем больше максимальная вероятность, тем увереннее модель
        
        # 2. Сила sentiment сигнала (0 до 1)
        sentiment_strength = abs(sentiment_score)  # |sentiment| от 0 до 1
        
        # 3. Комбинированная уверенность
        # Модель уверена И sentiment сильный = большая позиция
        combined_confidence = (model_confidence * 0.6 + sentiment_strength * 0.4)
        
        # 4. Дополнительные факторы
        additional_multiplier = 1.0
        
        if market_conditions:
            # Увеличиваем размер при низкой волатильности
            volatility = market_conditions.get('volatility', 0.5)
            if volatility < 0.3:
                additional_multiplier *= 1.2
            elif volatility > 0.7:
                additional_multiplier *= 0.8
                
            # Учитываем текущий drawdown
            drawdown = market_conditions.get('drawdown', 0)
            if drawdown < -0.10:  # Если просадка больше 10%
                additional_multiplier *= 0.7  # Уменьшаем риск
        
        # 5. Рассчитываем финальный процент от депозита
        position_pct = (self.min_position_pct + 
                       (self.max_position_pct - self.min_position_pct) * combined_confidence)
        
        position_pct *= additional_multiplier
        
        # Ограничиваем пределами
        position_pct = np.clip(position_pct, self.min_position_pct, self.max_position_pct)
        
        # 6. Конвертируем в количество единиц
        position_value = current_balance * position_pct
        position_size = position_value / current_price
        
        # Логируем для отладки
        print(f"🎯 Position Sizing:")
        print(f"   Model confidence: {model_confidence:.3f}")
        print(f"   Sentiment strength: {sentiment_strength:.3f}")
        print(f"   Combined confidence: {combined_confidence:.3f}")
        print(f"   Position %: {position_pct:.1%}")
        print(f"   Position size: {position_size:.4f} units")
        print(f"   Position value: ${position_value:.2f}")
        
        return position_size

class EnhancedTradingEnvironment:
    """
    Улучшенная торговая среда с динамическим размером позиции
    """
    
    def __init__(self, data, initial_balance=10000, commission=0.001):
        # Ваша стандартная инициализация...
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Инициализируем position sizer
        self.position_sizer = ConfidenceBasedPositionSizer(
            min_position_pct=0.05,  # 5%
            max_position_pct=0.20,  # 20%
            base_position_pct=0.10  # 10%
        )
        
        # Состояние среды
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: нет позиции, 1: лонг
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # Для расчета рыночных условий
        self.equity_history = []
        self.volatility_window = 24  # 24 часа
        
    def step(self, action, action_probabilities):
        """
        Шаг среды с динамическим размером позиции
        """
        current_price = self.data['GOLD_Close'].iloc[self.current_step]
        current_sentiment = self.data['sentiment'].iloc[self.current_step]
        
        # Рассчитываем текущие рыночные условия
        market_conditions = self._get_market_conditions()
        
        reward = 0
        
        if action == 1:  # BUY
            if self.position == 0:  # Открываем позицию только если нет текущей
                
                # 🔥 ДИНАМИЧЕСКИЙ РАЗМЕР ПОЗИЦИИ
                dynamic_position_size = self.position_sizer.calculate_position_size(
                    action_probabilities=action_probabilities,
                    sentiment_score=current_sentiment,
                    current_balance=self.balance,
                    current_price=current_price,
                    market_conditions=market_conditions
                )
                
                # Проверяем, хватает ли денег
                required_money = dynamic_position_size * current_price * (1 + self.commission)
                
                if required_money <= self.balance:
                    self.position = 1
                    self.position_size = dynamic_position_size
                    self.entry_price = current_price
                    self.balance -= required_money
                    
                    # Награда пропорциональна размеру позиции и уверенности
                    confidence_bonus = torch.max(action_probabilities).item()
                    reward += confidence_bonus * 5 + abs(current_sentiment) * 3
                    
                else:
                    # Недостаточно средств - берем максимум возможного
                    max_possible_size = (self.balance * 0.95) / (current_price * (1 + self.commission))
                    if max_possible_size > 0:
                        self.position = 1
                        self.position_size = max_possible_size
                        self.entry_price = current_price
                        self.balance -= max_possible_size * current_price * (1 + self.commission)
                        reward += 1  # Меньшая награда за частичную позицию
                    
        elif action == 2:  # SELL
            if self.position == 1:  # Закрываем позицию
                
                # Продаем всю позицию
                sale_proceeds = self.position_size * current_price * (1 - self.commission)
                self.balance += sale_proceeds
                
                # Рассчитываем прибыль
                profit = (current_price - self.entry_price) * self.position_size
                profit_pct = profit / (self.entry_price * self.position_size)
                
                # Награда пропорциональна прибыли и размеру позиции
                reward += profit_pct * 100 * (self.position_size / (self.balance / current_price))
                
                # Бонус за правильное использование sentiment
                if current_sentiment < -0.1:  # Продаем при негативном sentiment
                    reward += abs(current_sentiment) * 5
                
                # Сбрасываем позицию
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
        
        # Hold (action == 0) - адаптивная награда
        else:
            if abs(current_sentiment) < 0.1:
                reward += 0.5  # Бонус за hold при неопределенности
            else:
                reward -= 0.2  # Легкий штраф за пропуск сильного сигнала
        
        # Обновляем историю для расчета волатильности
        current_equity = self.balance
        if self.position == 1:
            current_equity += self.position_size * current_price
            
        self.equity_history.append(current_equity)
        if len(self.equity_history) > self.volatility_window:
            self.equity_history.pop(0)
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, {
            'position_size': self.position_size,
            'position_value_pct': (self.position_size * current_price) / current_equity if current_equity > 0 else 0
        }
    
    def _get_market_conditions(self) -> Dict:
        """Рассчитывает текущие рыночные условия"""
        conditions = {}
        
        # Волатильность на основе истории equity
        if len(self.equity_history) >= 5:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            conditions['volatility'] = np.std(returns) * np.sqrt(24 * 365)  # Годовая волатильность
        else:
            conditions['volatility'] = 0.5  # Средняя волатильность по умолчанию
        
        # Текущий drawdown
        if self.equity_history:
            peak = max(self.equity_history)
            current = self.equity_history[-1]
            conditions['drawdown'] = (current - peak) / peak if peak > 0 else 0
        else:
            conditions['drawdown'] = 0
        
        return conditions
    
    def _get_observation(self):
        """Получение наблюдения - нужно адаптировать под вашу модель"""
        # Здесь ваша логика создания observation
        # Включите информацию о текущем размере позиции
        pass

# Модификация PPO агента для работы с вероятностями
class ConfidenceAwarePPOAgent:
    """PPO агент, который возвращает action probabilities для position sizing"""
    
    def __init__(self, state_dim, action_dim, lr=1e-4):
        # Ваша стандартная инициализация PPO...
        pass
    
    def select_action_with_probabilities(self, state, memory):
        """
        Возвращает действие И вероятности для динамического sizing
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_probs, _ = self.policy_old(state_tensor)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Сохраняем для обучения
            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            
            return action.item(), action_probs.squeeze(0)  # Возвращаем и действие, и вероятности