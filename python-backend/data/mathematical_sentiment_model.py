# mathematical_sentiment_model.py
"""
Реализация математической модели анализа настроений для RL торговли золотом
Основана на взвешенной сумме различных источников sentiment
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SentimentSource:
    """Структура для источника sentiment"""
    name: str
    weight: float
    reliability: float
    current_value: float
    
class MathematicalSentimentAnalyzer:
    """
    Математическая модель анализа настроений S(t) = Σ wi × sentiment_i(t)
    
    Источники sentiment с весами:
    1. MarketAux API (новости) - высокий вес
    2. Fear & Greed Index - средний вес  
    3. Golden Cross состояние - средний вес
    4. VIX momentum - низкий вес
    5. Price momentum - низкий вес
    """
    
    def __init__(self, custom_weights: dict = None, custom_reliabilities: dict = None):
        self.sources = self._initialize_sources(custom_weights, custom_reliabilities)
        self.history = []

        logger.info("🧮 Математическая модель sentiment инициализирована")
    
    def _initialize_sources(self, custom_weights: dict = None, custom_reliabilities: dict = None) -> dict:
        """Инициализация источников с настраиваемыми весами"""
        
        # Дефолтные веса (если не переданы custom)
        default_weights = {
            'marketaux_news': 0.35,
            'fear_greed_index': 0.25, 
            'golden_cross_trend': 0.20,
            'vix_momentum': 0.10,
            'price_momentum': 0.10
        }
        
        # Дефолтные reliability (если не переданы custom)
        default_reliabilities = {
            'marketaux_news': 0.8,
            'fear_greed_index': 0.9,
            'golden_cross_trend': 0.7,
            'vix_momentum': 0.6,
            'price_momentum': 0.5
        }
        
        # Используем custom веса если переданы, иначе дефолтные
        weights = custom_weights if custom_weights else default_weights
        reliabilities = custom_reliabilities if custom_reliabilities else default_reliabilities
        
        sources = {
            'marketaux_news': SentimentSource(
                name='MarketAux News',
                weight=weights['marketaux_news'],
                reliability=reliabilities['marketaux_news'],
                current_value=0.0
            ),
            'fear_greed_index': SentimentSource(
                name='Fear & Greed Index',
                weight=weights['fear_greed_index'],
                reliability=reliabilities['fear_greed_index'],
                current_value=0.0
            ),
            'golden_cross_trend': SentimentSource(
                name='Golden Cross Trend',
                weight=weights['golden_cross_trend'],
                reliability=reliabilities['golden_cross_trend'],
                current_value=0.0
            ),
            'vix_momentum': SentimentSource(
                name='VIX Momentum',
                weight=weights['vix_momentum'],
                reliability=reliabilities['vix_momentum'],
                current_value=0.0
            ),
            'price_momentum': SentimentSource(
                name='Price Momentum',
                weight=weights['price_momentum'],
                reliability=reliabilities['price_momentum'],
                current_value=0.0
            )
        }
        
        # Нормализация весов
        total_weight = sum(source.weight for source in sources.values())
        if abs(total_weight - 1.0) > 1e-6:
            for source in sources.values():
                source.weight /= total_weight
        
        return sources
    
    def update_marketaux_sentiment(self, marketaux_sentiment: float) -> None:
        """Обновление sentiment от MarketAux API"""
        self.sources['marketaux_news'].current_value = np.clip(marketaux_sentiment, -1, 1)
        logger.debug(f"📰 MarketAux sentiment обновлен: {marketaux_sentiment:.3f}")
    
    def update_fear_greed_sentiment(self, fear_greed_value: float) -> None:
        """Обновление Fear & Greed Index (0-100 → -1 to 1)"""
        # Конвертируем 0-100 в -1 до 1
        if fear_greed_value <= 25:
            normalized = -1 + (fear_greed_value / 25) * 0.5  # 0-25 → -1 to -0.5
        elif fear_greed_value <= 75:
            normalized = -0.5 + ((fear_greed_value - 25) / 50) * 1  # 25-75 → -0.5 to 0.5
        else:
            normalized = 0.5 + ((fear_greed_value - 75) / 25) * 0.5  # 75-100 → 0.5 to 1
        
        self.sources['fear_greed_index'].current_value = normalized
        logger.debug(f"😰 Fear&Greed sentiment обновлен: {fear_greed_value} → {normalized:.3f}")
    
    def update_golden_cross_sentiment(self, golden_cross_trend: float, ma_spread: float) -> None:
        """Обновление Golden Cross sentiment"""
        # Базовый сигнал от тренда (-1, 0, 1)
        base_sentiment = golden_cross_trend * 0.5  # Умножаем на 0.5 для умеренности
        
        # Модификация по силе сигнала (ma_spread)
        strength_modifier = np.tanh(ma_spread)  # Ограничиваем в пределах -1 to 1
        
        final_sentiment = base_sentiment + (strength_modifier * 0.3)
        self.sources['golden_cross_trend'].current_value = np.clip(final_sentiment, -1, 1)
        
        logger.debug(f"🏆 Golden Cross sentiment: trend={golden_cross_trend} + spread={ma_spread:.3f} → {final_sentiment:.3f}")
    
    def update_vix_sentiment(self, vix_change: float, vix_level: float) -> None:
        """Обновление VIX momentum sentiment"""
        # VIX рост = страх = негативный sentiment для золота (обычно)
        # Но высокий VIX иногда = demand for safe haven = позитивный для золота
        
        # Momentum компонент
        momentum_sentiment = -np.tanh(vix_change * 10)  # VIX рост = негативный
        
        # Level компонент  
        if vix_level > 30:  # Высокий страх
            level_sentiment = 0.3  # Позитивный для золота (safe haven)
        elif vix_level > 20:
            level_sentiment = 0.1
        else:
            level_sentiment = -0.1  # Низкий страх = негативный для золота
        
        combined_sentiment = 0.7 * momentum_sentiment + 0.3 * level_sentiment
        self.sources['vix_momentum'].current_value = np.clip(combined_sentiment, -1, 1)
        
        logger.debug(f"📉 VIX sentiment: change={vix_change:.4f}, level={vix_level:.1f} → {combined_sentiment:.3f}")
    
    def update_price_momentum_sentiment(self, price_change_1h: float, price_change_5h: float) -> None:
        """Обновление price momentum sentiment"""
        # Краткосрочный momentum
        short_momentum = np.tanh(price_change_1h * 20)
        
        # Среднесрочный momentum  
        medium_momentum = np.tanh(price_change_5h * 10)
        
        # Комбинируем с весами
        combined_sentiment = 0.6 * short_momentum + 0.4 * medium_momentum
        self.sources['price_momentum'].current_value = np.clip(combined_sentiment, -1, 1)
        
        logger.debug(f"⚡ Price momentum: 1h={price_change_1h:.4f}, 5h={price_change_5h:.4f} → {combined_sentiment:.3f}")
    
    def calculate_composite_sentiment(self) -> Tuple[float, Dict[str, float]]:
        """
        Основная формула: S(t) = Σ wi × sentiment_i(t)
        
        Returns:
            Tuple[float, Dict]: (composite_sentiment, source_contributions)
        """
        composite_sentiment = 0.0
        source_contributions = {}
        
        for name, source in self.sources.items():
            # Применяем вес и reliability
            contribution = source.weight * source.current_value * source.reliability
            composite_sentiment += contribution
            source_contributions[name] = contribution
            
            logger.debug(f"   {source.name}: {source.current_value:.3f} × {source.weight:.2f} × {source.reliability:.2f} = {contribution:.4f}")
        
        # Нормализуем в диапазон [-1, 1]
        composite_sentiment = np.clip(composite_sentiment, -1, 1)
        
        # Сохраняем в историю
        timestamp = datetime.now()
        self.history.append({
            'timestamp': timestamp,
            'composite_sentiment': composite_sentiment,
            'sources': {name: source.current_value for name, source in self.sources.items()},
            'contributions': source_contributions
        })
        
        #logger.info(f"🧮 Composite sentiment: {composite_sentiment:.3f}") убираем, чтобы не спамить 
        return composite_sentiment, source_contributions
    
    def get_sentiment_breakdown(self) -> Dict[str, float]:
        """Получение детализации по источникам"""
        breakdown = {}
        for name, source in self.sources.items():
            breakdown[name] = {
                'value': source.current_value,
                'weight': source.weight,
                'reliability': source.reliability,
                'weighted_contribution': source.current_value * source.weight * source.reliability
            }
        return breakdown

    def get_sentiment_confidence(self) -> float:
        values = [source.current_value for source in self.sources.values()]
        std_dev = np.std(values)
        return max(0, 1 - std_dev)
    
    def update_all_sources(self, data_row: pd.Series) -> float:
        """
        Обновление всех источников из строки данных и расчет composite sentiment
        
        Args:
            data_row: Строка данных с необходимыми колонками
            
        Returns:
            float: Composite sentiment
        """
        try:
            # MarketAux sentiment (если есть)
            if 'sentiment' in data_row and not pd.isna(data_row['sentiment']):
                self.update_marketaux_sentiment(data_row['sentiment'])
            
            # Fear & Greed (через VIX как прокси, если FG недоступен)
            if 'VIX_Close' in data_row:
                # Простая эвристика: VIX → Fear & Greed прокси
                vix_value = data_row['VIX_Close']
                fear_greed_proxy = max(0, min(100, 100 - (vix_value - 10) * 3))  # VIX 10-40 → FG 100-10
                self.update_fear_greed_sentiment(fear_greed_proxy)
            
            # Golden Cross
            if 'golden_cross_trend' in data_row and 'ma_spread_normalized' in data_row:
                self.update_golden_cross_sentiment(
                    data_row['golden_cross_trend'], 
                    data_row['ma_spread_normalized']
                )
            
            # VIX momentum
            if 'VIX_change' in data_row and 'VIX_Close' in data_row:
                self.update_vix_sentiment(data_row['VIX_change'], data_row['VIX_Close'])
            
            # Price momentum
            if 'Price_Change_1' in data_row and 'Price_Change_5' in data_row:
                self.update_price_momentum_sentiment(
                    data_row['Price_Change_1'], 
                    data_row['Price_Change_5']
                )
            
            # Рассчитываем композитный sentiment
            composite_sentiment, _ = self.calculate_composite_sentiment()
            return composite_sentiment
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления sentiment: {e}")
            return 0.0
    
    def get_sentiment_confidence(self) -> float:
        """Расчет уверенности в sentiment на основе согласованности источников"""
        values = [source.current_value for source in self.sources.values()]
        
        # Стандартное отклонение как мера разброса
        std_dev = np.std(values)
        
        # Конвертируем в confidence (низкий разброс = высокая уверенность)
        confidence = max(0, 1 - std_dev)
        
        return confidence
    
    def get_sentiment_history(self, last_n: int = 10) -> List[Dict]:
        """Получение истории sentiment"""
        return self.history[-last_n:] if self.history else []


# Пример использования для интеграции в RL модель
def get_enhanced_sentiment(analyzer: MathematicalSentimentAnalyzer, data_row: pd.Series) -> Tuple[float, float]:
    """
    ОПТИМИЗИРОВАННАЯ функция для получения улучшенного sentiment с confidence
    Теперь использует существующий анализатор вместо создания нового
    
    Args:
        analyzer: Уже созданный экземпляр MathematicalSentimentAnalyzer
        data_row: Строка данных из датасета
        
    Returns:
        Tuple[float, float]: (sentiment_value, confidence)
    """
    sentiment = analyzer.update_all_sources(data_row)
    confidence = analyzer.get_sentiment_confidence()
    
    return sentiment, confidence


if __name__ == "__main__":
    print("✅ Mathematical Sentiment Model загружен успешно")
    print("=" * 50)
    
    analyzer = MathematicalSentimentAnalyzer()
    
    # Симуляция обновлений
    analyzer.update_marketaux_sentiment(0.3)
    analyzer.update_fear_greed_sentiment(75)
    analyzer.update_golden_cross_sentiment(1.0, -0.5)
    analyzer.update_vix_sentiment(-0.02, 18.5)
    analyzer.update_price_momentum_sentiment(0.001, 0.005)
    
    # Расчет композитного sentiment
    composite, contributions = analyzer.calculate_composite_sentiment()
    
    print(f"\n📊 Результаты:")
    print(f"Композитный sentiment: {composite:.3f}")
    print(f"Уверенность: {analyzer.get_sentiment_confidence():.3f}")
    
    print(f"\n📋 Детализация по источникам:")
    breakdown = analyzer.get_sentiment_breakdown()
    for source, details in breakdown.items():
        print(f"  {source}:")
        print(f"    Значение: {details['value']:.3f}")
        print(f"    Вклад: {details['weighted_contribution']:.4f}")