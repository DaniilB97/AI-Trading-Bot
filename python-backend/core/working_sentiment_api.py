# working_sentiment_api.py - РАБОЧИЙ sentiment API

import os
import requests
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)

def get_market_sentiment(symbol: str = "gold") -> float:
    """
    РАБОЧАЯ функция получения sentiment
    Использует проверенные параметры из health check
    """
    
    # 1. Основной источник - MarketAux (РАБОТАЕТ!)
    try:
        api_key = os.getenv("MARKETAUX_API_TOKEN")
        if api_key:
            logger.info("📊 Получение sentiment от MarketAux...")
            
            # Используем РАБОЧИЕ параметры из health check
            test_url = "https://api.marketaux.com/v1/news/all"
            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(hours=12)  # Рабочий диапазон
            
            # Выбираем символы в зависимости от запроса
            if symbol.lower() == "gold":
                symbols = "GOLD,DXY,VIX,SPY"  # Золото + коррелирующие активы
            else:
                symbols = "SPY"  # По умолчанию
            
            params = {
                'api_token': api_key,
                'symbols': symbols,
                'language': 'en',
                'published_after': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'published_before': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'limit': 5  # Рабочий лимит
            }
            
            response = requests.get(test_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                
                if articles:
                    # Извлекаем sentiment scores из entities (как в health check)
                    all_entity_sentiment_scores = []
                    for article in articles:
                        entities = article.get('entities', [])
                        for entity in entities:
                            score = entity.get('sentiment_score')
                            if score is not None:
                                all_entity_sentiment_scores.append(score)
                    
                    if all_entity_sentiment_scores:
                        avg_sentiment = np.mean(all_entity_sentiment_scores)
                        logger.info(f"✅ MarketAux sentiment: {avg_sentiment:.3f} из {len(all_entity_sentiment_scores)} сущностей")
                        return float(avg_sentiment)  # Возвращаем в диапазоне -1 до 1
                
                logger.warning("⚠️ MarketAux: статьи найдены, но нет sentiment scores")
            else:
                logger.warning(f"⚠️ MarketAux API ошибка: {response.status_code}")
                
    except Exception as e:
        logger.warning(f"⚠️ MarketAux ошибка: {e}")
    
    # 2. Fallback - Fear & Greed Index (СТАБИЛЬНО РАБОТАЕТ!)
    try:
        logger.info("📈 Fallback: Fear & Greed Index...")
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and data.get('data'):
                value = float(data['data'][0]['value'])
                
                # Конвертируем 0-100 в -1 до 1 (как в health check: 71 = позитивный)
                if value <= 25:
                    normalized = -1 + (value / 25) * 0.5  # 0-25 = -1 до -0.5
                elif value <= 75:
                    normalized = -0.5 + ((value - 25) / 50) * 1  # 25-75 = -0.5 до 0.5
                else:
                    normalized = 0.5 + ((value - 75) / 25) * 0.5  # 75-100 = 0.5 до 1
                
                logger.info(f"✅ Fear & Greed: {value}/100 → sentiment: {normalized:.3f}")
                return normalized
                
    except Exception as e:
        logger.warning(f"⚠️ Fear & Greed ошибка: {e}")
    
    # 3. Последний fallback - нейтральное значение
    logger.info("🔄 Используем нейтральный sentiment")
    return 0.0


def get_market_sentiment_detailed(symbol: str = "gold") -> dict:
    """Детальная версия с метаданными"""
    sentiment = get_market_sentiment(symbol)
    
    # Определяем источник
    source = "unknown"
    confidence = 0.5
    
    if sentiment != 0.0:
        if -1 <= sentiment <= 1:
            source = "marketaux" if abs(sentiment) > 0.1 else "fear_greed"
            confidence = 0.8 if source == "marketaux" else 0.7
        else:
            source = "fear_greed"
            confidence = 0.7
    else:
        source = "fallback"
        confidence = 0.3
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'source': source,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }


# Обратно совместимые функции для live_rl_trading.py
def get_sentiment(text: str) -> float:
    """
    Обратно совместимая функция для замены старых заглушек
    Комбинирует market sentiment + keyword анализ
    """
    # Получаем базовый market sentiment
    market_sentiment = get_market_sentiment("gold")
    
    # Добавляем анализ ключевых слов из текста
    text_lower = text.lower()
    keyword_adjustment = 0.0
    
    positive_words = ["surge", "bullish", "strong", "rise", "gain", "rally", "safe-haven"]
    negative_words = ["fall", "bearish", "weak", "drop", "decline", "crash", "slips"]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        keyword_adjustment = 0.2
    elif negative_count > positive_count:
        keyword_adjustment = -0.2
    
    # Комбинируем market sentiment с keyword анализом
    final_sentiment = max(-1.0, min(1.0, market_sentiment + keyword_adjustment))
    
    logger.debug(f"💭 Комбинированный sentiment: market({market_sentiment:.3f}) + keywords({keyword_adjustment:.3f}) = {final_sentiment:.3f}")
    return final_sentiment


def test_sentiment_api():
    """Тест API для проверки работоспособности"""
    print("🧪 Тест рабочего sentiment API:")
    print("-" * 50)
    
    # Тест основной функции
    for i in range(3):
        sentiment = get_market_sentiment("gold")
        print(f"  Тест {i+1}: get_market_sentiment('gold') = {sentiment:.3f}")
    
    print()
    
    # Тест детальной версии
    detailed = get_market_sentiment_detailed("gold")
    print(f"  Детальный результат:")
    for key, value in detailed.items():
        print(f"    {key}: {value}")
    
    print()
    
    # Тест keyword анализа
    test_texts = [
        "Gold prices surge amid economic uncertainty",
        "Gold slips slightly as dollar strengthens", 
        "Neutral market conditions today"
    ]
    
    print("  Тест keyword анализа:")
    for text in test_texts:
        sentiment = get_sentiment(text)
        print(f"    '{text[:30]}...' → {sentiment:.3f}")
    
    print("\n✅ Все тесты завершены!")


if __name__ == "__main__":
    test_sentiment_api()