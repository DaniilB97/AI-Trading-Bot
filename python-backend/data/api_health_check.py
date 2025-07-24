import os
import requests
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import numpy as np # Добавляем numpy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

def check_marketaux_api():
    logger.info("--- Проверка MarketAux API ---")
    api_key = os.getenv("MARKETAUX_API_TOKEN")
    if not api_key:
        logger.error("MarketAux API key не найден в .env")
        return

    test_url = "https://api.marketaux.com/v1/news/all"
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(hours=12) # Увеличил период до 12 часов для большей вероятности найти новости

    params = {
        'api_token': api_key,
        'symbols': 'SPY',  # ← ОДИН символ
        'language': 'en',
        'published_after': from_date.strftime('%Y-%m-%dT%H:%M:%S'),  # ← 1 час
        'published_before': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'limit': 2  
    }
    
    try:
        logger.info(f"Отправка запроса к MarketAux: {test_url}")
        response = requests.get(test_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', [])
            found_articles_count = data.get('meta', {}).get('found', 0)
            logger.info(f"MarketAux: ✅ Успех (Статус: {response.status_code}). Всего статей по запросу: {found_articles_count}, получено: {len(articles)}")

            if articles:
                all_entity_sentiment_scores = []
                for article in articles:
                    entities = article.get('entities', [])
                    for entity in entities:
                        score = entity.get('sentiment_score')
                        if score is not None:
                            all_entity_sentiment_scores.append(score)

                if all_entity_sentiment_scores:
                    avg_sentiment = np.mean(all_entity_sentiment_scores)
                    normalized_sentiment = (avg_sentiment + 1) / 2
                    logger.info(f"MarketAux: ✅ Извлеченный сентимент: {avg_sentiment:.3f} (нормализованный: {normalized_sentiment:.3f}) из {len(all_entity_sentiment_scores)} сущностей.")
                else:
                    logger.warning("MarketAux: Нет сущностей с sentiment scores в полученных статьях.")
            else:
                logger.warning("MarketAux: Статьи не найдены по тестовому запросу, даже после расширения параметров.")
        else:
            logger.error(f"MarketAux: ❌ Ошибка (Статус: {response.status_code}, Ответ: {response.text})")
    except requests.exceptions.Timeout:
        logger.error("MarketAux: ❌ Таймаут соединения (30 секунд).")
    except requests.exceptions.RequestException as e:
        logger.error(f"MarketAux: ❌ Ошибка запроса: {e}")
    except Exception as e:
        logger.error(f"MarketAux: ❌ Неизвестная ошибка: {e}")

def check_newsapi():
    logger.info("--- Проверка NewsAPI ---")
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.error("NewsAPI key не найден в .env")
        return

    test_url = "https://newsapi.org/v2/everything"
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(hours=12) # Увеличил период до 12 часов

    params = {
        'q': 'gold OR DXY OR VIX OR stock market', # Расширил запрос
        'language': 'en',
        'sortBy': 'relevancy',
        'from': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'to': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'pageSize': 5, # Увеличил лимит статей
        'apiKey': api_key
    }

    try:
        logger.info(f"Отправка запроса к NewsAPI: {test_url}")
        response = requests.get(test_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            total_results = data.get('totalResults', 0)
            logger.info(f"NewsAPI: ✅ Успех (Статус: {response.status_code}). Всего статей по запросу: {total_results}, получено: {len(articles)}")
            
            if articles:
                # NewsAPI не предоставляет sentiment score напрямую в бесплатных планах.
                # Сентимент из NewsAPI в твоем пайплайне, вероятно, должен
                # вычисляться с помощью отдельного NLP-модуля.
                logger.info("NewsAPI: Статьи успешно получены. Сентимент обычно рассчитывается отдельно.")
                # Если бы NewsAPI возвращал сентимент, мы бы извлекали его здесь,
                # но для NewsAPI обычно требуется свой анализатор текста.
            else:
                logger.warning("NewsAPI: Статьи не найдены по тестовому запросу, даже после расширения параметров.")
        else:
            logger.error(f"NewsAPI: ❌ Ошибка (Статус: {response.status_code}, Ответ: {response.text})")
    except requests.exceptions.Timeout:
        logger.error("NewsAPI: ❌ Таймаут соединения (30 секунд).")
    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI: ❌ Ошибка запроса: {e}")
    except Exception as e:
        logger.error(f"NewsAPI: ❌ Неизвестная ошибка: {e}")

def check_fear_greed_api():
    logger.info("--- Проверка Fear & Greed Index API (alternative.me) ---")
    test_url = "https://api.alternative.me/fng/?limit=1"
    
    try:
        logger.info(f"Отправка запроса к Fear & Greed API: {test_url}")
        response = requests.get(test_url, timeout=10) # Этот API обычно быстрее
        
        if response.status_code == 200:
            data = response.json()
            if data and data.get('data'):
                value = data['data'][0]['value']
                logger.info(f"Fear & Greed Index: ✅ Успех (Статус: {response.status_code}). Последнее значение: {value}")
            else:
                logger.warning("Fear & Greed Index: Данные не получены.")
        else:
            logger.error(f"Fear & Greed Index: ❌ Ошибка (Статус: {response.status_code}, Ответ: {response.text})")
    except requests.exceptions.Timeout:
        logger.error("Fear & Greed Index: ❌ Таймаут соединения (10 секунд).")
    except requests.exceptions.RequestException as e:
        logger.error(f"Fear & Greed Index: ❌ Ошибка запроса: {e}")
    except Exception as e:
        logger.error(f"Fear & Greed Index: ❌ Неизвестная ошибка: {e}")

if __name__ == "__main__":
    logger.info("--- Запуск проверки доступности API и извлечения сентимента ---")
    check_marketaux_api()
    print("\n")
    check_newsapi()
    print("\n")
    check_fear_greed_api()
    logger.info("--- Проверка завершена ---")