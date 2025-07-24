# real_news_api.py
"""
Real News API Client for sentiment analysis
Supports multiple news sources: NewsAPI.org, MarketAux
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class RealNewsAPIClient:
    """
    Real news API client that fetches actual news from multiple sources
    """
    
    def __init__(self, news_api_key: str = None, marketaux_api_key: str = None):
        self.news_api_key = news_api_key
        self.marketaux_api_key = marketaux_api_key
        
        # API endpoints
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.marketaux_url = "https://api.marketaux.com/v1/news/all"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 second between requests
        
        logger.info(f"Initialized with NewsAPI: {'✅' if news_api_key else '❌'}, MarketAux: {'✅' if marketaux_api_key else '❌'}")

    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def get_newsapi_articles(self, query: str = "gold", language: str = "en") -> List[Dict[str, Any]]:
        """
        Fetch articles from NewsAPI.org
        """
        if not self.news_api_key:
            return []
        
        self._rate_limit()
        
        try:
            # Get articles from the last 24 hours
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            params = {
                'q': f'{query} OR "gold price" OR "gold market" OR "precious metals"',
                'language': language,
                'sortBy': 'publishedAt',
                'from': from_date,
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                logger.info(f"📰 NewsAPI: Retrieved {len(articles)} articles")
                return articles
            else:
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request error: {e}")
            return []
        except Exception as e:
            logger.error(f"NewsAPI unexpected error: {e}")
            return []

    def get_marketaux_articles(self, query: str = "gold") -> List[Dict[str, Any]]:
        """
        Fetch articles from MarketAux API (financial news focused)
        MarketAux provides ready-made sentiment scores!
        """
        if not self.marketaux_api_key:
            return []
        
        self._rate_limit()
        
        try:
            # Use the working configuration from debug script
            params = {
                'search': query,
                'language': 'en',
                'limit': 15,
                'api_token': self.marketaux_api_key
            }
            
            logger.info(f"MarketAux request with search term: {query}")
            
            response = requests.get(self.marketaux_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data:
                articles = data['data']
                logger.info(f"📊 MarketAux: Retrieved {len(articles)} articles with sentiment data")
                
                # Convert to standard format and extract MarketAux sentiment
                standardized_articles = []
                for article in articles:
                    # Extract MarketAux sentiment data
                    sentiment_data = article.get('sentiment', {})
                    
                    # MarketAux provides sentiment as positive/negative/neutral scores
                    marketaux_sentiment = self._extract_marketaux_sentiment(sentiment_data)
                    
                    standardized_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('snippet', ''),
                        'publishedAt': article.get('published_at', ''),
                        'source': {'name': article.get('source', 'MarketAux')},
                        'url': article.get('url', ''),
                        'entities': article.get('entities', []),
                        # Add MarketAux-specific fields
                        'marketaux_sentiment': marketaux_sentiment,
                        'has_ready_sentiment': True if sentiment_data else False,
                        'sentiment_confidence': self._get_sentiment_confidence(sentiment_data),
                        'raw_sentiment_data': sentiment_data  # Keep for debugging
                    }
                    
                    standardized_articles.append(standardized_article)
                
                return standardized_articles
            else:
                logger.error(f"MarketAux unexpected response structure: {data}")
                return []
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"MarketAux HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"MarketAux request error: {e}")
            return []
        except Exception as e:
            logger.error(f"MarketAux unexpected error: {e}")
            return []

    def _get_sentiment_confidence(self, sentiment_data: Dict) -> float:
        """
        Calculate confidence score from sentiment data
        """
        if not sentiment_data:
            return 0.0
        
        try:
            # If MarketAux provides confidence directly
            if 'confidence' in sentiment_data:
                return float(sentiment_data['confidence'])
            
            # Calculate confidence based on sentiment strength
            positive = float(sentiment_data.get('positive', 0))
            negative = float(sentiment_data.get('negative', 0))
            neutral = float(sentiment_data.get('neutral', 0))
            
            # Higher confidence when sentiment is more decisive
            max_sentiment = max(positive, negative, neutral)
            return max_sentiment
            
        except (ValueError, TypeError):
            return 0.5  # Default confidence

    def _extract_marketaux_sentiment(self, sentiment_data: Dict) -> float:
        """
        Extract sentiment score from MarketAux sentiment data
        MarketAux provides sentiment as: {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
        Convert to our -1 to +1 scale
        """
        if not sentiment_data:
            return 0.0
        
        try:
            positive = float(sentiment_data.get('positive', 0))
            negative = float(sentiment_data.get('negative', 0))
            neutral = float(sentiment_data.get('neutral', 0))
            
            # Calculate net sentiment: positive - negative
            # Neutral acts as a dampening factor
            net_sentiment = positive - negative
            
            # Apply neutral dampening (high neutral reduces absolute sentiment)
            dampening_factor = 1.0 - (neutral * 0.5)  # Max 50% dampening
            
            final_sentiment = net_sentiment * dampening_factor
            
            # Ensure it's in [-1, 1] range
            return max(-1.0, min(1.0, final_sentiment))
            
        except (ValueError, TypeError):
            logger.warning("Could not parse MarketAux sentiment data")
            return 0.0

    def get_news_with_sentiment(self, query: str = "gold", language: str = "en") -> tuple[List[Dict[str, Any]], float]:
        """
        Get news and calculate overall sentiment, prioritizing MarketAux ready-made sentiments
        Returns: (articles_list, overall_sentiment_score)
        """
        all_articles = []
        
        # Try MarketAux first (has ready sentiment)
        marketaux_articles = []
        if self.marketaux_api_key:
            marketaux_articles = self.get_marketaux_articles(query)
            all_articles.extend(marketaux_articles)
        
        # Try NewsAPI (need to analyze sentiment)
        newsapi_articles = []
        if self.news_api_key:
            newsapi_articles = self.get_newsapi_articles(query, language)
            # Mark these as needing sentiment analysis
            for article in newsapi_articles:
                article['has_ready_sentiment'] = False
            all_articles.extend(newsapi_articles)
        
        # Remove duplicates
        unique_articles = self._remove_duplicate_articles(all_articles)
        
        # Sort by publication date (newest first)
        try:
            unique_articles.sort(
                key=lambda x: datetime.fromisoformat(x.get('publishedAt', '').replace('Z', '+00:00')), 
                reverse=True
            )
        except:
            pass
        
        # Calculate overall sentiment with priority system
        overall_sentiment = self._calculate_overall_sentiment(unique_articles)
        
        logger.info(f"📊 Total articles: {len(unique_articles)} | Overall sentiment: {overall_sentiment:.3f}")
        logger.info(f"   📈 MarketAux (ready sentiment): {len(marketaux_articles)}")
        logger.info(f"   📰 NewsAPI (analyzed sentiment): {len(newsapi_articles)}")
        
        return unique_articles, overall_sentiment

    def _calculate_overall_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """
        Calculate overall sentiment from articles, prioritizing MarketAux ready sentiments
        """
        if not articles:
            return 0.0
        
        sentiments = []
        weights = []
        
        for i, article in enumerate(articles[:15]):  # Limit to 15 most recent articles
            sentiment_score = 0.0
            confidence = 0.5
            
            if article.get('has_ready_sentiment', False):
                # Use MarketAux ready sentiment (higher priority)
                sentiment_score = article.get('marketaux_sentiment', 0.0)
                confidence = article.get('sentiment_confidence', 0.8)  # Higher confidence for ready sentiment
                logger.debug(f"MarketAux sentiment: {sentiment_score:.3f} (confidence: {confidence:.2f})")
            else:
                # Analyze sentiment for NewsAPI articles
                title = article.get('title', '')
                description = article.get('description', '')
                text_to_analyze = f"{title} {description}".strip()
                
                if text_to_analyze:
                    sentiment_score = get_advanced_sentiment(text_to_analyze)
                    confidence = 0.6  # Lower confidence for analyzed sentiment
                    logger.debug(f"Analyzed sentiment: {sentiment_score:.3f} (confidence: {confidence:.2f})")
            
            if sentiment_score != 0.0:  # Only include non-neutral sentiments
                # Weight calculation: newer articles + confidence + ready sentiment bonus
                time_weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, etc.
                confidence_weight = confidence
                ready_sentiment_bonus = 1.3 if article.get('has_ready_sentiment', False) else 1.0
                
                final_weight = time_weight * confidence_weight * ready_sentiment_bonus
                
                sentiments.append(sentiment_score)
                weights.append(final_weight)
        
        if not sentiments:
            return 0.0
        
        # Calculate weighted average
        weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def _remove_duplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Remove duplicate articles based on title similarity
        """
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            if not title:
                continue
                
            # Simple deduplication - check if we've seen very similar title
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(title)
        
        return unique_articles

    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """
        Check if two titles are similar (simple word overlap check)
        """
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = overlap / union if union > 0 else 0
        return similarity >= threshold

# Advanced sentiment analysis using VADER
def get_advanced_sentiment(text: str) -> float:
    """
    Advanced sentiment analysis using VADER (if available) or fallback to simple method
    Returns sentiment score from -1 (very negative) to +1 (very positive)
    """
    if not text:
        return 0.0
    
    try:
        # Try to use VADER sentiment analyzer
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Add financial keywords to VADER lexicon
        financial_words = {
            'bullish': 3.0, 'bearish': -3.0, 'rally': 2.5, 'surge': 2.5,
            'plunge': -2.5, 'crash': -3.0, 'soar': 2.5, 'tumble': -2.5,
            'uncertainty': -1.5, 'volatility': -1.0, 'safe-haven': 2.0,
            'hedge': 1.5, 'inflation': -1.0, 'recession': -2.5
        }
        
        for word, score in financial_words.items():
            analyzer.lexicon[word] = score
        
        scores = analyzer.polarity_scores(text)
        # Use compound score which normalizes to [-1, 1]
        return scores['compound']
        
    except ImportError:
        # Fallback to simple sentiment analysis
        logger.warning("VADER not available, using simple sentiment analysis")
        return get_simple_sentiment(text)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return get_simple_sentiment(text)

def get_simple_sentiment(text: str) -> float:
    """
    Simple sentiment analysis fallback
    """
    if not text:
        return 0.0
    
    text = text.lower()
    
    # Positive financial keywords
    positive_words = [
        'surge', 'rally', 'bullish', 'rise', 'gain', 'strong', 'up', 'higher',
        'safe-haven', 'hedge', 'positive', 'optimistic', 'growth', 'increase',
        'soar', 'climb', 'advance', 'boost', 'strengthen'
    ]
    
    # Negative financial keywords  
    negative_words = [
        'fall', 'drop', 'plunge', 'bearish', 'decline', 'weak', 'down', 'lower',
        'crash', 'tumble', 'negative', 'pessimistic', 'uncertainty', 'risk',
        'volatile', 'unstable', 'concern', 'worry', 'fear'
    ]
    
    # Neutral words that often appear with important context
    context_words = [
        'market', 'price', 'trading', 'investment', 'analyst', 'forecast'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    context_count = sum(1 for word in context_words if word in text)
    
    # Calculate sentiment with context weighting
    if positive_count == 0 and negative_count == 0:
        return 0.1 if context_count > 0 else 0.0
    
    # Normalize to [-1, 1] range
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words == 0:
        return 0.0
    
    raw_sentiment = (positive_count - negative_count) / total_sentiment_words
    
    # Apply context boost
    context_multiplier = min(1.5, 1.0 + (context_count * 0.1))
    
    return max(-1.0, min(1.0, raw_sentiment * context_multiplier))