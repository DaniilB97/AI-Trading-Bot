# Комплексная система анализа и предсказания цены ETH
# Версия без TA-Lib для реальной торговли

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from typing import List, Dict, Tuple, Optional
import logging

# Telegram парсинг
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.errors import FloodWaitError

# ML и NLP
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

# Визуализация
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ БЕЗ TA-LIB
# ================================

class TechnicalIndicators:
    """Расчет технических индикаторов для реальной торговли"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range - важный индикатор волатильности"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def gap_indicator(open_price: pd.Series, close: pd.Series) -> pd.Series:
        """GAP индикатор - разрывы между закрытием и открытием"""
        gap = open_price - close.shift(1)
        gap_pct = (gap / close.shift(1)) * 100
        return gap_pct
    
    @staticmethod
    def volume_profile(volume: pd.Series, close: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Volume Profile - анализ объемов"""
        volume_sma = volume.rolling(window=period).mean()
        volume_ratio = volume / volume_sma
        
        # VWAP - Volume Weighted Average Price
        typical_price = close
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return {
            'volume_sma': volume_sma,
            'volume_ratio': volume_ratio,
            'vwap': vwap
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    @staticmethod
    def support_resistance_levels(high: pd.Series, low: pd.Series, window: int = 20) -> Dict[str, List[float]]:
        """Динамические уровни поддержки и сопротивления"""
        # Локальные минимумы и максимумы
        local_min = low.rolling(window=window, center=True).min() == low
        local_max = high.rolling(window=window, center=True).max() == high
        
        support_levels = low[local_min].dropna().unique()[-5:]  # Последние 5 уровней
        resistance_levels = high[local_max].dropna().unique()[-5:]
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels)
        }

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv_series = pd.Series(0.0, index=close.index, name="OBV")
        obv_series.iloc[0] = volume.iloc[0]
        price_diff = close.diff()
        
        for i in range(1, len(close)):
            if price_diff.iloc[i] > 0:
                obv_series.iloc[i] = obv_series.iloc[i-1] + volume.iloc[i]
            elif price_diff.iloc[i] < 0:
                obv_series.iloc[i] = obv_series.iloc[i-1] - volume.iloc[i]
            else:
                obv_series.iloc[i] = obv_series.iloc[i-1]
        return obv_series

# ================================
# 1. TELEGRAM ПАРСЕР
# ================================

class TelegramParser:
    """Универсальный парсер Telegram каналов"""
    
    def __init__(self, api_id: str, api_hash: str, phone: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        
    async def connect(self):
        """Подключение к Telegram"""
        self.client = TelegramClient('session', self.api_id, self.api_hash)
        await self.client.start(phone=self.phone)
        logger.info("✅ Подключились к Telegram")
    
    async def parse_channel(self, channel_id: str, limit: int = 1000, 
                          days_back: int = 30) -> List[Dict]:
        """Парсинг сообщений из канала"""
        messages = []
        
        try:
            # Получаем entity канала
            channel = await self.client.get_entity(channel_id)
            
            # Дата начала парсинга
            date_from = datetime.now() - timedelta(days=days_back)
            
            # Получаем историю сообщений
            async for message in self.client.iter_messages(
                channel, 
                limit=limit,
                offset_date=datetime.now(),
                reverse=False
            ):
                if message.date.replace(tzinfo=None) < date_from:
                    break
                
                if message.text:
                    messages.append({
                        'channel_id': channel_id,
                        'channel_name': channel.title,
                        'message_id': message.id,
                        'date': message.date,
                        'text': message.text,
                        'views': message.views or 0,
                        'forwards': message.forwards or 0,
                        'reactions': self._count_reactions(message),
                        'has_media': message.media is not None,
                        'reply_to': message.reply_to_msg_id
                    })
            
            logger.info(f"📥 Собрано {len(messages)} сообщений из {channel.title}")
            
        except FloodWaitError as e:
            logger.warning(f"⏳ Flood wait: {e.seconds} секунд")
            await asyncio.sleep(e.seconds)
        except Exception as e:
            logger.error(f"❌ Ошибка при парсинге {channel_id}: {e}")
        
        return messages
    
    def _count_reactions(self, message) -> int:
        """Подсчет реакций на сообщение"""
        if not message.reactions:
            return 0
        return sum(reaction.count for reaction in message.reactions.results)
    
    async def parse_multiple_channels(self, channel_ids: List[str], 
                                    **kwargs) -> pd.DataFrame:
        """Парсинг нескольких каналов"""
        all_messages = []
        
        for channel_id in channel_ids:
            messages = await self.parse_channel(channel_id, **kwargs)
            all_messages.extend(messages)
            
            # Задержка между каналами
            await asyncio.sleep(2)
        
        df = pd.DataFrame(all_messages)
        return df
    
    async def close(self):
        """Закрытие соединения"""
        if self.client:
            await self.client.disconnect()

# ================================
# 2. МОДЕЛЬ ВАЛИДАЦИИ НОВОСТЕЙ
# ================================

class NewsRelevanceClassifier:
    """Классификатор релевантности новостей для ETH"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # 3 класса: нерелевантно, нейтрально, важно
        )
        
        self.model = nn.Sequential(self.bert, self.classifier).to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        # Удаляем ссылки
        text = re.sub(r'http\S+', '', text)
        # Удаляем эмодзи
        text = re.sub(r'[^\w\s]', ' ', text)
        # Нормализация пробелов
        text = ' '.join(text.split())
        return text.lower()
    
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """Извлечение признаков из текстов"""
        features = []
        
        for text in texts:
            # Токенизация
            inputs = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Получаем эмбеддинги
            with torch.no_grad():
                outputs = self.bert(**inputs)
                # Используем CLS токен
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                features.append(cls_embedding)
        
        return torch.cat(features, dim=0)
    
    def analyze_impact(self, text: str) -> Dict:
        """Анализ потенциального влияния новости"""
        # Ключевые слова и их веса для реальной торговли
        keywords = {
            'positive': {
                'upgrade': 3, 'partnership': 3, 'adoption': 3, 'bullish': 2,
                'growth': 2, 'innovation': 2, 'launch': 2, 'success': 2,
                'institutional': 3, 'etf': 3, 'defi': 2, 'scaling': 2,
                'merge': 3, 'staking': 2, 'burn': 2
            },
            'negative': {
                'hack': -3, 'scam': -3, 'crash': -3, 'regulation': -2,
                'ban': -3, 'lawsuit': -2, 'vulnerability': -3, 'bearish': -2,
                'sell': -2, 'dump': -2, 'fraud': -3, 'investigation': -2,
                'sec': -2, 'fine': -2, 'exploit': -3
            }
        }
        
        text_lower = text.lower()
        sentiment_score = 0
        found_keywords = []
        
        # Подсчет sentiment score
        for category, words in keywords.items():
            for word, weight in words.items():
                if word in text_lower:
                    sentiment_score += weight
                    found_keywords.append((word, weight))
        
        # Определение уровня влияния
        if abs(sentiment_score) >= 5:
            impact_level = 'high'
        elif abs(sentiment_score) >= 2:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        return {
            'sentiment_score': sentiment_score,
            'impact_level': impact_level,
            'keywords': found_keywords,
            'sentiment': 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
        }

# ================================
# 3. СБОР И АНАЛИЗ ТЕХНИЧЕСКИХ ДАННЫХ
# ================================

class TechnicalAnalyzer:
    """Технический анализ ETH для внутридневной торговли"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.patterns = {}
        
    def fetch_eth_data(self, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
        """Загрузка данных ETH с часовым интервалом для внутридневной торговли"""
        eth = yf.Ticker("ETH-USD")
        df = eth.history(period=period, interval=interval)
        logger.info(f"📊 Загружено {len(df)} записей ETH с интервалом {interval}")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех технических индикаторов для реальной торговли"""
        # Базовые данные
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        volume = df['Volume']
        
        # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
        df['SMA_5'] = self.indicators.sma(close, 5)
        df['SMA_20'] = self.indicators.sma(close, 20)
        df['SMA_50'] = self.indicators.sma(close, 50)
        df['EMA_12'] = self.indicators.ema(close, 12)
        df['EMA_26'] = self.indicators.ema(close, 26)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_histogram'] = self.indicators.macd(close)
        
        # === ОСЦИЛЛЯТОРЫ ===
        df['RSI'] = self.indicators.rsi(close, 14)  # Критичный индикатор
        df['RSI_7'] = self.indicators.rsi(close, 7)  # Быстрый RSI для скальпинга
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = self.indicators.stochastic(high, low, close)
        
        # === ВОЛАТИЛЬНОСТЬ ===
        df['ATR'] = self.indicators.atr(high, low, close)  # Важно для стоп-лоссов
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.indicators.bollinger_bands(close)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # === ОБЪЕМ ===
        volume_data = self.indicators.volume_profile(volume, close)
        df['Volume_SMA'] = volume_data['volume_sma']
        df['Volume_ratio'] = volume_data['volume_ratio']
        df['VWAP'] = volume_data['vwap']
        df['OBV'] = self.indicators.obv(close, volume) # Calculate OBV
        
        # === GAP ИНДИКАТОР ===
        df['GAP'] = self.indicators.gap_indicator(open_price, close)
        df['GAP_abs'] = abs(df['GAP'])
        
        # === КАСТОМНЫЕ ИНДИКАТОРЫ ДЛЯ КРИПТО ===
        df['Price_change'] = close.pct_change()
        df['Volatility'] = df['Price_change'].rolling(20).std()
        df['High_Low_ratio'] = (high - low) / close
        
        # Momentum
        df['MOM_10'] = close - close.shift(10)
        df['ROC_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        
        # Для внутридневной торговли - микроструктура
        df['Spread'] = high - low
        df['Close_to_High'] = (high - close) / high
        df['Close_to_Low'] = (close - low) / low
        
        # Индикаторы тренда
        df['Trend_strength'] = abs(df['SMA_5'] - df['SMA_20']) / close
        df['Above_VWAP'] = (close > df['VWAP']).astype(int)
        
        logger.info(f"✅ Рассчитано {len(df.columns)} индикаторов")
        return df
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Определение паттернов для торговли"""
        patterns = {
            'support_resistance': self._find_support_resistance(df),
            'trend': self._determine_trend(df),
            'momentum': self._analyze_momentum(df),
            'volume_analysis': self._analyze_volume(df),
            'gap_analysis': self._analyze_gaps(df)
        }
        return patterns
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Поиск уровней поддержки и сопротивления"""
        levels = self.indicators.support_resistance_levels(
            df['High'], df['Low'], window=20
        )
        
        current_price = df['Close'].iloc[-1]
        
        # Ближайшие уровни
        nearest_support = None
        nearest_resistance = None
        
        for support in levels['support']:
            if support < current_price:
                nearest_support = support
        
        for resistance in levels['resistance']:
            if resistance > current_price:
                nearest_resistance = resistance
                break
        
        return {
            'support': levels['support'],
            'resistance': levels['resistance'],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': current_price
        }
    
    def _determine_trend(self, df: pd.DataFrame) -> Dict:
        """Определение текущего тренда"""
        sma_5 = df['SMA_5'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Краткосрочный тренд
        if current_price > sma_5 > sma_20:
            short_trend = 'bullish'
        elif current_price < sma_5 < sma_20:
            short_trend = 'bearish'
        else:
            short_trend = 'neutral'
        
        # Среднесрочный тренд
        if sma_20 > sma_50:
            medium_trend = 'bullish'
        elif sma_20 < sma_50:
            medium_trend = 'bearish'
        else:
            medium_trend = 'neutral'
        
        return {
            'short_term': short_trend,
            'medium_term': medium_trend,
            'strength': df['Trend_strength'].iloc[-1]
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Анализ моментума для определения точек входа"""
        rsi = df['RSI'].iloc[-1]
        rsi_7 = df['RSI_7'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]
        stoch_k = df['STOCH_K'].iloc[-1]
        
        # Сигналы перекупленности/перепроданности
        signals = []
        
        if rsi < 30:
            signals.append('oversold')
        elif rsi > 70:
            signals.append('overbought')
        
        if macd > macd_signal and df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
            signals.append('macd_bullish_crossover')
        elif macd < macd_signal and df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
            signals.append('macd_bearish_crossover')
        
        return {
            'rsi': rsi,
            'rsi_fast': rsi_7,
            'macd_histogram': df['MACD_histogram'].iloc[-1],
            'stochastic': stoch_k,
            'signals': signals
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Анализ объема для подтверждения движений"""
        volume_ratio = df['Volume_ratio'].iloc[-1]
        above_vwap = df['Above_VWAP'].iloc[-1]
        
        # Анализ объема
        if volume_ratio > 1.5:
            volume_signal = 'high_volume'
        elif volume_ratio < 0.5:
            volume_signal = 'low_volume'
        else:
            volume_signal = 'normal_volume'
        
        return {
            'volume_ratio': volume_ratio,
            'above_vwap': bool(above_vwap),
            'signal': volume_signal
        }
    
    def _analyze_gaps(self, df: pd.DataFrame) -> Dict:
        """Анализ гэпов"""
        recent_gaps = df[df['GAP_abs'] > 0.5].tail(5)
        
        gap_info = []
        for idx, row in recent_gaps.iterrows():
            gap_info.append({
                'date': idx,
                'gap_size': row['GAP'],
                'direction': 'up' if row['GAP'] > 0 else 'down'
            })
        
        return {
            'recent_gaps': gap_info,
            'last_gap': df['GAP'].iloc[-1] if not pd.isna(df['GAP'].iloc[-1]) else 0
        }

# ================================
# 4. КОМПЛЕКСНАЯ GRU МОДЕЛЬ
# ================================

class ETHPricePredictor(nn.Module):
    """GRU модель для предсказания цены ETH - оптимизирована для внутридневной торговли"""
    
    def __init__(self, technical_features: int, news_features: int, 
                 hidden_size: int = 128, num_layers: int = 3):
        super(ETHPricePredictor, self).__init__()
        
        # GRU для технических данных
        self.technical_rnn = nn.GRU(
            technical_features, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Обработка новостных данных
        self.news_encoder = nn.Sequential(
            nn.Linear(news_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=0.2
        )
        
        # Объединение и предсказание
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, technical_data, news_data):
        # RNN (GRU) для технических данных
        rnn_out, _ = self.technical_rnn(technical_data)
        
        # Attention
        attended, _ = self.attention(
            rnn_out.transpose(0, 1),
            rnn_out.transpose(0, 1),
            rnn_out.transpose(0, 1)
        )
        attended = attended.transpose(0, 1)
        
        # Берем последний временной шаг
        technical_features = attended[:, -1, :]
        
        # Обработка новостей
        news_features = self.news_encoder(news_data)
        
        # Объединение
        combined = torch.cat([technical_features, news_features], dim=1)
        
        # Предсказание
        prediction = self.fusion(combined)
        
        return prediction

class PricePredictor:
    """Обертка для обучения и использования модели"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Mac M3
        
        self.model = None
        self.technical_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        self.news_processor = None
        
    def prepare_training_data(self, df: pd.DataFrame, news_df: pd.DataFrame) -> Tuple:
        """Подготовка данных для обучения"""
        # Технические признаки для внутридневной торговли
        technical_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_20', 'EMA_12', 'RSI', 'RSI_7',
            'MACD', 'ATR', 'VWAP', 'Price_change', 'Volatility',
            'BB_width', 'Volume_ratio', 'GAP', 'MOM_10',
            'Spread', 'Close_to_High', 'Close_to_Low'
        ]
        
        # Убираем NaN
        df = df.dropna()
        
        # Нормализация технических данных
        technical_data = self.technical_scaler.fit_transform(df[technical_features])
        
        # Подготовка новостных данных
        news_features = self._prepare_news_features(news_df, df.index)
        
        # Целевая переменная
        target = self.price_scaler.fit_transform(df[['Close']])
        
        # Создание последовательностей (меньше для внутридневной торговли)
        sequence_length = 24  # 24 часа для часовых данных
        X_tech, X_news, y = [], [], []
        
        for i in range(sequence_length, len(technical_data) - 1):
            X_tech.append(technical_data[i-sequence_length:i])
            X_news.append(news_features[i])
            y.append(target[i + 1])  # Предсказываем следующий час
        
        return (
            torch.FloatTensor(np.array(X_tech)),
            torch.FloatTensor(np.array(X_news)),
            torch.FloatTensor(np.array(y))
        )
    
    def _prepare_news_features(self, news_df: pd.DataFrame, 
                              price_dates: pd.DatetimeIndex) -> np.ndarray:
        """Подготовка новостных признаков"""
        # Агрегация новостей по часам
        news_features = []
        
        for date in price_dates:
            # Для часовых данных - новости за последний час
            hour_start = date - timedelta(hours=1)
            hour_news = news_df[
                (news_df['date'] >= hour_start) & 
                (news_df['date'] < date)
            ] if not news_df.empty else pd.DataFrame()
            
            if len(hour_news) > 0:
                features = {
                    'message_count': len(hour_news),
                    'avg_sentiment': hour_news['sentiment_score'].mean() if 'sentiment_score' in hour_news else 0,
                    'max_impact': hour_news['impact_score'].max() if 'impact_score' in hour_news else 0,
                    'total_views': hour_news['views'].sum() if 'views' in hour_news else 0,
                    'total_reactions': hour_news['reactions'].sum() if 'reactions' in hour_news else 0
                }
            else:
                features = {
                    'message_count': 0,
                    'avg_sentiment': 0,
                    'max_impact': 0,
                    'total_views': 0,
                    'total_reactions': 0
                }
            
            news_features.append(list(features.values()))
        
        return np.array(news_features)
    
    def train(self, X_tech, X_news, y, epochs: int = 50, batch_size: int = 32):
        """Обучение модели"""
        # Инициализация модели
        technical_features = X_tech.shape[2]
        news_features = X_news.shape[1]
        
        self.model = ETHPricePredictor(
            technical_features=technical_features,
            news_features=news_features
        ).to(self.device)
        
        logger.info(f"✅ Используем GRU модель на {self.device}")
        
        # Подготовка данных
        X_tech = X_tech.to(self.device)
        X_news = X_news.to(self.device)
        y = y.to(self.device)
        
        # Train/Val split
        train_size = int(0.8 * len(X_tech))
        
        X_tech_train, X_tech_val = X_tech[:train_size], X_tech[train_size:]
        X_news_train, X_news_val = X_news[:train_size], X_news[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Оптимизатор и loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.HuberLoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_tech_train, X_news_train)
            train_loss = criterion(train_pred.squeeze(), y_train)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_tech_val, X_news_val)
                val_loss = criterion(val_pred.squeeze(), y_val)
            
            scheduler.step(val_loss)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, technical_data: np.ndarray, news_data: np.ndarray, 
                hours_ahead: int = 24) -> Dict:
        """Предсказание цены на несколько часов вперед"""
        self.model.eval()
        predictions = []
        
        # Преобразуем в тензоры
        tech_tensor = torch.FloatTensor(technical_data).unsqueeze(0).to(self.device)
        news_tensor = torch.FloatTensor(news_data).unsqueeze(0).to(self.device)
        
        # Предсказываем на несколько часов
        current_tech = tech_tensor.clone()
        
        for hour in range(hours_ahead):
            with torch.no_grad():
                pred = self.model(current_tech, news_tensor)
                predictions.append(pred.cpu().numpy()[0, 0])
                
                # Обновляем technical data для следующего предсказания
                current_tech = torch.cat([
                    current_tech[:, 1:, :],
                    current_tech[:, -1:, :]  # Повторяем последний
                ], dim=1)
        
        # Обратная трансформация
        predictions = self.price_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return {
            'predictions': predictions.tolist(),
            'timestamps': [(datetime.now() + timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M') 
                          for i in range(hours_ahead)]
        }

# ================================
# 5. ГЛАВНАЯ СИСТЕМА
# ================================

class ETHAnalysisSystem:
    """Главная система анализа и предсказания ETH для реальной торговли"""
    
    def __init__(self, telegram_config: Dict):
        self.telegram_parser = TelegramParser(**telegram_config)
        self.news_classifier = NewsRelevanceClassifier()
        self.technical_analyzer = TechnicalAnalyzer()
        self.price_predictor = PricePredictor()
        self.results = {}
        
    async def collect_news_data(self, channel_ids: List[str], 
                               days_back: int = 7) -> pd.DataFrame:
        """Сбор новостей из Telegram для внутридневной торговли"""
        await self.telegram_parser.connect()
        
        try:
            news_df = await self.telegram_parser.parse_multiple_channels(
                channel_ids, 
                days_back=days_back
            )
            
            # Анализ влияния каждой новости
            if not news_df.empty:
                impacts = []
                for _, row in news_df.iterrows():
                    impact = self.news_classifier.analyze_impact(row['text'])
                    impacts.append(impact)
                
                news_df['sentiment_score'] = [i['sentiment_score'] for i in impacts]
                news_df['impact_level'] = [i['impact_level'] for i in impacts]
                news_df['keywords'] = [i['keywords'] for i in impacts]
            
            logger.info(f"✅ Собрано и проанализировано {len(news_df)} новостей")
            return news_df
            
        finally:
            await self.telegram_parser.close()
    
    def analyze_technical_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Технический анализ для внутридневной торговли"""
        # Загрузка часовых данных
        df = self.technical_analyzer.fetch_eth_data(period="60d", interval="1h")
        
        # Расчет индикаторов
        df = self.technical_analyzer.calculate_indicators(df)
        
        # Определение паттернов
        patterns = self.technical_analyzer.detect_patterns(df)
        
        return df, patterns
    
    def generate_trading_signals(self, df: pd.DataFrame, patterns: Dict) -> Dict:
        """Генерация торговых сигналов для реальной торговли"""
        signals = {
            'action': 'hold',
            'confidence': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'reasons': []
        }
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Анализ сигналов
        buy_signals = 0
        sell_signals = 0
        
        # RSI сигналы
        momentum = patterns['momentum']
        if 'oversold' in momentum['signals']:
            buy_signals += 2
            signals['reasons'].append('RSI oversold (<30)')
        elif 'overbought' in momentum['signals']:
            sell_signals += 2
            signals['reasons'].append('RSI overbought (>70)')
        
        # MACD сигналы
        if 'macd_bullish_crossover' in momentum['signals']:
            buy_signals += 1
            signals['reasons'].append('MACD bullish crossover')
        elif 'macd_bearish_crossover' in momentum['signals']:
            sell_signals += 1
            signals['reasons'].append('MACD bearish crossover')
        
        # Объем
        volume = patterns['volume_analysis']
        if volume['above_vwap'] and volume['signal'] == 'high_volume':
            buy_signals += 1
            signals['reasons'].append('Price above VWAP with high volume')
        elif not volume['above_vwap'] and volume['signal'] == 'high_volume':
            sell_signals += 1
            signals['reasons'].append('Price below VWAP with high volume')
        
        # Определение действия
        if buy_signals > sell_signals and buy_signals >= 2:
            signals['action'] = 'buy'
            signals['confidence'] = min(buy_signals / 5, 1.0)
            signals['entry_price'] = current_price
            signals['stop_loss'] = current_price - (atr * 2)  # 2 ATR stop loss
            signals['take_profit'] = current_price + (atr * 3)  # 3:1 risk/reward
        elif sell_signals > buy_signals and sell_signals >= 2:
            signals['action'] = 'sell'
            signals['confidence'] = min(sell_signals / 5, 1.0)
            signals['entry_price'] = current_price
            signals['stop_loss'] = current_price + (atr * 2)
            signals['take_profit'] = current_price - (atr * 3)
        
        return signals
    
    async def run_full_analysis(self, channel_ids: List[str]) -> Dict:
        """Запуск полного анализа для торговли"""
        logger.info("🚀 Запуск анализа ETH для внутридневной торговли...")
        
        # 1. Технический анализ
        logger.info("📊 Технический анализ...")
        price_df, patterns = self.analyze_technical_data()
        
        # 2. Торговые сигналы
        trading_signals = self.generate_trading_signals(price_df, patterns)
        
        # 3. Сбор новостей (опционально)
        news_df = pd.DataFrame()
        try:
            logger.info("📰 Сбор новостей...")
            news_df = await self.collect_news_data(channel_ids, days_back=7)
        except Exception as e:
            logger.warning(f"Не удалось собрать новости: {e}")
        
        # 4. Создание отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_price': price_df['Close'].iloc[-1],
            'patterns': patterns,
            'trading_signals': trading_signals,
            'technical_summary': {
                'rsi': patterns['momentum']['rsi'],
                'atr': price_df['ATR'].iloc[-1],
                'volume_ratio': patterns['volume_analysis']['volume_ratio'],
                'trend': patterns['trend']
            }
        }
        
        self.results = {
            'report': report,
            'price_df': price_df,
            'news_df': news_df
        }
        
        logger.info("✅ Анализ завершен!")
        return self.results
