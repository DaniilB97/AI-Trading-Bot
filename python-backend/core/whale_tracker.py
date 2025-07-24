# whale_tracker.py
# Модуль отслеживания крупных игроков (китов) для торгового бота

import os
import time
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Установи эти библиотеки: pip install aiohttp web3 python-decouple

class WhaleActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    TRANSFER = "transfer"
    UNKNOWN = "unknown"

@dataclass
class WhaleTransaction:
    hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    value_usd: float
    value_eth: float
    action_type: WhaleActionType
    token_symbol: str
    gas_fee: float
    confidence_score: float  # 0-1, насколько уверены что это торговая операция

class WhaleTracker:
    """
    Отслеживает активность кита и генерирует торговые сигналы
    """
    
    def __init__(self, whale_address: str, config: Dict = None):
        self.whale_address = whale_address.lower()
        self.config = config or {}
        
        # API ключи (нужны для получения данных)
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        self.moralis_api_key = os.getenv('MORALIS_API_KEY')
        
        # Настройки отслеживания
        self.min_transaction_usd = config.get('min_transaction_usd', 100000)  # Минимум $100k
        self.max_age_hours = config.get('max_age_hours', 24)  # Учитываем транзакции за 24 часа
        self.check_interval_minutes = config.get('check_interval_minutes', 15)  # Проверяем каждые 15 минут
        
        # Кэш транзакций
        self.transaction_cache = {}
        self.last_check_time = None
        
        # Торговые пары для анализа
        self.trading_pairs = {
            'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
            'USDC': '0xa0b86a33e6441e7c4c0aaa5c5f2c5c6e5df6c5c1',
            'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
            'WBTC': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
        }
        
        logging.info(f"🐋 Whale Tracker initialized for address: {whale_address}")
    
    async def get_whale_transactions(self, limit: int = 100) -> List[WhaleTransaction]:
        """
        Получает последние транзакции кита из различных источников
        """
        transactions = []
        
        try:
            # Используем несколько API для надежности
            if self.etherscan_api_key:
                etherscan_txs = await self._get_etherscan_transactions(limit)
                transactions.extend(etherscan_txs)
            
            if self.alchemy_api_key:
                alchemy_txs = await self._get_alchemy_transactions(limit)
                transactions.extend(alchemy_txs)
            
            if self.moralis_api_key:
                moralis_txs = await self._get_moralis_transactions(limit)
                transactions.extend(moralis_txs)
            
            # Убираем дубликаты и сортируем по времени
            unique_transactions = self._deduplicate_transactions(transactions)
            unique_transactions.sort(key=lambda x: x.timestamp, reverse=True)
            
            logging.info(f"🐋 Retrieved {len(unique_transactions)} unique whale transactions")
            return unique_transactions[:limit]
            
        except Exception as e:
            logging.error(f"Error fetching whale transactions: {e}")
            return []
    
    async def _get_etherscan_transactions(self, limit: int) -> List[WhaleTransaction]:
        """Получает транзакции через Etherscan API"""
        url = "https://api.etherscan.io/api"
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': self.whale_address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': limit,
            'sort': 'desc',
            'apikey': self.etherscan_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data['status'] == '1':
                        transactions = []
                        for tx in data['result']:
                            whale_tx = await self._parse_etherscan_transaction(tx)
                            if whale_tx:
                                transactions.append(whale_tx)
                        return transactions
                    else:
                        logging.warning(f"Etherscan API error: {data.get('message', 'Unknown error')}")
                        return []
        except Exception as e:
            logging.error(f"Etherscan API error: {e}")
            return []
    
    async def _get_alchemy_transactions(self, limit: int) -> List[WhaleTransaction]:
        """Получает транзакции через Alchemy API"""
        # Alchemy имеет более детальную информацию о DeFi транзакциях
        url = f"https://eth-mainnet.alchemyapi.io/v2/{self.alchemy_api_key}"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [{
                "fromBlock": "0x0",
                "toBlock": "latest",
                "fromAddress": self.whale_address,
                "category": ["external", "erc20", "erc721", "erc1155"],
                "maxCount": hex(limit),
                "order": "desc"
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    
                    if 'result' in data:
                        transactions = []
                        for transfer in data['result']['transfers']:
                            whale_tx = await self._parse_alchemy_transaction(transfer)
                            if whale_tx:
                                transactions.append(whale_tx)
                        return transactions
                    else:
                        logging.warning(f"Alchemy API error: {data.get('error', 'Unknown error')}")
                        return []
        except Exception as e:
            logging.error(f"Alchemy API error: {e}")
            return []
    
    async def _get_moralis_transactions(self, limit: int) -> List[WhaleTransaction]:
        """Получает транзакции через Moralis API"""
        url = f"https://deep-index.moralis.io/api/v2/{self.whale_address}"
        headers = {
            'X-API-Key': self.moralis_api_key,
            'accept': 'application/json'
        }
        params = {
            'chain': 'eth',
            'limit': limit,
            'order': 'desc'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    data = await response.json()
                    
                    if 'result' in data:
                        transactions = []
                        for tx in data['result']:
                            whale_tx = await self._parse_moralis_transaction(tx)
                            if whale_tx:
                                transactions.append(whale_tx)
                        return transactions
                    else:
                        logging.warning(f"Moralis API error: {data}")
                        return []
        except Exception as e:
            logging.error(f"Moralis API error: {e}")
            return []
    
    async def _parse_etherscan_transaction(self, tx_data: dict) -> Optional[WhaleTransaction]:
        """Парсит транзакцию из Etherscan"""
        try:
            # Получаем текущую цену ETH для конвертации
            eth_price = await self._get_eth_price()
            
            value_eth = float(tx_data['value']) / 1e18
            value_usd = value_eth * eth_price
            
            # Фильтруем мелкие транзакции
            if value_usd < self.min_transaction_usd:
                return None
            
            # Определяем тип действия
            action_type = WhaleActionType.BUY if tx_data['to'].lower() == self.whale_address else WhaleActionType.SELL
            
            return WhaleTransaction(
                hash=tx_data['hash'],
                timestamp=datetime.fromtimestamp(int(tx_data['timeStamp'])),
                from_address=tx_data['from'].lower(),
                to_address=tx_data['to'].lower(),
                value_usd=value_usd,
                value_eth=value_eth,
                action_type=action_type,
                token_symbol='ETH',
                gas_fee=float(tx_data['gasUsed']) * float(tx_data['gasPrice']) / 1e18 * eth_price,
                confidence_score=0.8  # Базовая уверенность для ETH транзакций
            )
        except Exception as e:
            logging.error(f"Error parsing Etherscan transaction: {e}")
            return None
    
    async def _parse_alchemy_transaction(self, transfer_data: dict) -> Optional[WhaleTransaction]:
        """Парсит транзакцию из Alchemy"""
        try:
            value_usd = float(transfer_data.get('value', 0))
            
            if value_usd < self.min_transaction_usd:
                return None
            
            # Определяем направление транзакции
            action_type = WhaleActionType.BUY if transfer_data['to'].lower() == self.whale_address else WhaleActionType.SELL
            
            return WhaleTransaction(
                hash=transfer_data['uniqueId'],
                timestamp=datetime.fromisoformat(transfer_data['metadata']['blockTimestamp'].replace('Z', '+00:00')),
                from_address=transfer_data['from'].lower(),
                to_address=transfer_data['to'].lower(),
                value_usd=value_usd,
                value_eth=float(transfer_data.get('value', 0)),
                action_type=action_type,
                token_symbol=transfer_data.get('asset', 'ETH'),
                gas_fee=0,  # Alchemy не всегда предоставляет gas fee
                confidence_score=0.9  # Высокая уверенность для Alchemy данных
            )
        except Exception as e:
            logging.error(f"Error parsing Alchemy transaction: {e}")
            return None
    
    async def _parse_moralis_transaction(self, tx_data: dict) -> Optional[WhaleTransaction]:
        """Парсит транзакцию из Moralis"""
        # Аналогично другим методам парсинга
        try:
            eth_price = await self._get_eth_price()
            value_eth = float(tx_data['value']) / 1e18
            value_usd = value_eth * eth_price
            
            if value_usd < self.min_transaction_usd:
                return None
            
            action_type = WhaleActionType.BUY if tx_data['to_address'].lower() == self.whale_address else WhaleActionType.SELL
            
            return WhaleTransaction(
                hash=tx_data['hash'],
                timestamp=datetime.fromisoformat(tx_data['block_timestamp'].replace('Z', '+00:00')),
                from_address=tx_data['from_address'].lower(),
                to_address=tx_data['to_address'].lower(),
                value_usd=value_usd,
                value_eth=value_eth,
                action_type=action_type,
                token_symbol='ETH',
                gas_fee=float(tx_data.get('gas_price', 0)) * float(tx_data.get('gas_used', 0)) / 1e18 * eth_price,
                confidence_score=0.85
            )
        except Exception as e:
            logging.error(f"Error parsing Moralis transaction: {e}")
            return None
    
    def _deduplicate_transactions(self, transactions: List[WhaleTransaction]) -> List[WhaleTransaction]:
        """Убирает дубликаты транзакций"""
        seen_hashes = set()
        unique_transactions = []
        
        for tx in transactions:
            if tx.hash not in seen_hashes:
                seen_hashes.add(tx.hash)
                unique_transactions.append(tx)
        
        return unique_transactions
    
    async def _get_eth_price(self) -> float:
        """Получает текущую цену ETH в USD"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    return float(data['ethereum']['usd'])
        except Exception as e:
            logging.error(f"Error fetching ETH price: {e}")
            return 2000.0  # Fallback цена
    
    def analyze_whale_sentiment(self, transactions: List[WhaleTransaction]) -> Dict:
        """
        Анализирует активность кита и генерирует торговый сигнал
        """
        if not transactions:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'No recent whale activity',
                'transactions_analyzed': 0
            }
        
        # Фильтруем транзакции по времени
        recent_transactions = [
            tx for tx in transactions
            if tx.timestamp > datetime.now() - timedelta(hours=self.max_age_hours)
        ]
        
        if not recent_transactions:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'No recent whale activity in specified timeframe',
                'transactions_analyzed': 0
            }
        
        # Анализируем паттерны
        total_buy_volume = sum(tx.value_usd for tx in recent_transactions if tx.action_type == WhaleActionType.BUY)
        total_sell_volume = sum(tx.value_usd for tx in recent_transactions if tx.action_type == WhaleActionType.SELL)
        
        buy_count = len([tx for tx in recent_transactions if tx.action_type == WhaleActionType.BUY])
        sell_count = len([tx for tx in recent_transactions if tx.action_type == WhaleActionType.SELL])
        
        # Рассчитываем сигнал
        net_volume = total_buy_volume - total_sell_volume
        volume_ratio = total_buy_volume / (total_sell_volume + 1)  # +1 чтобы избежать деления на 0
        
        # Определяем сигнал
        signal = 'NEUTRAL'
        confidence = 0.0
        reason = ''
        
        if net_volume > 500000:  # Кит накупил на $500k+ больше чем продал
            signal = 'BULLISH'
            confidence = min(0.9, net_volume / 1000000)  # Максимум 90% уверенности
            reason = f'Whale accumulated ${net_volume:,.0f} net (Buy: ${total_buy_volume:,.0f}, Sell: ${total_sell_volume:,.0f})'
        elif net_volume < -500000:  # Кит продал на $500k+ больше чем купил
            signal = 'BEARISH'
            confidence = min(0.9, abs(net_volume) / 1000000)
            reason = f'Whale distributed ${abs(net_volume):,.0f} net (Buy: ${total_buy_volume:,.0f}, Sell: ${total_sell_volume:,.0f})'
        else:
            if buy_count > sell_count * 2:  # Много мелких покупок
                signal = 'SLIGHTLY_BULLISH'
                confidence = 0.3
                reason = f'Whale making frequent small buys ({buy_count} buys vs {sell_count} sells)'
            elif sell_count > buy_count * 2:  # Много мелких продаж
                signal = 'SLIGHTLY_BEARISH'
                confidence = 0.3
                reason = f'Whale making frequent small sells ({sell_count} sells vs {buy_count} buys)'
            else:
                reason = f'Balanced whale activity (Buy: ${total_buy_volume:,.0f}, Sell: ${total_sell_volume:,.0f})'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'transactions_analyzed': len(recent_transactions),
            'total_buy_volume': total_buy_volume,
            'total_sell_volume': total_sell_volume,
            'net_volume': net_volume,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'whale_address': self.whale_address,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def get_whale_signal(self) -> Dict:
        """
        Основной метод для получения сигнала от кита
        """
        try:
            logging.info(f"🐋 Analyzing whale activity for {self.whale_address}")
            
            # Получаем транзакции
            transactions = await self.get_whale_transactions(limit=50)
            
            # Анализируем сигнал
            analysis = self.analyze_whale_sentiment(transactions)
            
            # Логируем результат
            logging.info(f"🐋 Whale signal: {analysis['signal']} (confidence: {analysis['confidence']:.2f})")
            logging.info(f"🐋 Reason: {analysis['reason']}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error getting whale signal: {e}")
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'reason': f'Error analyzing whale: {str(e)}',
                'error': str(e)
            }

