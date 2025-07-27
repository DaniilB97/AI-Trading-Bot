#!/usr/bin/env python3
"""
Paper Trading Engine - Симуляция торговли по сигналам китов и DEX трейдеров
Отслеживает сигналы и выполняет виртуальные сделки
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import requests
from dataclasses import dataclass, asdict

from utils.logger import ServiceLogger

@dataclass
class Position:
    """Виртуальная позиция"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    source_wallet: str
    source_tx: str
    source_chain: str  # 'eth' or 'solana'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def pnl(self) -> float:
        """Расчет PnL"""
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size
    
    @property
    def pnl_percentage(self) -> float:
        """PnL в процентах"""
        if self.entry_price == 0:
            return 0
        return (self.pnl / (self.entry_price * self.size)) * 100

@dataclass
class Trade:
    """Выполненная сделка"""
    id: str
    position_id: str
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    type: str  # 'open', 'close', 'stop_loss', 'take_profit'
    source_wallet: str
    source_tx: str
    pnl: Optional[float] = None

class PaperTradingEngine:
    def __init__(self):
        self.logger = ServiceLogger("paper_trading")
        
        # Конфигурация
        self.config = self._load_config()
        
        # Состояние портфеля
        self.initial_balance = self.config['portfolio']['initial_balance']
        self.current_balance = self.initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
        # Файлы данных
        self.positions_file = "data/paper_trading/positions.json"
        self.trades_file = "data/paper_trading/trades.json"
        self.portfolio_file = "data/paper_trading/portfolio.json"
        
        # Кеш цен
        self.price_cache = {}
        self.last_price_update = {}
        
        # Обработанные сигналы
        self.processed_signals = set()
        
        self._load_state()
        self.logger.info(f"Paper Trading Engine инициализирован. Баланс: ${self.current_balance}")
    
    def _load_config(self) -> Dict:
        """Загрузка конфигурации paper trading"""
        default_config = {
            "portfolio": {
                "initial_balance": 10000.0,  # $10,000 стартовый капитал
                "max_position_size": 0.1,    # 10% от баланса на позицию
                "max_total_exposure": 0.8,   # 80% максимальная экспозиция
                "risk_per_trade": 0.02       # 2% риск на сделку
            },
            "signals": {
                "max_signal_age_minutes": 5,  # Максимальный возраст сигнала
                "min_whale_trade_usd": 1000,  # Минимальная сумма сделки кита
                "copy_percentage": 0.1,       # Копируем 10% от размера сделки
                "enable_eth_signals": True,
                "enable_solana_signals": True
            },
            "risk_management": {
                "default_stop_loss": 0.05,   # 5% стоп-лосс
                "default_take_profit": 0.15, # 15% тейк-профит
                "max_daily_loss": 0.05,      # 5% максимальные дневные потери
                "max_drawdown": 0.20         # 20% максимальная просадка
            },
            "price_sources": {
                "eth_api": "https://api.coingecko.com/api/v3/simple/price",
                "solana_api": "https://api.coingecko.com/api/v3/simple/price",
                "price_update_interval": 30  # секунд
            }
        }
        
        config_file = "config/paper_trading_config.json"
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            os.makedirs("config", exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _load_state(self):
        """Загрузка состояния портфеля"""
        try:
            # Загружаем позиции
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    positions_data = json.load(f)
                    self.positions = {
                        pos_id: Position(**pos_data) 
                        for pos_id, pos_data in positions_data.items()
                    }
            
            # Загружаем сделки
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    self.trades = [Trade(**trade_data) for trade_data in trades_data]
            
            # Загружаем портфель
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                    self.current_balance = portfolio_data.get('current_balance', self.initial_balance)
                    self.processed_signals = set(portfolio_data.get('processed_signals', []))
            
            self.trade_counter = len(self.trades)
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки состояния: {e}")
    
    def _save_state(self):
        """Сохранение состояния портфеля"""
        try:
            os.makedirs("data/paper_trading", exist_ok=True)
            
            # Сохраняем позиции
            positions_data = {pos_id: asdict(pos) for pos_id, pos in self.positions.items()}
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2, default=str)
            
            # Сохраняем сделки
            trades_data = [asdict(trade) for trade in self.trades]
            with open(self.trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)
            
            # Сохраняем портфель
            portfolio_data = {
                'current_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'processed_signals': list(self.processed_signals),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния: {e}")
    
    async def get_price(self, symbol: str, chain: str = 'eth') -> Optional[float]:
        """Получение текущей цены токена"""
        cache_key = f"{chain}_{symbol}"
        current_time = time.time()
        
        # Проверяем кеш
        if (cache_key in self.price_cache and 
            current_time - self.last_price_update.get(cache_key, 0) < self.config['price_sources']['price_update_interval']):
            return self.price_cache[cache_key]
        
        try:
            if chain == 'eth':
                # Для ETH токенов через CoinGecko
                if symbol.upper() == 'ETH':
                    symbol_id = 'ethereum'
                elif symbol.upper() == 'USDT':
                    symbol_id = 'tether'
                elif symbol.upper() == 'USDC':
                    symbol_id = 'usd-coin'
                else:
                    symbol_id = symbol.lower()
                
                url = f"{self.config['price_sources']['eth_api']}?ids={symbol_id}&vs_currencies=usd"
                
            elif chain == 'solana':
                # Для Solana токенов
                if symbol.upper() == 'SOL':
                    symbol_id = 'solana'
                elif symbol.upper() == 'BONK':
                    symbol_id = 'bonk'
                else:
                    symbol_id = symbol.lower()
                
                url = f"{self.config['price_sources']['solana_api']}?ids={symbol_id}&vs_currencies=usd"
            
            else:
                return None
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = data.get(symbol_id, {}).get('usd')
                if price:
                    self.price_cache[cache_key] = float(price)
                    self.last_price_update[cache_key] = current_time
                    return float(price)
            
        except Exception as e:
            self.logger.warning(f"Ошибка получения цены для {symbol}: {e}")
        
        # Возвращаем кешированную цену если не удалось обновить
        return self.price_cache.get(cache_key)
    
    async def process_whale_signal(self, signal_data: Dict):
        """Обработка сигнала от кита (ETH)"""
        try:
            tx_hash = signal_data.get('tx_hash')
            if not tx_hash or tx_hash in self.processed_signals:
                return
            
            # Проверяем возраст сигнала
            signal_time = datetime.fromisoformat(signal_data.get('timestamp', datetime.now().isoformat()))
            age_minutes = (datetime.now() - signal_time).total_seconds() / 60
            
            if age_minutes > self.config['signals']['max_signal_age_minutes']:
                self.logger.info(f"Сигнал слишком старый: {age_minutes:.1f} мин")
                return
            
            # Извлекаем данные о сделке
            token_in = signal_data.get('token_in_symbol', 'ETH')
            token_out = signal_data.get('token_out_symbol', 'UNKNOWN')
            amount_usd = signal_data.get('amount_usd', 0)
            
            if amount_usd < self.config['signals']['min_whale_trade_usd']:
                return
            
            # Определяем направление сделки
            if token_in == 'ETH' or token_in in ['USDT', 'USDC']:
                # Покупка токена
                await self._open_position(
                    symbol=token_out,
                    side='long',
                    usd_amount=amount_usd * self.config['signals']['copy_percentage'],
                    source_wallet=signal_data.get('wallet_address', 'unknown'),
                    source_tx=tx_hash,
                    chain='eth'
                )
            elif token_out == 'ETH' or token_out in ['USDT', 'USDC']:
                # Продажа токена
                await self._close_positions_by_symbol(token_in, 'whale_sell')
            
            self.processed_signals.add(tx_hash)
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки сигнала кита: {e}")
    
    async def process_dex_signal(self, signal_data: Dict):
        """Обработка сигнала от DEX трейдера (Solana)"""
        try:
            tx_signature = signal_data.get('signature')
            if not tx_signature or tx_signature in self.processed_signals:
                return
            
            # Проверяем возраст сигнала
            signal_time = datetime.fromtimestamp(signal_data.get('timestamp', time.time()))
            age_minutes = (datetime.now() - signal_time).total_seconds() / 60
            
            if age_minutes > self.config['signals']['max_signal_age_minutes']:
                self.logger.info(f"DEX сигнал слишком старый: {age_minutes:.1f} мин")
                return
            
            # Анализируем изменения токенов
            token_changes = signal_data.get('token_changes', [])
            if len(token_changes) < 2:
                return
            
            # Определяем покупку/продажу
            for change in token_changes:
                if change.get('change', 0) > 0:  # Увеличение = покупка
                    mint = change.get('mint')
                    if mint and mint != 'SOL':
                        # Получаем примерную стоимость в USD (упрощенно)
                        sol_price = await self.get_price('SOL', 'solana')
                        if sol_price:
                            estimated_usd = abs(change.get('change', 0)) * sol_price
                            
                            await self._open_position(
                                symbol=mint[:8],  # Сокращаем длинные адреса
                                side='long',
                                usd_amount=estimated_usd * self.config['signals']['copy_percentage'],
                                source_wallet=signal_data.get('wallet_address', 'unknown'),
                                source_tx=tx_signature,
                                chain='solana'
                            )
            
            self.processed_signals.add(tx_signature)
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки DEX сигнала: {e}")
    
    async def _open_position(self, symbol: str, side: str, usd_amount: float, 
                           source_wallet: str, source_tx: str, chain: str):
        """Открытие позиции"""
        try:
            # Проверяем риск-менеджмент
            max_position_usd = self.current_balance * self.config['portfolio']['max_position_size']
            position_size = min(usd_amount, max_position_usd)
            
            if position_size < 10:  # Минимум $10
                return
            
            # Получаем цену входа
            entry_price = await self.get_price(symbol, chain)
            if not entry_price:
                self.logger.warning(f"Не удалось получить цену для {symbol}")
                return
            
            # Рассчитываем размер позиции
            size = position_size / entry_price
            
            # Создаем позицию
            position_id = f"{chain}_{symbol}_{int(time.time())}"
            position = Position(
                id=position_id,
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=datetime.now(),
                source_wallet=source_wallet,
                source_tx=source_tx,
                source_chain=chain,
                stop_loss=entry_price * (1 - self.config['risk_management']['default_stop_loss']),
                take_profit=entry_price * (1 + self.config['risk_management']['default_take_profit'])
            )
            
            # Сохраняем позицию
            self.positions[position_id] = position
            
            # Записываем сделку
            trade = Trade(
                id=f"trade_{self.trade_counter}",
                position_id=position_id,
                symbol=symbol,
                side=side,
                size=size,
                price=entry_price,
                timestamp=datetime.now(),
                type='open',
                source_wallet=source_wallet,
                source_tx=source_tx
            )
            
            self.trades.append(trade)
            self.trade_counter += 1
            
            # Обновляем баланс
            self.current_balance -= position_size
            
            self.logger.info(f"🟢 Открыта позиция: {symbol} {side} ${position_size:.2f} @ ${entry_price:.4f}")
            
        except Exception as e:
            self.logger.error(f"Ошибка открытия позиции: {e}")
    
    async def _close_positions_by_symbol(self, symbol: str, reason: str = 'manual'):
        """Закрытие всех позиций по символу"""
        positions_to_close = [pos for pos in self.positions.values() if pos.symbol == symbol]
        
        for position in positions_to_close:
            await self._close_position(position.id, reason)
    
    async def _close_position(self, position_id: str, reason: str = 'manual'):
        """Закрытие позиции"""
        try:
            if position_id not in self.positions:
                return
            
            position = self.positions[position_id]
            
            # Получаем текущую цену
            current_price = await self.get_price(position.symbol, position.source_chain)
            if not current_price:
                current_price = position.current_price
            
            # Рассчитываем PnL
            position.current_price = current_price
            pnl = position.pnl
            
            # Записываем сделку закрытия
            trade = Trade(
                id=f"trade_{self.trade_counter}",
                position_id=position_id,
                symbol=position.symbol,
                side='sell' if position.side == 'long' else 'buy',
                size=position.size,
                price=current_price,
                timestamp=datetime.now(),
                type=reason,
                source_wallet=position.source_wallet,
                source_tx=position.source_tx,
                pnl=pnl
            )
            
            self.trades.append(trade)
            self.trade_counter += 1
            
            # Обновляем баланс
            position_value = position.size * current_price
            self.current_balance += position_value
            
            # Удаляем позицию
            del self.positions[position_id]
            
            self.logger.info(f"🔴 Закрыта позиция: {position.symbol} PnL: ${pnl:.2f} ({position.pnl_percentage:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Ошибка закрытия позиции: {e}")
    
    async def update_positions(self):
        """Обновление цен и проверка стоп-лоссов/тейк-профитов"""
        for position in list(self.positions.values()):
            try:
                # Обновляем цену
                current_price = await self.get_price(position.symbol, position.source_chain)
                if current_price:
                    position.current_price = current_price
                    
                    # Проверяем стоп-лосс
                    if position.stop_loss and current_price <= position.stop_loss:
                        await self._close_position(position.id, 'stop_loss')
                        continue
                    
                    # Проверяем тейк-профит
                    if position.take_profit and current_price >= position.take_profit:
                        await self._close_position(position.id, 'take_profit')
                        continue
                
            except Exception as e:
                self.logger.error(f"Ошибка обновления позиции {position.id}: {e}")
    
    def get_portfolio_stats(self) -> Dict:
        """Получение статистики портфеля"""
        total_pnl = sum(pos.pnl for pos in self.positions.values())
        total_value = self.current_balance + sum(pos.size * pos.current_price for pos in self.positions.values())
        
        # Статистика сделок
        closed_trades = [t for t in self.trades if t.pnl is not None]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_value': total_value,
            'unrealized_pnl': total_pnl,
            'total_pnl': total_value - self.initial_balance,
            'total_pnl_percentage': ((total_value - self.initial_balance) / self.initial_balance) * 100,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0,
            'avg_win': sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        }
    
    async def monitor_signals(self):
        """Основной цикл мониторинга сигналов"""
        while True:
            try:
                # Обновляем позиции
                await self.update_positions()
                
                # Сохраняем состояние
                self._save_state()
                
                # Ждем перед следующей итерацией
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Мониторинг сигналов остановлен")
                break
            except Exception as e:
                self.logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(60)