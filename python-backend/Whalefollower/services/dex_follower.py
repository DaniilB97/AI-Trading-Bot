#!/usr/bin/env python3
"""
Solana DEX Follower - Мониторинг торговцев на Solana DEX
Отслеживает успешных мем-торговцев и анализирует их стратегии
"""

import os
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import base64

from utils.logger import ServiceLogger

load_dotenv()

class DexFollower:
    def __init__(self):
        self.logger = ServiceLogger("dex_follower")
        
        # Конфигурация Solana
        self.quicknode_url = os.getenv('QUICKNODE_SOLANA_URL')
        self.solscan_api_key = os.getenv('SOLSCAN_API_KEY')  # Если есть
        
        # Файлы конфигурации
        self.wallets_file = "config/dex_wallets.json"
        self.processed_txs_file = "data/processed_dex_transactions.json"
        self.trade_analysis_file = "data/dex_trade_analysis.json"
        
        # Состояние
        self.monitored_wallets = self._load_dex_wallets()
        self.processed_txs = self._load_processed_txs()
        self.trade_stats = {}
        
        # Настройки мониторинга
        self.config = self._load_config()
        
        # Rate limiting для Solana RPC
        self.rpc_delay = 0.1
        self.last_rpc_call = 0
        
        if not self.quicknode_url:
            self.logger.warning("QuickNode URL не найден в переменных окружения")
        
        self.logger.info(f"DexFollower инициализирован. Отслеживается {len(self.monitored_wallets)} кошельков")
    
    def _load_config(self) -> Dict:
        """Загрузка конфигурации для Solana DEX мониторинга"""
        default_config = {
            "monitor_interval": 20,  # секунд
            "analyze_patterns": False,  # Заглушка для Gemini анализа
            "min_transaction_amount_sol": 0.1,
            "track_meme_coins": True,
            "profitability_threshold": 1.5,  # 150% прибыль для успешной сделки
            "max_transactions_per_scan": 50,
            "save_transaction_details": True,
            "programs_to_track": [
                "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter
                "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",  # Raydium
                "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"   # Orca
            ]
        }
        
        config_file = "config/dex_config.json"
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            os.makedirs("config", exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _load_dex_wallets(self) -> List[Dict]:
        """Загрузка кошельков DEX торговцев"""
        try:
            with open(self.wallets_file, 'r') as f:
                wallets = json.load(f)
                self.logger.info(f"Загружено {len(wallets)} DEX кошельков")
                return wallets
        except FileNotFoundError:
            # Создаем пример файла с тестовым кошельком
            example_wallets = [
                {
                    "address": "J29AYczWMaUY61cHmhdFdhZnpk5mATmqN2GRCddFnHKi",
                    "name": "Meme Trader #1",
                    "category": "meme_trader",
                    "track_since": datetime.now().isoformat(),
                    "notes": "Успешный торговец мем-коинами"
                }
            ]
            
            os.makedirs("config", exist_ok=True)
            with open(self.wallets_file, 'w') as f:
                json.dump(example_wallets, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Создан файл с примером: {self.wallets_file}")
            return example_wallets
    
    def _load_processed_txs(self) -> set:
        """Загрузка обработанных транзакций"""
        try:
            with open(self.processed_txs_file, 'r') as f:
                return set(json.load(f))
        except FileNotFoundError:
            os.makedirs("data", exist_ok=True)
            return set()
    
    def _save_processed_tx(self, tx_signature: str):
        """Сохранение обработанной транзакции"""
        self.processed_txs.add(tx_signature)
        os.makedirs("data", exist_ok=True)
        with open(self.processed_txs_file, 'w') as f:
            json.dump(list(self.processed_txs), f)
    
    def _rate_limit_rpc(self):
        """Rate limiting для Solana RPC"""
        elapsed = time.time() - self.last_rpc_call
        if elapsed < self.rpc_delay:
            time.sleep(self.rpc_delay - elapsed)
        self.last_rpc_call = time.time()
    
    async def get_wallet_transactions(self, wallet_address: str, limit: int = 50) -> List[Dict]:
        """Получение транзакций кошелька через Solana RPC"""
        if not self.quicknode_url:
            self.logger.error("QuickNode URL не настроен")
            return []
        
        self._rate_limit_rpc()
        
        try:
            # Получаем подписи транзакций
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    wallet_address,
                    {
                        "limit": limit,
                        "commitment": "confirmed"
                    }
                ]
            }
            
            response = requests.post(self.quicknode_url, json=payload, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Ошибка RPC запроса: {response.status_code}")
                return []
            
            data = response.json()
            
            if 'error' in data:
                self.logger.error(f"Ошибка Solana RPC: {data['error']}")
                return []
            
            signatures = data.get('result', [])
            self.logger.debug(f"Получено {len(signatures)} подписей для {wallet_address[:10]}...")
            
            # Получаем детали транзакций
            transactions = []
            for sig_info in signatures[:self.config['max_transactions_per_scan']]:
                if sig_info['signature'] in self.processed_txs:
                    continue
                
                tx_details = await self.get_transaction_details(sig_info['signature'])
                if tx_details:
                    transactions.append(tx_details)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Ошибка получения транзакций: {e}")
            return []
    
    async def get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Получение деталей транзакции"""
        self._rate_limit_rpc()
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {
                        "encoding": "json",
                        "commitment": "confirmed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }
            
            response = requests.post(self.quicknode_url, json=payload, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if 'error' in data or not data.get('result'):
                return None
            
            tx_data = data['result']
            
            # Парсим транзакцию
            parsed_tx = self._parse_solana_transaction(tx_data, signature)
            return parsed_tx
            
        except Exception as e:
            self.logger.error(f"Ошибка получения деталей транзакции {signature}: {e}")
            return None
    
    def _parse_solana_transaction(self, tx_data: Dict, signature: str) -> Optional[Dict]:
        """Парсинг Solana транзакции для определения DEX свопов"""
        try:
            meta = tx_data.get('meta', {})
            transaction = tx_data.get('transaction', {})
            
            # Проверяем успешность транзакции
            if meta.get('err'):
                return None
            
            # Получаем информацию о программах
            message = transaction.get('message', {})
            account_keys = message.get('accountKeys', [])
            instructions = message.get('instructions', [])
            
            # Ищем инструкции DEX программ
            dex_instruction = None
            for instruction in instructions:
                program_id_index = instruction.get('programIdIndex', 0)
                if program_id_index < len(account_keys):
                    program_id = account_keys[program_id_index]
                    if program_id in self.config['programs_to_track']:
                        dex_instruction = instruction
                        break
            
            if not dex_instruction:
                return None
            
            # Анализируем изменения балансов
            pre_balances = meta.get('preBalances', [])
            post_balances = meta.get('postBalances', [])
            
            # Основная информация о транзакции
            parsed_tx = {
                "signature": signature,
                "timestamp": tx_data.get('blockTime', int(time.time())),
                "slot": tx_data.get('slot'),
                "fee": meta.get('fee', 0),
                "status": "success" if not meta.get('err') else "failed",
                "program_id": account_keys[dex_instruction.get('programIdIndex', 0)],
                "accounts_involved": len(account_keys),
                "balance_changes": []
            }
            
            # Анализируем изменения токенов
            pre_token_balances = meta.get('preTokenBalances', [])
            post_token_balances = meta.get('postTokenBalances', [])
            
            # Определяем изменения токенов
            token_changes = self._analyze_token_changes(pre_token_balances, post_token_balances)
            parsed_tx["token_changes"] = token_changes
            
            # Определяем тип свопа
            if len(token_changes) >= 2:
                parsed_tx["swap_type"] = "token_to_token"
                parsed_tx["tokens_involved"] = [change["mint"] for change in token_changes]
            
            return parsed_tx
            
        except Exception as e:
            self.logger.error(f"Ошибка парсинга транзакции: {e}")
            return None
    
    def _analyze_token_changes(self, pre_balances: List, post_balances: List) -> List[Dict]:
        """Анализ изменений токенов в транзакции"""
        changes = []
        
        # Создаем словари для быстрого поиска
        pre_dict = {f"{b['accountIndex']}_{b.get('mint', 'SOL')}": b for b in pre_balances}
        post_dict = {f"{b['accountIndex']}_{b.get('mint', 'SOL')}": b for b in post_balances}
        
        # Находим все уникальные ключи
        all_keys = set(pre_dict.keys()) | set(post_dict.keys())
        
        for key in all_keys:
            pre_balance = pre_dict.get(key, {})
            post_balance = post_dict.get(key, {})
            
            pre_amount = float(pre_balance.get('uiTokenAmount', {}).get('uiAmount', 0))
            post_amount = float(post_balance.get('uiTokenAmount', {}).get('uiAmount', 0))
            
            if pre_amount != post_amount:
                change = {
                    "account_index": pre_balance.get('accountIndex') or post_balance.get('accountIndex'),
                    "mint": pre_balance.get('mint') or post_balance.get('mint', 'SOL'),
                    "pre_amount": pre_amount,
                    "post_amount": post_amount,
                    "change": post_amount - pre_amount
                }
                changes.append(change)
        
        return changes
    
    def _is_meme_coin(self, token_mint: str) -> bool:
        """Простая проверка на мем-коин (заглушка)"""
        # TODO: Реализовать проверку через токен листы или API
        meme_tokens = [
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT (not meme, just for testing)
        ]
        return token_mint in meme_tokens
    
    async def analyze_transaction_profitability(self, tx: Dict) -> Dict:
        """
        Анализ прибыльности транзакции
        Пока простая версия, в будущем здесь будет Gemini API
        """
        # ЗАГЛУШКА для Gemini анализа
        analysis = {
            "tx_signature": tx["signature"],
            "timestamp": tx["timestamp"],
            "profitability_score": 0.0,
            "profit_estimation": "unknown",
            "risk_level": "medium",
            "pattern_analysis": "gemini_analysis_placeholder",  # Здесь будет Gemini
            "recommendations": [],
            "analysis_method": "basic_calculation"  # Позже "gemini_ai"
        }
        
        # Простой анализ изменений токенов
        token_changes = tx.get("token_changes", [])
        
        if len(token_changes) >= 2:
            # Ищем входящие и исходящие токены
            inbound = [t for t in token_changes if t["change"] > 0]
            outbound = [t for t in token_changes if t["change"] < 0]
            
            if inbound and outbound:
                analysis["swap_detected"] = True
                analysis["tokens_in"] = len(inbound)
                analysis["tokens_out"] = len(outbound)
                
                # Простая оценка (заглушка)
                if any(self._is_meme_coin(t["mint"]) for t in inbound):
                    analysis["meme_coin_purchase"] = True
                    analysis["risk_level"] = "high"
        
        # TODO: Здесь будет вызов Gemini API для глубокого анализа
        # analysis = await self._analyze_with_gemini(tx, analysis)
        
        return analysis
    
    async def _analyze_with_gemini(self, tx_data: Dict, basic_analysis: Dict) -> Dict:
        """
        ЗАГЛУШКА для анализа через Gemini API
        В будущем здесь будет реальный анализ паттернов
        """
        # Placeholder для будущей интеграции с Gemini
        self.logger.info("🤖 Анализ через Gemini API (заглушка)")
        
        enhanced_analysis = basic_analysis.copy()
        enhanced_analysis.update({
            "gemini_patterns": "pattern_analysis_placeholder",
            "market_context": "context_analysis_placeholder", 
            "success_probability": 0.65,  # Заглушка
            "similar_trades": 42,  # Заглушка
            "analysis_method": "gemini_ai_enhanced"
        })
        
        return enhanced_analysis
    
    async def monitor_wallet(self, wallet: Dict):
        """Мониторинг одного кошелька"""
        address = wallet['address']
        name = wallet.get('name', 'Unknown')
        
        self.logger.wallet_scan_start(address, name)
        
        # Получаем транзакции
        transactions = await self.get_wallet_transactions(address)
        
        for tx in transactions:
            if tx['signature'] in self.processed_txs:
                continue
            
            self.logger.transaction_found(tx['signature'], tx.get('swap_type', 'unknown'))
            
            # Анализируем прибыльность
            if self.config['save_transaction_details']:
                analysis = await self.analyze_transaction_profitability(tx)
                
                # Сохраняем анализ
                self._save_transaction_analysis(tx, analysis)
                
                # Логгируем интересные находки
                if analysis.get('meme_coin_purchase'):
                    tokens = tx.get('tokens_involved', [])
                    self.logger.info(f"🎯 Покупка мем-коина: {tokens}")
            
            # Отмечаем как обработанную
            self._save_processed_tx(tx['signature'])
    
    def _save_transaction_analysis(self, tx: Dict, analysis: Dict):
        """Сохранение анализа транзакции"""
        try:
            # Загружаем существующие данные
            analysis_data = []
            if os.path.exists(self.trade_analysis_file):
                with open(self.trade_analysis_file, 'r') as f:
                    analysis_data = json.load(f)
            
            # Добавляем новый анализ
            combined_data = {
                "transaction": tx,
                "analysis": analysis,
                "analyzed_at": datetime.now().isoformat()
            }
            
            analysis_data.append(combined_data)
            
            # Сохраняем
            os.makedirs("data", exist_ok=True)
            with open(self.trade_analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            self.logger.data_saved(self.trade_analysis_file, len(analysis_data))
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения анализа: {e}")
    
    async def monitor_all_wallets(self):
        """Мониторинг всех кошельков"""
        while True:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"🔍 Сканирование {len(self.monitored_wallets)} Solana кошельков...")
                
                for wallet in self.monitored_wallets:
                    await self.monitor_wallet(wallet)
                
                self.logger.info(f"⏰ Ожидание {self.config['monitor_interval']} секунд...")
                await asyncio.sleep(self.config['monitor_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Мониторинг остановлен пользователем")
                break
            except Exception as e:
                self.logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(60)  # Ждем перед повтором
    
    def print_statistics(self):
        """Вывод статистики по собранным данным"""
        try:
            if os.path.exists(self.trade_analysis_file):
                with open(self.trade_analysis_file, 'r') as f:
                    data = json.load(f)
                
                print(f"\n📊 СТАТИСТИКА DEX АНАЛИЗА")
                print(f"{'='*50}")
                print(f"Всего проанализировано транзакций: {len(data)}")
                
                # Подсчет статистики
                meme_purchases = sum(1 for d in data if d['analysis'].get('meme_coin_purchase'))
                successful_trades = sum(1 for d in data if d['analysis'].get('profitability_score', 0) > 0.5)
                
                print(f"Покупки мем-коинов: {meme_purchases}")
                print(f"Потенциально успешные сделки: {successful_trades}")
                print(f"Обработано подписей: {len(self.processed_txs)}")
                
            else:
                print("📊 Пока нет данных для анализа")
                
        except Exception as e:
            self.logger.error(f"Ошибка вывода статистики: {e}")