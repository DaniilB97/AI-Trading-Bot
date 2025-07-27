#!/usr/bin/env python3
"""
DEX Data Collector - Сбор исторических данных по Solana кошелькам
Собирает полную историю транзакций для анализа паттернов торговли
"""

import os
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from utils.logger import ServiceLogger

load_dotenv()

class DexDataCollector:
    def __init__(self):
        self.logger = ServiceLogger("dex_data_collector")
        
        # Конфигурация
        self.quicknode_url = os.getenv('QUICKNODE_SOLANA_URL')
        self.solscan_api_key = os.getenv('SOLSCAN_API_KEY')
        
        # Файлы данных
        self.wallets_file = "config/dex_wallets.json"
        self.historical_data_dir = "data/historical"
        self.collection_status_file = "data/collection_status.json"
        
        # Настройки сбора
        self.config = self._load_config()
        
        # Rate limiting
        self.rpc_delay = 0.2  # Медленнее для исторических данных
        self.last_rpc_call = 0
        
        # Статус сбора
        self.collection_status = self._load_collection_status()
        
        self.logger.info("DexDataCollector инициализирован")
    
    def _load_config(self) -> Dict:
        """Конфигурация для сбора исторических данных"""
        default_config = {
            "collection_settings": {
                "max_transactions_per_wallet": 1000,  # Максимум транзакций на кошелек
                "days_back": 30,  # Сколько дней назад собирать
                "batch_size": 100,  # Транзакций за один запрос
                "delay_between_batches": 1.0,  # Секунд между батчами
                "retry_failed": True,
                "skip_errors": True
            },
            "filtering": {
                "min_amount_sol": 0.01,  # Минимальная сумма транзакции
                "include_failed_txs": False,
                "dex_programs_only": True,
                "meme_coins_priority": True
            },
            "analysis": {
                "calculate_pnl": True,  # Рассчитывать прибыль/убыток
                "track_token_prices": True,
                "identify_patterns": True,
                "save_detailed_logs": True
            }
        }
        
        config_file = "config/data_collection_config.json"
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            os.makedirs("config", exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _load_collection_status(self) -> Dict:
        """Загрузка статуса сбора данных"""
        try:
            with open(self.collection_status_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_collection_status(self):
        """Сохранение статуса сбора"""
        os.makedirs("data", exist_ok=True)
        with open(self.collection_status_file, 'w') as f:
            json.dump(self.collection_status, f, indent=2, default=str)
    
    def _rate_limit_rpc(self):
        """Rate limiting для RPC запросов"""
        elapsed = time.time() - self.last_rpc_call
        if elapsed < self.rpc_delay:
            time.sleep(self.rpc_delay - elapsed)
        self.last_rpc_call = time.time()
    
    async def collect_wallet_history(self, wallet: Dict) -> Dict:
        """Сбор полной истории транзакций кошелька"""
        address = wallet['address']
        name = wallet.get('name', 'Unknown')
        
        self.logger.info(f"🔍 Начинаем сбор истории для: {name} ({address[:10]}...)")
        
        # Проверяем статус предыдущего сбора
        if address in self.collection_status:
            last_collection = self.collection_status[address].get('last_collection_date')
            if last_collection:
                self.logger.info(f"Последний сбор: {last_collection}")
        
        # Создаем папку для кошелька
        wallet_dir = os.path.join(self.historical_data_dir, address)
        os.makedirs(wallet_dir, exist_ok=True)
        
        # Начинаем сбор
        collection_start = datetime.now()
        total_transactions = 0
        successful_transactions = 0
        errors = 0
        
        try:
            # Получаем все подписи транзакций
            all_signatures = await self._get_all_signatures(address)
            total_transactions = len(all_signatures)
            
            self.logger.info(f"📝 Найдено {total_transactions} транзакций")
            
            # Обрабатываем батчами
            batch_size = self.config['collection_settings']['batch_size']
            batches = [all_signatures[i:i + batch_size] for i in range(0, len(all_signatures), batch_size)]
            
            processed_transactions = []
            
            for i, batch in enumerate(batches):
                self.logger.info(f"📦 Обработка батча {i+1}/{len(batches)} ({len(batch)} транзакций)")
                
                batch_results = await self._process_transaction_batch(batch, address)
                processed_transactions.extend(batch_results)
                successful_transactions += len(batch_results)
                
                # Задержка между батчами
                if i < len(batches) - 1:
                    await asyncio.sleep(self.config['collection_settings']['delay_between_batches'])
            
            # Анализируем собранные данные
            analysis_results = await self._analyze_wallet_history(processed_transactions, wallet)
            
            # Сохраняем данные
            await self._save_wallet_data(address, processed_transactions, analysis_results)
            
            # Обновляем статус
            collection_end = datetime.now()
            self.collection_status[address] = {
                'last_collection_date': collection_end.isoformat(),
                'total_transactions': total_transactions,
                'successful_transactions': successful_transactions,
                'errors': errors,
                'collection_duration_minutes': (collection_end - collection_start).total_seconds() / 60,
                'analysis_completed': True
            }
            
            self._save_collection_status()
            
            self.logger.info(f"✅ Сбор завершен: {successful_transactions}/{total_transactions} транзакций")
            
            return {
                'status': 'success',
                'wallet': address,
                'transactions_collected': successful_transactions,
                'analysis': analysis_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сбора данных для {address}: {e}")
            return {
                'status': 'error',
                'wallet': address,
                'error': str(e)
            }
    
    async def _get_all_signatures(self, address: str) -> List[str]:
        """Получение всех подписей транзакций кошелька"""
        all_signatures = []
        before = None  # Для пагинации
        max_transactions = self.config['collection_settings']['max_transactions_per_wallet']
        
        while len(all_signatures) < max_transactions:
            self._rate_limit_rpc()
            
            try:
                params = [
                    address,
                    {
                        "limit": 1000,  # Максимум за запрос
                        "commitment": "confirmed"
                    }
                ]
                
                if before:
                    params[1]["before"] = before
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": params
                }
                
                response = requests.post(self.quicknode_url, json=payload, timeout=15)
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                
                if 'error' in data or not data.get('result'):
                    break
                
                signatures = data['result']
                
                if not signatures:
                    break
                
                # Добавляем подписи
                signature_list = [sig['signature'] for sig in signatures]
                all_signatures.extend(signature_list)
                
                # Устанавливаем курсор для следующей страницы
                before = signatures[-1]['signature']
                
                self.logger.debug(f"Собрано {len(all_signatures)} подписей...")
                
                # Если получили меньше 1000, значит достигли конца
                if len(signatures) < 1000:
                    break
                    
            except Exception as e:
                self.logger.error(f"Ошибка получения подписей: {e}")
                break
        
        return all_signatures[:max_transactions]
    
    async def _process_transaction_batch(self, signatures: List[str], wallet_address: str) -> List[Dict]:
        """Обработка батча транзакций"""
        processed = []
        errors = 0
        
        for i, signature in enumerate(signatures):
            try:
                self._rate_limit_rpc()
                
                # Получаем детали транзакции
                tx_details = await self._get_transaction_details(signature)
                
                if tx_details:
                    # Фильтруем по настройкам
                    if self._should_include_transaction(tx_details):
                        # Добавляем метаданные
                        tx_details['wallet_address'] = wallet_address
                        tx_details['collected_at'] = datetime.now().isoformat()
                        
                        processed.append(tx_details)
                        
                        # Лог каждые 10 транзакций
                        if (i + 1) % 10 == 0:
                            self.logger.debug(f"Обработано {i + 1}/{len(signatures)} транзакций в батче")
                
            except Exception as e:
                errors += 1
                if not self.config['collection_settings']['skip_errors']:
                    raise
                self.logger.warning(f"Пропуск транзакции {signature[:20]}...: {e}")
                
                # Если слишком много ошибок, прерываем
                if errors > len(signatures) * 0.5:  # Больше 50% ошибок
                    self.logger.error(f"Слишком много ошибок в батче ({errors}/{len(signatures)}), прерываем")
                    break
        
        if errors > 0:
            self.logger.warning(f"Батч завершен с {errors} ошибками")
        
        return processed
    
    async def _get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Получение детальной информации о транзакции"""
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
        
        response = requests.post(self.quicknode_url, json=payload, timeout=15)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if 'error' in data or not data.get('result'):
            return None
        
        # Парсим транзакцию
        return self._parse_transaction_details(data['result'], signature)
    
    def _parse_transaction_details(self, tx_data: Dict, signature: str) -> Optional[Dict]:
        """Детальный парсинг транзакции"""
        try:
            meta = tx_data.get('meta', {})
            transaction = tx_data.get('transaction', {})
            
            # Базовая информация
            parsed = {
                'signature': signature,
                'slot': tx_data.get('slot'),
                'blockTime': tx_data.get('blockTime'),
                'fee': meta.get('fee', 0),
                'status': 'success' if not meta.get('err') else 'failed',
                'error': meta.get('err') if meta.get('err') else None
            }
            
            # Анализ программ
            message = transaction.get('message', {})
            account_keys = message.get('accountKeys', [])
            instructions = message.get('instructions', [])
            
            # Определяем DEX программы
            dex_programs = []
            for instruction in instructions:
                try:
                    program_idx = instruction.get('programIdIndex', 0)
                    if program_idx < len(account_keys):
                        program_id = account_keys[program_idx]
                        if self._is_dex_program(program_id):
                            dex_programs.append(program_id)
                except (KeyError, IndexError, TypeError):
                    continue
            
            parsed['dex_programs'] = dex_programs
            parsed['is_dex_transaction'] = len(dex_programs) > 0
            
            # Анализ токенов (с защитой от ошибок)
            try:
                pre_token_balances = meta.get('preTokenBalances', [])
                post_token_balances = meta.get('postTokenBalances', [])
                
                token_changes = self._calculate_token_changes(pre_token_balances, post_token_balances)
                parsed['token_changes'] = token_changes
                
                # Определяем тип свапа
                if len(token_changes) >= 2:
                    parsed['swap_type'] = self._determine_swap_type(token_changes)
                    parsed['tokens_involved'] = list(set(change['mint'] for change in token_changes if change.get('mint')))
                
            except Exception as e:
                self.logger.warning(f"Ошибка анализа токенов в транзакции {signature}: {e}")
                parsed['token_changes'] = []
                parsed['tokens_involved'] = []
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Ошибка парсинга транзакции {signature}: {e}")
            return None
    
    def _should_include_transaction(self, tx: Dict) -> bool:
        """Фильтрация транзакций по настройкам"""
        # Пропускаем неуспешные если настроено
        if not self.config['filtering']['include_failed_txs'] and tx['status'] != 'success':
            return False
        
        # Только DEX транзакции если настроено
        if self.config['filtering']['dex_programs_only'] and not tx.get('is_dex_transaction'):
            return False
        
        # Минимальная сумма
        # TODO: Реализовать проверку суммы через token_changes
        
        return True
    
    def _is_dex_program(self, program_id: str) -> bool:
        """Проверка является ли программа DEX"""
        dex_programs = [
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter
            "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",  # Raydium
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Orca
            "22Y43yTVxuUkoRKdm9thyRhQ3SdgQS7c7kB6UNCiaczD",  # Serum
        ]
        return program_id in dex_programs
    
    def _calculate_token_changes(self, pre_balances: List, post_balances: List) -> List[Dict]:
        """Расчет изменений токенов"""
        changes = []
        
        try:
            # Создаем индексы для быстрого поиска
            pre_dict = {}
            post_dict = {}
            
            for balance in pre_balances:
                key = f"{balance['accountIndex']}_{balance.get('mint', 'SOL')}"
                pre_dict[key] = balance
            
            for balance in post_balances:
                key = f"{balance['accountIndex']}_{balance.get('mint', 'SOL')}"
                post_dict[key] = balance
            
            # Находим изменения
            all_keys = set(pre_dict.keys()) | set(post_dict.keys())
            
            for key in all_keys:
                try:
                    pre = pre_dict.get(key)
                    post = post_dict.get(key)
                    
                    pre_amount = 0.0
                    post_amount = 0.0
                    mint = None
                    account_index = None
                    
                    if pre:
                        ui_amount = pre.get('uiTokenAmount', {}).get('uiAmount')
                        pre_amount = float(ui_amount) if ui_amount is not None else 0.0
                        mint = pre.get('mint')
                        account_index = pre.get('accountIndex')
                    
                    if post:
                        ui_amount = post.get('uiTokenAmount', {}).get('uiAmount')
                        post_amount = float(ui_amount) if ui_amount is not None else 0.0
                        mint = mint or post.get('mint')
                        account_index = account_index or post.get('accountIndex')
                    
                    # Пропускаем если нет значимых изменений
                    if abs(pre_amount - post_amount) < 1e-9:
                        continue
                    
                    changes.append({
                        'account_index': account_index,
                        'mint': mint,
                        'pre_amount': pre_amount,
                        'post_amount': post_amount,
                        'change': post_amount - pre_amount,
                        'is_increase': post_amount > pre_amount
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки изменения токена {key}: {e}")
                    continue
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета изменений токенов: {e}")
            return []
    
    def _determine_swap_type(self, token_changes: List[Dict]) -> str:
        """Определение типа свапа"""
        increases = [c for c in token_changes if c['is_increase']]
        decreases = [c for c in token_changes if not c['is_increase']]
        
        if len(increases) == 1 and len(decreases) == 1:
            return "simple_swap"
        elif len(increases) > 1 or len(decreases) > 1:
            return "complex_swap"
        else:
            return "unknown"
    
    async def _analyze_wallet_history(self, transactions: List[Dict], wallet: Dict) -> Dict:
        """Анализ истории торговли кошелька"""
        analysis = {
            'wallet_address': wallet['address'],
            'wallet_name': wallet.get('name', 'Unknown'),
            'analysis_date': datetime.now().isoformat(),
            'total_transactions': len(transactions),
            'dex_transactions': len([tx for tx in transactions if tx.get('is_dex_transaction')]),
            'successful_transactions': len([tx for tx in transactions if tx['status'] == 'success']),
            'failed_transactions': len([tx for tx in transactions if tx['status'] != 'success']),
            'time_range': {
                'earliest': None,
                'latest': None
            },
            'trading_patterns': {},
            'tokens_traded': [],
            'profitability_analysis': {}
        }
        
        if transactions:
            # Временной диапазон
            timestamps = [tx['blockTime'] for tx in transactions if tx.get('blockTime')]
            if timestamps:
                analysis['time_range']['earliest'] = min(timestamps)
                analysis['time_range']['latest'] = max(timestamps)
            
            # Анализ токенов
            all_tokens = set()
            for tx in transactions:
                tokens = tx.get('tokens_involved', [])
                all_tokens.update(tokens)
            
            analysis['tokens_traded'] = list(all_tokens)
            analysis['unique_tokens_count'] = len(all_tokens)
            
            # Простые паттерны
            swap_types = [tx.get('swap_type') for tx in transactions if tx.get('swap_type')]
            analysis['trading_patterns'] = {
                'swap_types': list(set(swap_types)),
                'most_common_swap_type': max(set(swap_types), key=swap_types.count) if swap_types else None,
                'avg_transactions_per_day': self._calculate_avg_transactions_per_day(transactions)
            }
            
            # TODO: Здесь будет более сложный анализ прибыльности
            # analysis['profitability_analysis'] = await self._calculate_profitability(transactions)
        
        return analysis
    
    def _calculate_avg_transactions_per_day(self, transactions: List[Dict]) -> float:
        """Расчет среднего количества транзакций в день"""
        timestamps = [tx['blockTime'] for tx in transactions if tx.get('blockTime')]
        
        if len(timestamps) < 2:
            return 0
        
        min_time = min(timestamps)
        max_time = max(timestamps)
        days = (max_time - min_time) / (24 * 60 * 60)  # Секунды в дни
        
        return len(transactions) / max(days, 1)
    
    async def _save_wallet_data(self, address: str, transactions: List[Dict], analysis: Dict):
        """Сохранение данных кошелька"""
        wallet_dir = os.path.join(self.historical_data_dir, address)
        os.makedirs(wallet_dir, exist_ok=True)
        
        # Сохраняем транзакции
        transactions_file = os.path.join(wallet_dir, "transactions.json")
        with open(transactions_file, 'w') as f:
            json.dump(transactions, f, indent=2, default=str)
        
        # Сохраняем анализ
        analysis_file = os.path.join(wallet_dir, "analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Сохраняем краткую сводку
        summary = {
            'wallet_address': address,
            'collection_date': datetime.now().isoformat(),
            'total_transactions': len(transactions),
            'dex_transactions': analysis.get('dex_transactions', 0),
            'unique_tokens': analysis.get('unique_tokens_count', 0),
            'time_range': analysis.get('time_range', {}),
            'file_paths': {
                'transactions': transactions_file,
                'analysis': analysis_file
            }
        }
        
        summary_file = os.path.join(wallet_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.data_saved(wallet_dir, len(transactions))
    
    async def collect_all_wallets_history(self):
        """Сбор истории для всех кошельков"""
        try:
            with open(self.wallets_file, 'r') as f:
                wallets = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Файл кошельков не найден: {self.wallets_file}")
            return
        
        self.logger.info(f"📊 Начинаем сбор истории для {len(wallets)} кошельков")
        
        results = []
        start_time = datetime.now()
        
        for i, wallet in enumerate(wallets, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Кошелек {i}/{len(wallets)}: {wallet.get('name', 'Unknown')}")
            
            wallet_start = datetime.now()
            result = await self.collect_wallet_history(wallet)
            wallet_duration = (datetime.now() - wallet_start).total_seconds() / 60
            
            result['duration_minutes'] = wallet_duration
            results.append(result)
            
            # Промежуточная статистика
            successful = len([r for r in results if r['status'] == 'success'])
            total_txs = sum(r.get('transactions_collected', 0) for r in results)
            
            self.logger.info(f"⏱️  Кошелек обработан за {wallet_duration:.1f} мин")
            self.logger.info(f"📊 Прогресс: {successful}/{i} успешно, {total_txs} транзакций")
            
            # Пауза между кошельками (кроме последнего)
            if i < len(wallets):
                pause_time = 5
                self.logger.info(f"⏳ Пауза {pause_time}с между кошельками...")
                await asyncio.sleep(pause_time)
        
        # Финальная статистика
        total_duration = (datetime.now() - start_time).total_seconds() / 60
        successful = len([r for r in results if r['status'] == 'success'])
        total_transactions = sum(r.get('transactions_collected', 0) for r in results)
        
        self.logger.info(f"\n🎉 СБОР ЗАВЕРШЕН!")
        self.logger.info(f"⏱️  Общее время: {total_duration:.1f} минут")
        self.logger.info(f"✅ Успешно: {successful}/{len(wallets)} кошельков")
        self.logger.info(f"📊 Всего собрано: {total_transactions} транзакций")
        
        # Сохраняем общий отчет
        final_report = {
            'collection_date': datetime.now().isoformat(),
            'total_wallets': len(wallets),
            'successful_wallets': successful,
            'total_transactions_collected': total_transactions,
            'duration_minutes': total_duration,
            'wallets_results': results
        }
        
        report_file = os.path.join("data", "collection_report.json")
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Отчет сохранен: {report_file}")
        
        return results
    
    def print_collection_status(self):
        """Вывод статуса сбора данных"""
        print(f"\n📊 СТАТУС СБОРА ИСТОРИЧЕСКИХ ДАННЫХ")
        print("="*60)
        
        if not self.collection_status:
            print("❌ Сбор данных еще не запускался")
            return
        
        total_wallets = len(self.collection_status)
        completed_wallets = len([w for w in self.collection_status.values() if w.get('analysis_completed')])
        total_transactions = sum(w.get('successful_transactions', 0) for w in self.collection_status.values())
        
        print(f"📋 Кошельков обработано: {completed_wallets}/{total_wallets}")
        print(f"💾 Всего транзакций собрано: {total_transactions}")
        
        if completed_wallets > 0:
            avg_txs = total_transactions / completed_wallets
            print(f"📊 Среднее транзакций на кошелек: {avg_txs:.1f}")
        
        # Детали по кошелькам
        print(f"\n📋 Детали по кошелькам:")
        for address, status in list(self.collection_status.items())[:10]:  # Показываем первые 10
            name = f"({address[:8]}...{address[-4:]})"
            date = status.get('last_collection_date', 'Не завершен')[:16]
            txs = status.get('successful_transactions', 0)
            duration = status.get('collection_duration_minutes', 0)
            
            print(f"  • {name:15s}: {txs:4d} тx, {duration:5.1f} мин ({date})")
        
        if len(self.collection_status) > 10:
            print(f"  ... и еще {len(self.collection_status) - 10} кошельков")
        
        # Проверяем наличие файлов данных
        data_dir = self.historical_data_dir
        if os.path.exists(data_dir):
            wallet_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"\n💾 Файлы данных: {len(wallet_dirs)} папок в {data_dir}")
        else:
            print(f"\n💾 Папка данных не создана: {data_dir}")
    
    def get_wallet_summary(self, wallet_address: str) -> Optional[Dict]:
        """Получить краткую сводку по кошельку"""
        wallet_dir = os.path.join(self.historical_data_dir, wallet_address)
        summary_file = os.path.join(wallet_dir, "summary.json")
        
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def list_collected_wallets(self) -> List[str]:
        """Список кошельков с собранными данными"""
        if not os.path.exists(self.historical_data_dir):
            return []
        
        return [d for d in os.listdir(self.historical_data_dir) 
                if os.path.isdir(os.path.join(self.historical_data_dir, d))]
    
    async def _analyze_wallet_history(self, transactions: List[Dict], wallet: Dict) -> Dict:
        """Анализ истории торговли кошелька"""
        analysis = {
            'wallet_address': wallet['address'],
            'wallet_name': wallet.get('name', 'Unknown'),
            'analysis_date': datetime.now().isoformat(),
            'total_transactions': len(transactions),
            'dex_transactions': len([tx for tx in transactions if tx.get('is_dex_transaction')]),
            'successful_transactions': len([tx for tx in transactions if tx['status'] == 'success']),
            'failed_transactions': len([tx for tx in transactions if tx['status'] != 'success']),
            'time_range': {
                'earliest': None,
                'latest': None
            },
            'trading_patterns': {},
            'tokens_traded': [],
            'profitability_analysis': {}
        }
        
        if transactions:
            # Временной диапазон
            timestamps = [tx['blockTime'] for tx in transactions if tx.get('blockTime')]
            if timestamps:
                analysis['time_range']['earliest'] = min(timestamps)
                analysis['time_range']['latest'] = max(timestamps)
            
            # Анализ токенов
            all_tokens = set()
            for tx in transactions:
                tokens = tx.get('tokens_involved', [])
                all_tokens.update(tokens)
            
            analysis['tokens_traded'] = list(all_tokens)
            analysis['unique_tokens_count'] = len(all_tokens)
            
            # Простые паттерны
            swap_types = [tx.get('swap_type') for tx in transactions if tx.get('swap_type')]
            analysis['trading_patterns'] = {
                'swap_types': list(set(swap_types)),
                'most_common_swap_type': max(set(swap_types), key=swap_types.count) if swap_types else None,
                'avg_transactions_per_day': self._calculate_avg_transactions_per_day(transactions)
            }
            
            # TODO: Здесь будет более сложный анализ прибыльности
            # analysis['profitability_analysis'] = await self._calculate_profitability(transactions)
        
        return analysis
    
    def _calculate_avg_transactions_per_day(self, transactions: List[Dict]) -> float:
        """Расчет среднего количества транзакций в день"""
        timestamps = [tx['blockTime'] for tx in transactions if tx.get('blockTime')]
        
        if len(timestamps) < 2:
            return 0
        
        min_time = min(timestamps)
        max_time = max(timestamps)
        days = (max_time - min_time) / (24 * 60 * 60)  # Секунды в дни
        
        return len(transactions) / max(days, 1)
    
    async def _save_wallet_data(self, address: str, transactions: List[Dict], analysis: Dict):
        """Сохранение данных кошелька"""
        wallet_dir = os.path.join(self.historical_data_dir, address)
        os.makedirs(wallet_dir, exist_ok=True)
        
        # Сохраняем транзакции
        transactions_file = os.path.join(wallet_dir, "transactions.json")
        with open(transactions_file, 'w') as f:
            json.dump(transactions, f, indent=2, default=str)
        
        # Сохраняем анализ
        analysis_file = os.path.join(wallet_dir, "analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.data_saved(wallet_dir, len(transactions))
    
    async def collect_all_wallets_history(self):
        """Сбор истории для всех кошельков"""
        try:
            with open(self.wallets_file, 'r') as f:
                wallets = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Файл кошельков не найден: {self.wallets_file}")
            return
        
        self.logger.info(f"📊 Начинаем сбор истории для {len(wallets)} кошельков")
        
        results = []
        
        for i, wallet in enumerate(wallets, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Кошелек {i}/{len(wallets)}: {wallet.get('name', 'Unknown')}")
            
            result = await self.collect_wallet_history(wallet)
            results.append(result)
            
            # Пауза между кошельками
            if i < len(wallets):
                self.logger.info("⏳ Пауза между кошельками...")
                await asyncio.sleep(5)
        
        # Сводная статистика
        successful = len([r for r in results if r['status'] == 'success'])
        total_transactions = sum(r.get('transactions_collected', 0) for r in results)
        
        self.logger.info(f"\n🎉 СБОР ЗАВЕРШЕН!")
        self.logger.info(f"Успешно: {successful}/{len(wallets)} кошельков")
        self.logger.info(f"Всего собрано: {total_transactions} транзакций")
        
        return results
    
    def print_collection_status(self):
        """Вывод статуса сбора данных"""
        print(f"\n📊 СТАТУС СБОРА ИСТОРИЧЕСКИХ ДАННЫХ")
        print("="*60)
        
        if not self.collection_status:
            print("❌ Сбор данных еще не запускался")
            return
        
        total_wallets = len(self.collection_status)
        completed_wallets = len([w for w in self.collection_status.values() if w.get('analysis_completed')])
        total_transactions = sum(w.get('successful_transactions', 0) for w in self.collection_status.values())
        
        print(f"Кошельков обработано: {completed_wallets}/{total_wallets}")
        print(f"Всего транзакций собрано: {total_transactions}")
        
        # Детали по кошелькам
        print(f"\n📋 Детали по кошелькам:")
        for address, status in self.collection_status.items():
            name = f"({address[:10]}...)"
            date = status.get('last_collection_date', 'Не завершен')
            txs = status.get('successful_transactions', 0)
            duration = status.get('collection_duration_minutes', 0)
            
            print(f"  • {name}: {txs} транзакций, {duration:.1f} мин ({date[:10]})")