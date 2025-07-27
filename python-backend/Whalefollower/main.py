#!/usr/bin/env python3
"""
Whale Signal Follower - Main Orchestrator
Координирует работу всех микросервисов для отслеживания криптовалютных сигналов
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List
import logging

from utils.logger import setup_logger
from services.whale_follower import WhaleFollower
from services.dex_follower import DexFollower
from services.dex_data_collector import DexDataCollector

class WhaleSignalOrchestrator:
    def __init__(self):
        self.logger = setup_logger("orchestrator")
        self.config = self._load_config()
        
        # Инициализация сервисов
        self.whale_follower = None
        self.dex_follower = None
        self.dex_data_collector = None
        
        self.running_services = []
        
    def _load_config(self) -> Dict:
        """Загрузка общей конфигурации"""
        default_config = {
            "services": {
                "whale_follower": {
                    "enabled": True,
                    "monitor_interval": 30
                },
                "dex_follower": {
                    "enabled": True,
                    "monitor_interval": 20
                },
                "dex_data_collector": {
                    "enabled": True,
                    "auto_collect": False
                }
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True
            },
            "data_collection": {
                "save_raw_data": True,
                "analyze_patterns": False  # Заглушка для Gemini
            },
            "general": {
                "timezone": "UTC",
                "max_concurrent_services": 3,
                "error_retry_attempts": 3,
                "health_check_interval": 300
            }
        }
        
        config_path = "config/main_config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            os.makedirs("config", exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Создан файл конфигурации: {config_path}")
            return default_config
    
    async def initialize_services(self):
        """Инициализация всех сервисов"""
        self.logger.info("🚀 Инициализация сервисов...")
        
        # Инициализация Whale Follower (ETH)
        if self.config["services"]["whale_follower"]["enabled"]:
            try:
                self.whale_follower = WhaleFollower()
                self.logger.info("✅ Whale Follower (ETH) инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка инициализации Whale Follower: {e}")
        
        # Инициализация DEX Follower (Solana)
        if self.config["services"]["dex_follower"]["enabled"]:
            try:
                self.dex_follower = DexFollower()
                self.logger.info("✅ DEX Follower (Solana) инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка инициализации DEX Follower: {e}")
        
        # Инициализация DexDataCollector
        if self.config["services"]["dex_data_collector"]["enabled"]:
            try:
                self.dex_data_collector = DexDataCollector()
                self.logger.info("✅ DEX Data Collector инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка инициализации DEX Data Collector: {e}")
    
    async def start_whale_monitoring(self):
        """Запуск мониторинга ETH китов"""
        if self.whale_follower:
            self.logger.info("🐋 Запуск мониторинга ETH китов...")
            try:
                await self.whale_follower.monitor_all_wallets()
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге китов: {e}")
    
    async def start_dex_monitoring(self):
        """Запуск мониторинга Solana DEX торговцев"""
        if self.dex_follower:
            self.logger.info("📊 Запуск мониторинга Solana DEX...")
            try:
                await self.dex_follower.monitor_all_wallets()
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге DEX: {e}")
    
    async def start_historical_collection(self):
        """Запуск сбора исторических данных"""
        if self.dex_data_collector:
            self.logger.info("📈 Запуск сбора исторических данных...")
            try:
                await self.dex_data_collector.collect_all_wallets_history()
            except Exception as e:
                self.logger.error(f"Ошибка сбора исторических данных: {e}")
    
    async def run_all_services(self):
        """Запуск всех сервисов параллельно"""
        tasks = []
        
        if self.whale_follower:
            tasks.append(asyncio.create_task(self.start_whale_monitoring()))
        
        if self.dex_follower:
            tasks.append(asyncio.create_task(self.start_dex_monitoring()))
        
        if tasks:
            self.logger.info(f"🔄 Запуск {len(tasks)} сервисов мониторинга...")
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            self.logger.warning("⚠️  Нет активных сервисов для запуска")
    
    def print_status(self):
        """Показать статус всех сервисов"""
        print("\n" + "="*70)
        print("🎯 WHALE SIGNAL ORCHESTRATOR STATUS")
        print("="*70)
        
        # Статус ETH Whale Follower
        print(f"🐋 Whale Follower (ETH): {'✅ АКТИВЕН' if self.whale_follower else '❌ ОТКЛЮЧЕН'}")
        if self.whale_follower:
            wallets_count = len(self.whale_follower.monitored_wallets)
            auto_trade = self.whale_follower.config.get('enable_auto_trade', False)
            dry_run = self.whale_follower.config.get('dry_run', True)
            print(f"   📊 Отслеживается кошельков: {wallets_count}")
            print(f"   🔄 Авто-торговля: {'✅ ВКЛ' if auto_trade else '❌ ВЫКЛ'}")
            print(f"   🧪 Режим тестирования: {'✅ ВКЛ' if dry_run else '❌ ВЫКЛ'}")
        
        # Статус Solana DEX Follower
        print(f"\n📊 DEX Follower (Solana): {'✅ АКТИВЕН' if self.dex_follower else '❌ ОТКЛЮЧЕН'}")
        if self.dex_follower:
            dex_wallets_count = len(self.dex_follower.monitored_wallets)
            print(f"   📊 Отслеживается кошельков: {dex_wallets_count}")
            print(f"   ⏱️  Интервал сканирования: {self.config['services']['dex_follower']['monitor_interval']}с")
        
        # Статус DEX Data Collector
        print(f"\n📈 DEX Data Collector: {'✅ АКТИВЕН' if self.dex_data_collector else '❌ ОТКЛЮЧЕН'}")
        if self.dex_data_collector:
            auto_collect = self.config['services']['dex_data_collector']['auto_collect']
            print(f"   🔄 Режим автосбора: {'✅ ВКЛ' if auto_collect else '❌ ВЫКЛ'}")
            
            # Статистика сбора
            if hasattr(self.dex_data_collector, 'collection_status') and self.dex_data_collector.collection_status:
                completed = len([w for w in self.dex_data_collector.collection_status.values() if w.get('analysis_completed')])
                total_collected = sum(w.get('successful_transactions', 0) for w in self.dex_data_collector.collection_status.values())
                print(f"   📊 Собрано кошельков: {completed}")
                print(f"   💾 Всего транзакций: {total_collected}")
        
        # Общие настройки
        print(f"\n⚙️  Общие настройки:")
        print(f"   📈 Анализ паттернов: {'✅ ВКЛ' if self.config['data_collection']['analyze_patterns'] else '❌ ВЫКЛ (заглушка)'}")
        print(f"   💾 Сохранение данных: {'✅ ВКЛ' if self.config['data_collection']['save_raw_data'] else '❌ ВЫКЛ'}")
        print(f"   📝 Уровень логов: {self.config['logging']['level']}")
        
        print(f"\n⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _show_wallets(self):
        """Показать все отслеживаемые кошельки"""
        print("\n📋 ОТСЛЕЖИВАЕМЫЕ КОШЕЛЬКИ")
        print("="*70)
        
        # ETH кошельки
        if self.whale_follower and self.whale_follower.monitored_wallets:
            print("🐋 ETH Киты:")
            for i, wallet in enumerate(self.whale_follower.monitored_wallets, 1):
                name = wallet.get('name', 'Unknown')
                address = wallet.get('address', '')
                print(f"  {i:2d}. {name:25s} {address[:10]}...{address[-6:]}")
        else:
            print("🐋 ETH Киты: Не загружены")
        
        # Solana кошельки
        if self.dex_follower and self.dex_follower.monitored_wallets:
            print(f"\n📈 Solana DEX Торговцы:")
            for i, wallet in enumerate(self.dex_follower.monitored_wallets, 1):
                name = wallet.get('name', 'Unknown')
                address = wallet.get('address', '')
                category = wallet.get('category', 'N/A')
                win_rate = wallet.get('estimated_win_rate', 0)
                print(f"  {i:2d}. {name:25s} {address[:10]}...{address[-6:]} ({category}, WR: {win_rate:.0%})")
        else:
            print("\n📈 Solana DEX Торговцы: Не загружены")
    
    def _show_statistics(self):
        """Показать статистику работы"""
        print("\n📊 СТАТИСТИКА РАБОТЫ")
        print("="*70)
        
        # Статистика ETH
        if self.whale_follower:
            processed_txs = len(self.whale_follower.processed_txs) if hasattr(self.whale_follower, 'processed_txs') else 0
            print(f"🐋 ETH Whale Follower:")
            print(f"   ✅ Обработано транзакций: {processed_txs}")
            
            # Попытка показать недавние сделки
            try:
                if os.path.exists(self.whale_follower.trade_log_file):
                    with open(self.whale_follower.trade_log_file, 'r') as f:
                        trades = json.load(f)
                        print(f"   💰 Выполнено сделок: {len(trades)}")
                        if trades:
                            recent_trades = trades[-3:]
                            print(f"   📈 Последние сделки:")
                            for trade in recent_trades:
                                status = trade.get('status', 'unknown')
                                timestamp = trade.get('timestamp', '')[:16]
                                print(f"      - {timestamp}: {status}")
            except:
                print(f"   💰 Выполнено сделок: 0")
        
        # Статистика Solana DEX
        if self.dex_follower:
            dex_processed = len(self.dex_follower.processed_txs) if hasattr(self.dex_follower, 'processed_txs') else 0
            print(f"\n📊 Solana DEX Follower:")
            print(f"   ✅ Обработано транзакций: {dex_processed}")
            
            # Статистика анализа
            try:
                if hasattr(self.dex_follower, 'trade_analysis_file') and os.path.exists(self.dex_follower.trade_analysis_file):
                    with open(self.dex_follower.trade_analysis_file, 'r') as f:
                        analysis_data = json.load(f)
                        meme_purchases = sum(1 for d in analysis_data if d.get('analysis', {}).get('meme_coin_purchase'))
                        print(f"   🎯 Покупки мем-коинов: {meme_purchases}")
                        print(f"   📊 Проанализировано: {len(analysis_data)}")
            except:
                print(f"   📊 Проанализировано: 0")
        
        # Статистика сбора исторических данных
        if self.dex_data_collector:
            self.dex_data_collector.print_collection_status()
    
    async def interactive_menu(self):
        """Интерактивное меню управления"""
        while True:
            print("\n" + "="*70)
            print("🎯 WHALE SIGNAL ORCHESTRATOR")
            print("="*70)
            print("1. 🚀 Запустить все сервисы мониторинга")
            print("2. 📊 Показать статус сервисов")
            print("3. 🐋 Только ETH киты")
            print("4. 📈 Только Solana DEX мониторинг")
            print("5. 📊 Собрать исторические данные")
            print("6. 📋 Показать кошельки")
            print("7. 📈 Статистика и аналитика")
            print("8. ⚙️  Настройки")
            print("9. 🔄 Перезагрузить конфигурацию")
            print("10. 🧪 Тестовые функции")
            print("11. ❌ Выход")
            
            choice = input("\n➤ Выберите опцию: ").strip()
            
            if choice == '1':
                self.print_status()
                print("\n🚀 Запуск всех сервисов мониторинга...")
                print("⚠️  Нажмите Ctrl+C для остановки")
                try:
                    await self.run_all_services()
                except KeyboardInterrupt:
                    print("\n⏹️  Остановка всех сервисов...")
                    break
            
            elif choice == '2':
                self.print_status()
            
            elif choice == '3':
                if self.whale_follower:
                    print("\n🐋 Запуск только ETH мониторинга...")
                    print("⚠️  Нажмите Ctrl+C для остановки")
                    try:
                        await self.start_whale_monitoring()
                    except KeyboardInterrupt:
                        print("\n⏹️  Остановка ETH мониторинга...")
                else:
                    print("❌ ETH Whale Follower не инициализирован")
            
            elif choice == '4':
                if self.dex_follower:
                    print("\n📈 Запуск только Solana DEX мониторинга...")
                    print("⚠️  Нажмите Ctrl+C для остановки")
                    try:
                        await self.start_dex_monitoring()
                    except KeyboardInterrupt:
                        print("\n⏹️  Остановка DEX мониторинга...")
                else:
                    print("❌ DEX Follower не инициализирован")
            
            elif choice == '5':
                if self.dex_data_collector:
                    print("\n📊 СБОР ИСТОРИЧЕСКИХ ДАННЫХ")
                    print("-" * 50)
                    print("1. Собрать данные для всех кошельков")
                    print("2. Показать статус сбора")
                    print("3. Собрать данные для одного кошелька")
                    print("4. Назад")
                    
                    sub_choice = input("Выберите опцию: ").strip()
                    
                    if sub_choice == '1':
                        print("\n🚀 Запуск сбора исторических данных для всех кошельков...")
                        print("⚠️  Это может занять много времени (часы)!")
                        print("⚠️  Будут использованы RPC запросы к Solana")
                        confirm = input("Продолжить? (y/N): ").strip().lower()
                        if confirm == 'y':
                            try:
                                await self.start_historical_collection()
                            except KeyboardInterrupt:
                                print("\n⏹️  Сбор данных прерван пользователем")
                    
                    elif sub_choice == '2':
                        self.dex_data_collector.print_collection_status()
                    
                    elif sub_choice == '3':
                        # Показать доступные кошельки
                        try:
                            with open('config/dex_wallets.json', 'r') as f:
                                wallets = json.load(f)
                            
                            print("\nДоступные кошельки:")
                            for i, wallet in enumerate(wallets, 1):
                                name = wallet.get('name', 'Unknown')
                                address = wallet.get('address', '')
                                print(f"{i}. {name} ({address[:10]}...)")
                            
                            wallet_num = input(f"Выберите кошелек (1-{len(wallets)}): ").strip()
                            try:
                                wallet_idx = int(wallet_num) - 1
                                if 0 <= wallet_idx < len(wallets):
                                    selected_wallet = wallets[wallet_idx]
                                    print(f"Сбор данных для: {selected_wallet.get('name')}")
                                    await self.dex_data_collector.collect_wallet_history(selected_wallet)
                                else:
                                    print("❌ Неверный номер кошелька")
                            except ValueError:
                                print("❌ Введите корректный номер")
                        except FileNotFoundError:
                            print("❌ Файл dex_wallets.json не найден")
                else:
                    print("❌ DEX Data Collector не инициализирован")
            
            elif choice == '6':
                self._show_wallets()
            
            elif choice == '7':
                self._show_statistics()
            
            elif choice == '8':
                print("\n⚙️  НАСТРОЙКИ")
                print("-" * 50)
                print("1. Показать текущую конфигурацию")
                print("2. Редактировать файлы конфигурации")
                print("3. Сбросить настройки по умолчанию")
                print("4. Назад")
                
                settings_choice = input("Выберите опцию: ").strip()
                
                if settings_choice == '1':
                    print("\nТекущая конфигурация:")
                    print(json.dumps(self.config, indent=2, ensure_ascii=False))
                
                elif settings_choice == '2':
                    print("\n📝 Файлы конфигурации для редактирования:")
                    print("• config/main_config.json - Основные настройки")
                    print("• config/trading_config.json - Настройки торговли (ETH)")
                    print("• config/dex_config.json - Настройки DEX (Solana)")
                    print("• config/data_collection_config.json - Настройки сбора данных")
                    print("• config/wallets.json - ETH кошельки")
                    print("• config/dex_wallets.json - Solana кошельки")
                
                elif settings_choice == '3':
                    confirm = input("⚠️  Сбросить все настройки? (y/N): ").strip().lower()
                    if confirm == 'y':
                        # Удаляем конфиг файлы для пересоздания
                        try:
                            os.remove("config/main_config.json")
                            self.config = self._load_config()
                            print("✅ Настройки сброшены")
                        except:
                            print("❌ Ошибка сброса настроек")
            
            elif choice == '9':
                print("🔄 Перезагрузка конфигурации...")
                self.config = self._load_config()
                await self.initialize_services()
                print("✅ Конфигурация перезагружена")
            
            elif choice == '10':
                print("\n🧪 ТЕСТОВЫЕ ФУНКЦИИ")
                print("-" * 50)
                print("1. Тест подключения к Solana RPC")
                print("2. Тест получения транзакций")
                print("3. Тест парсинга DEX транзакции")
                print("4. Назад")
                
                test_choice = input("Выберите тест: ").strip()
                
                if test_choice == '1':
                    if self.dex_follower:
                        # Простой тест RPC
                        print("🔗 Тестирование подключения к Solana RPC...")
                        try:
                            import requests
                            payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
                            response = requests.post(self.dex_follower.quicknode_url, json=payload, timeout=5)
                            if response.status_code == 200:
                                print("✅ Подключение к Solana RPC успешно")
                            else:
                                print(f"❌ Ошибка подключения: {response.status_code}")
                        except Exception as e:
                            print(f"❌ Ошибка: {e}")
                    else:
                        print("❌ DEX Follower не инициализирован")
                
                elif test_choice == '2':
                    if self.dex_follower and self.dex_follower.monitored_wallets:
                        wallet = self.dex_follower.monitored_wallets[0]
                        print(f"📊 Тест получения транзакций для {wallet.get('name')}...")
                        try:
                            txs = await self.dex_follower.get_wallet_transactions(wallet['address'], limit=5)
                            print(f"✅ Получено {len(txs)} транзакций")
                            for tx in txs[:3]:
                                print(f"   - {tx.get('signature', 'N/A')[:20]}...")
                        except Exception as e:
                            print(f"❌ Ошибка: {e}")
                    else:
                        print("❌ Нет доступных кошельков для тестирования")
            
            elif choice == '11':
                print("\n👋 Завершение работы Whale Signal Orchestrator...")
                print("До свидания! 🐋")
                break
            
            else:
                print("❌ Неверная опция. Попробуйте еще раз.")

async def main():
    """Главная функция"""
    print("🎯 Запуск Whale Signal Orchestrator...")
    
    orchestrator = WhaleSignalOrchestrator()
    
    try:
        # Инициализация сервисов
        await orchestrator.initialize_services()
        
        # Запуск интерактивного меню
        await orchestrator.interactive_menu()
        
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        logging.exception("Критическая ошибка")

if __name__ == "__main__":
    asyncio.run(main())