
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

class WhaleSinalOrchestrator:
    def __init__(self):
        self.logger = setup_logger("orchestrator")
        self.config = self._load_config()
        
        # Инициализация сервисов
        self.whale_follower = None
        self.dex_follower = None
        
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
                }
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True
            },
            "data_collection": {
                "save_raw_data": True,
                "analyze_patterns": False  # Заглушка для Gemini
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
    
    async def run_all_services(self):
        """Запуск всех сервисов параллельно"""
        tasks = []
        
        if self.whale_follower:
            tasks.append(asyncio.create_task(self.start_whale_monitoring()))
        
        if self.dex_follower:
            tasks.append(asyncio.create_task(self.start_dex_monitoring()))
        
        if tasks:
            self.logger.info(f"🔄 Запуск {len(tasks)} сервисов...")
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            self.logger.warning("⚠️  Нет активных сервисов для запуска")
    
    def print_status(self):
        """Показать статус всех сервисов"""
        print("\n" + "="*60)
        print("🎯 WHALE SIGNAL ORCHESTRATOR STATUS")
        print("="*60)
        
        print(f"🐋 Whale Follower (ETH): {'✅ АКТИВЕН' if self.whale_follower else '❌ ОТКЛЮЧЕН'}")
        if self.whale_follower:
            wallets_count = len(self.whale_follower.monitored_wallets)
            print(f"   Отслеживается кошельков: {wallets_count}")
            auto_trade = self.whale_follower.config.get('enable_auto_trade', False)
            print(f"   Авто-торговля: {'✅ ВКЛ' if auto_trade else '❌ ВЫКЛ'}")
        
        print(f"\n📊 DEX Follower (Solana): {'✅ АКТИВЕН' if self.dex_follower else '❌ ОТКЛЮЧЕН'}")
        if self.dex_follower:
            dex_wallets_count = len(self.dex_follower.monitored_wallets)
            print(f"   Отслеживается кошельков: {dex_wallets_count}")
        
        print(f"\n📈 Анализ паттернов: {'✅ ВКЛ' if self.config['data_collection']['analyze_patterns'] else '❌ ВЫКЛ (заглушка)'}")
        print(f"💾 Сохранение данных: {'✅ ВКЛ' if self.config['data_collection']['save_raw_data'] else '❌ ВЫКЛ'}")
        
        print(f"\n⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def interactive_menu(self):
        """Интерактивное меню управления"""
        while True:
            print("\n" + "="*60)
            print("🎯 WHALE SIGNAL ORCHESTRATOR")
            print("="*60)
            print("1. 🚀 Запустить все сервисы")
            print("2. 📊 Показать статус")
            print("3. 🐋 Только ETH киты")
            print("4. 📈 Только Solana DEX")
            print("5. ⚙️  Настройки")
            print("6. 📋 Показать кошельки")
            print("7. 🔄 Перезагрузить конфигурацию")
            print("8. ❌ Выход")
            
            choice = input("\n➤ Выберите опцию: ").strip()
            
            if choice == '1':
                self.print_status()
                print("\n🚀 Запуск всех сервисов... (Ctrl+C для остановки)")
                try:
                    await self.run_all_services()
                except KeyboardInterrupt:
                    print("\n⏹️  Остановка сервисов...")
                    break
            
            elif choice == '2':
                self.print_status()
            
            elif choice == '3':
                if self.whale_follower:
                    print("\n🐋 Запуск только ETH мониторинга...")
                    try:
                        await self.start_whale_monitoring()
                    except KeyboardInterrupt:
                        print("\n⏹️  Остановка ETH мониторинга...")
                else:
                    print("❌ ETH Whale Follower не инициализирован")
            
            elif choice == '4':
                if self.dex_follower:
                    print("\n📈 Запуск только Solana DEX мониторинга...")
                    try:
                        await self.start_dex_monitoring()
                    except KeyboardInterrupt:
                        print("\n⏹️  Остановка DEX мониторинга...")
                else:
                    print("❌ DEX Follower не инициализирован")
            
            elif choice == '5':
                print("\n⚙️  Редактируйте config/main_config.json для изменения настроек")
                print("Текущая конфигурация:")
                print(json.dumps(self.config, indent=2, ensure_ascii=False))
            
            elif choice == '6':
                self._show_wallets()
            
            elif choice == '7':
                self.config = self._load_config()
                await self.initialize_services()
                print("✅ Конфигурация перезагружена")
            
            elif choice == '8':
                print("👋 До свидания!")
                break
            
            else:
                print("❌ Неверная опция")
    
    def _show_wallets(self):
        """Показать все отслеживаемые кошельки"""
        print("\n📋 ОТСЛЕЖИВАЕМЫЕ КОШЕЛЬКИ")
        print("="*60)
        
        # ETH кошельки
        if self.whale_follower and self.whale_follower.monitored_wallets:
            print("🐋 ETH Киты:")
            for i, wallet in enumerate(self.whale_follower.monitored_wallets, 1):
                name = wallet.get('name', 'Unknown')
                address = wallet.get('address', '')
                print(f"  {i}. {name}: {address[:10]}...{address[-6:]}")
        
        # Solana кошельки
        if self.dex_follower and self.dex_follower.monitored_wallets:
            print("\n📈 Solana DEX Торговцы:")
            for i, wallet in enumerate(self.dex_follower.monitored_wallets, 1):
                name = wallet.get('name', 'Unknown')
                address = wallet.get('address', '')
                print(f"  {i}. {name}: {address[:10]}...{address[-6:]}")

async def main():
    """Главная функция"""
    orchestrator = WhaleSinalOrchestrator()
    
    try:
        # Инициализация сервисов
        await orchestrator.initialize_services()
        
        # Запуск интерактивного меню
        await orchestrator.interactive_menu()
        
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())