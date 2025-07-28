#!/usr/bin/env python3
"""
Web Server для Whale Signal Orchestrator
Предоставляет REST API для веб-дашборда
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
from dataclasses import asdict

from utils.logger import ServiceLogger
from services.whale_follower import WhaleFollower
from services.dex_follower import DexFollower
from services.dex_data_collector import DexDataCollector
from services.paper_trading import PaperTradingEngine

# Pydantic models for API
class ServiceCommand(BaseModel):
    action: str  # start, stop, restart
    service: Optional[str] = None  # whale, dex, collector, all

class DataCollectionRequest(BaseModel):
    wallets: Optional[List[str]] = None  # Specific wallets or all
    force: bool = False

# Global state
app_state = {
    "services": {
        "whale_follower": None,
        "dex_follower": None,
        "data_collector": None,
        "paper_trading": None
    },
    "running_tasks": {},
    "logs": [],
    "start_time": datetime.now()
}

class WhaleWebServer:
    def __init__(self):
        self.logger = ServiceLogger("web_server")
        self.app = FastAPI(title="Whale Signal API", version="1.0.0")
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("Web server инициализирован")
    
    def _add_log(self, message: str, log_type: str = "info"):
        """Добавить запись в лог"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "type": log_type
        }
        
        app_state["logs"].append(log_entry)
        
        # Ограничиваем размер лога
        if len(app_state["logs"]) > 1000:
            app_state["logs"] = app_state["logs"][-500:]
        
        # Также логгируем в основной логгер
        self.logger.info(f"[{log_type.upper()}] {message}")
    
    def _initialize_services(self):
        """Инициализация всех сервисов"""
        try:
            # Whale Follower (ETH)
            app_state["services"]["whale_follower"] = WhaleFollower()
            self._add_log("ETH Whale Follower инициализирован", "success")
        except Exception as e:
            self._add_log(f"Ошибка инициализации Whale Follower: {e}", "error")
        
        try:
            # DEX Follower (Solana)
            app_state["services"]["dex_follower"] = DexFollower()
            self._add_log("Solana DEX Follower инициализирован", "success")
        except Exception as e:
            self._add_log(f"Ошибка инициализации DEX Follower: {e}", "error")
        
        try:
            # Data Collector
            app_state["services"]["data_collector"] = DexDataCollector()
            self._add_log("DEX Data Collector инициализирован", "success")
        except Exception as e:
            self._add_log(f"Ошибка инициализации Data Collector: {e}", "error")
        
        try:
            # Paper Trading Engine
            app_state["services"]["paper_trading"] = PaperTradingEngine()
            self._add_log("Paper Trading Engine инициализирован", "success")
            
            # Подключаем Paper Trading к сигналам
            self._setup_trading_integration()
            
        except Exception as e:
            self._add_log(f"Ошибка инициализации Paper Trading: {e}", "error")
    
    def _setup_trading_integration(self):
        """Настройка интеграции с Paper Trading"""
        try:
            paper_trader = app_state["services"]["paper_trading"]
            dex_follower = app_state["services"]["dex_follower"]
            
            if paper_trader and dex_follower:
                # Добавляем метод для отправки сигналов
                async def send_dex_signal(signal_data):
                    await paper_trader.process_dex_signal(signal_data)
                
                # Присваиваем метод DEX Follower'у
                dex_follower._send_trading_signal = send_dex_signal
                
                self._add_log("Paper Trading интегрирован с DEX Follower", "success")
                
        except Exception as e:
            self._add_log(f"Ошибка настройки интеграции: {e}", "error")
    
    def _setup_routes(self):
        """Настройка API маршрутов"""
        
        # Serve static files (HTML dashboard)
        @self.app.get("/")
        async def serve_dashboard():
            return FileResponse("whale_dashboard.html")
        
        # API Routes
        @self.app.get("/api/status")
        async def get_status():
            """Получить статус всех сервисов"""
            return await self._get_system_status()
        
        @self.app.get("/api/wallets/{chain}")
        async def get_wallets(chain: str):
            """Получить список кошельков для указанной сети"""
            try:
                if chain == "eth":
                    file_path = "config/wallets.json"
                elif chain == "solana":
                    file_path = "config/dex_wallets.json"
                else:
                    raise HTTPException(status_code=400, detail="Unsupported chain")
                
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        wallets = json.load(f)
                    return wallets
                else:
                    return []
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/services/start-all")
        async def start_all_services(background_tasks: BackgroundTasks):
            """Запустить все сервисы мониторинга"""
            try:
                # Диагностическая информация
                whale_available = app_state["services"]["whale_follower"] is not None
                dex_available = app_state["services"]["dex_follower"] is not None
                
                self._add_log(f"Запуск сервисов: ETH={whale_available}, DEX={dex_available}", "info")
                
                if not whale_available and not dex_available:
                    raise Exception("Нет доступных сервисов для запуска")
                
                background_tasks.add_task(self._start_monitoring_services)
                self._add_log("Запуск всех сервисов через API", "info")
                return {"status": "success", "message": "Сервисы запускаются"}
            except Exception as e:
                self._add_log(f"Ошибка запуска сервисов: {e}", "error")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/services/stop-all")
        async def stop_all_services():
            """Остановить все сервисы"""
            try:
                await self._stop_monitoring_services()
                self._add_log("Все сервисы остановлены через API", "warning")
                return {"status": "success", "message": "Сервисы остановлены"}
            except Exception as e:
                self._add_log(f"Ошибка остановки сервисов: {e}", "error")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/data-collection/start")
        async def start_data_collection(background_tasks: BackgroundTasks):
            """Запустить сбор исторических данных"""
            try:
                data_collector = app_state["services"]["data_collector"]
                if not data_collector:
                    raise HTTPException(status_code=400, detail="Data collector не инициализирован")
                
                # Проверяем что сбор данных не запущен уже
                if "data_collection" in app_state["running_tasks"]:
                    return {"status": "already_running", "message": "Сбор данных уже выполняется"}
                
                background_tasks.add_task(self._run_data_collection)
                self._add_log("Начат сбор исторических данных", "info")
                return {"status": "success", "message": "Сбор данных запущен"}
            except HTTPException:
                raise
            except Exception as e:
                self._add_log(f"Ошибка запуска сбора данных: {e}", "error")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/logs")
        async def get_recent_logs(limit: int = 100):
            """Получить последние логи"""
            return app_state["logs"][-limit:]
        
        @self.app.get("/api/analytics/activity")
        async def get_activity_analytics():
            """Получить данные активности для графика"""
            return await self._generate_activity_data()
        
        @self.app.get("/api/transactions/recent")
        async def get_recent_transactions(limit: int = 20):
            """Получить последние транзакции"""
            return await self._get_recent_transactions(limit)
        
        @self.app.get("/api/settings")
        async def get_settings():
            """Получить текущие настройки"""
            return await self._get_current_settings()
        
        @self.app.post("/api/settings/speed")
        async def update_processing_speed(settings: dict):
            """Обновить настройки скорости обработки"""
            try:
                await self._update_processing_speed(settings)
                self._add_log(f"Настройки скорости обновлены: {settings}", "info")
                return {"status": "success", "message": "Настройки обновлены"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/wallets/{chain}/add")
        async def add_wallet(chain: str, wallet_data: dict):
            """Добавить новый кошелек"""
            try:
                result = await self._add_wallet(chain, wallet_data)
                self._add_log(f"Добавлен кошелек {wallet_data.get('name', 'Unknown')} в {chain}", "success")
                return result
            except Exception as e:
                self._add_log(f"Ошибка добавления кошелька: {e}", "error")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/wallets/{chain}/{wallet_address}")
        async def remove_wallet(chain: str, wallet_address: str):
            """Удалить кошелек"""
            try:
                result = await self._remove_wallet(chain, wallet_address)
                self._add_log(f"Удален кошелек {wallet_address[:10]}... из {chain}", "warning")
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/collection/progress")
        async def get_collection_progress():
            """Получить прогресс сбора данных"""
            return await self._get_collection_progress()
        
        @self.app.get("/api/paper-trading/portfolio")
        async def get_portfolio():
            """Получить статистику портфеля paper trading"""
            try:
                paper_trader = app_state["services"]["paper_trading"]
                if not paper_trader:
                    raise HTTPException(status_code=404, detail="Paper Trading не инициализирован")
                
                stats = paper_trader.get_portfolio_stats()
                positions = [asdict(pos) for pos in paper_trader.positions.values()]
                recent_trades = [asdict(trade) for trade in paper_trader.trades[-10:]]
                
                return {
                    "portfolio": stats,
                    "positions": positions,
                    "recent_trades": recent_trades
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/paper-trading/toggle")
        async def toggle_paper_trading():
            """Включить/выключить paper trading"""
            try:
                paper_trader = app_state["services"]["paper_trading"]
                if not paper_trader:
                    raise HTTPException(status_code=404, detail="Paper Trading не инициализирован")
                
                if "paper_trading" in app_state["running_tasks"]:
                    # Останавливаем
                    task = app_state["running_tasks"]["paper_trading"]
                    task.cancel()
                    del app_state["running_tasks"]["paper_trading"]
                    status = "stopped"
                else:
                    # Запускаем
                    task = asyncio.create_task(paper_trader.monitor_signals())
                    app_state["running_tasks"]["paper_trading"] = task
                    status = "started"
                
                self._add_log(f"Paper Trading {status}", "info")
                return {"status": status}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/paper-trading/reset")
        async def reset_portfolio():
            """Сбросить портфель к начальному состоянию"""
            try:
                paper_trader = app_state["services"]["paper_trading"]
                if not paper_trader:
                    raise HTTPException(status_code=404, detail="Paper Trading не инициализирован")
                
                # Закрываем все позиции
                for position_id in list(paper_trader.positions.keys()):
                    await paper_trader._close_position(position_id, "reset")
                
                # Сбрасываем баланс
                paper_trader.current_balance = paper_trader.initial_balance
                paper_trader.trades.clear()
                paper_trader.processed_signals.clear()
                paper_trader._save_state()
                
                self._add_log("Portfolio сброшен к начальному состоянию", "warning")
                return {"status": "reset", "balance": paper_trader.current_balance}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _initialize_services(self):
        """Инициализация всех сервисов"""
        try:
            # Whale Follower (ETH)
            app_state["services"]["whale_follower"] = WhaleFollower()
            self._add_log("ETH Whale Follower инициализирован", "success")
        except Exception as e:
            self._add_log(f"Ошибка инициализации Whale Follower: {e}", "error")
        
        try:
            # DEX Follower (Solana)
            app_state["services"]["dex_follower"] = DexFollower()
            self._add_log("Solana DEX Follower инициализирован", "success")
        except Exception as e:
            self._add_log(f"Ошибка инициализации DEX Follower: {e}", "error")
        
        try:
            # Data Collector
            app_state["services"]["data_collector"] = DexDataCollector()
            self._add_log("DEX Data Collector инициализирован", "success")
        except Exception as e:
            self._add_log(f"Ошибка инициализации Data Collector: {e}", "error")
    
    async def _start_monitoring_services(self):
        """Запуск сервисов мониторинга в фоне"""
        try:
            # Останавливаем существующие задачи
            await self._stop_monitoring_services()
            
            # Запускаем ETH мониторинг
            whale_follower = app_state["services"]["whale_follower"]
            if whale_follower:
                self._add_log("Запуск ETH мониторинга...", "info")
                whale_task = asyncio.create_task(
                    whale_follower.monitor_all_wallets(),
                    name="whale_monitoring"
                )
                app_state["running_tasks"]["whale_monitoring"] = whale_task
                self._add_log("ETH мониторинг запущен", "success")
            else:
                self._add_log("ETH Whale Follower не инициализирован", "warning")
            
            # Запускаем Solana мониторинг
            dex_follower = app_state["services"]["dex_follower"] 
            if dex_follower:
                self._add_log("Запуск Solana DEX мониторинга...", "info")
                dex_task = asyncio.create_task(
                    dex_follower.monitor_all_wallets(),
                    name="dex_monitoring"
                )
                app_state["running_tasks"]["dex_monitoring"] = dex_task
                self._add_log("Solana DEX мониторинг запущен", "success")
            else:
                self._add_log("Solana DEX Follower не инициализирован", "warning")
            
            # Небольшая задержка для стабилизации
            await asyncio.sleep(1)
            
        except Exception as e:
            self._add_log(f"Ошибка запуска мониторинга: {e}", "error")
            raise
    
    async def _stop_monitoring_services(self):
        """Остановка всех сервисов мониторинга"""
        for task_name, task in app_state["running_tasks"].items():
            try:
                task.cancel()
                self._add_log(f"Задача {task_name} остановлена", "warning")
            except Exception as e:
                self._add_log(f"Ошибка остановки {task_name}: {e}", "error")
        
        app_state["running_tasks"].clear()
    
    async def _run_data_collection(self):
        """Запуск сбора исторических данных"""
        try:
            collector = app_state["services"]["data_collector"]
            if not collector:
                self._add_log("Data Collector не доступен", "error")
                return
            
            # Отмечаем что сбор данных запущен
            app_state["running_tasks"]["data_collection"] = True
            self._add_log("Начат сбор исторических данных", "info")
            
            # Запускаем сбор
            await collector.collect_all_wallets_history()
            
            self._add_log("Сбор исторических данных завершен", "success")
            
        except Exception as e:
            self._add_log(f"Ошибка сбора данных: {e}", "error")
        finally:
            # Убираем задачу из списка запущенных
            if "data_collection" in app_state["running_tasks"]:
                del app_state["running_tasks"]["data_collection"]
    
    async def _get_system_status(self) -> Dict:
        """Получить статус системы"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - app_state["start_time"]).total_seconds(),
            "services": {},
            "recent_logs": app_state["logs"][-10:],
            "analytics": await self._generate_activity_data(),
            "recent_transactions": await self._get_recent_transactions(10)
        }
        
        # Статус каждого сервиса
        whale_follower = app_state["services"]["whale_follower"]
        if whale_follower:
            whale_active = "whale_monitoring" in app_state["running_tasks"]
            whale_task = app_state["running_tasks"].get("whale_monitoring")
            
            # Проверяем что задача действительно работает
            if whale_task and hasattr(whale_task, 'done') and whale_task.done():
                whale_active = False
                # Убираем завершенную задачу
                if "whale_monitoring" in app_state["running_tasks"]:
                    del app_state["running_tasks"]["whale_monitoring"]
            
            status["services"]["whale_follower"] = {
                "active": whale_active,
                "wallets_count": len(whale_follower.monitored_wallets),
                "processed_transactions": len(whale_follower.processed_txs) if hasattr(whale_follower, 'processed_txs') else 0,
                "auto_trade_enabled": whale_follower.config.get('enable_auto_trade', False)
            }
        
        dex_follower = app_state["services"]["dex_follower"]
        if dex_follower:
            dex_active = "dex_monitoring" in app_state["running_tasks"]
            dex_task = app_state["running_tasks"].get("dex_monitoring")
            
            # Проверяем что задача действительно работает
            if dex_task and hasattr(dex_task, 'done') and dex_task.done():
                dex_active = False
                # Убираем завершенную задачу
                if "dex_monitoring" in app_state["running_tasks"]:
                    del app_state["running_tasks"]["dex_monitoring"]
            
            status["services"]["dex_follower"] = {
                "active": dex_active,
                "wallets_count": len(dex_follower.monitored_wallets),
                "processed_transactions": len(dex_follower.processed_txs) if hasattr(dex_follower, 'processed_txs') else 0,
                "meme_purchases": self._count_meme_purchases()
            }
        
        data_collector = app_state["services"]["data_collector"]
        if data_collector:
            collection_status = data_collector.collection_status
            completed_wallets = len([w for w in collection_status.values() if w.get('analysis_completed')])
            total_transactions = sum(w.get('successful_transactions', 0) for w in collection_status.values())
            
            is_collecting = "data_collection" in app_state["running_tasks"]
            
            status["services"]["data_collector"] = {
                "active": is_collecting,
                "completed_wallets": completed_wallets,
                "total_transactions": total_transactions,
                "status": "Собирает данные" if is_collecting else "Готов к сбору"
            }
        
        # Paper Trading статус
        paper_trader = app_state["services"]["paper_trading"]
        if paper_trader:
            is_trading = "paper_trading" in app_state["running_tasks"]
            portfolio_stats = paper_trader.get_portfolio_stats()
            
            status["services"]["paper_trading"] = {
                "active": is_trading,
                "portfolio_value": portfolio_stats["total_value"],
                "total_pnl": portfolio_stats["total_pnl"],
                "total_pnl_percentage": portfolio_stats["total_pnl_percentage"],
                "open_positions": portfolio_stats["open_positions"],
                "win_rate": portfolio_stats["win_rate"]
            }
        
        return status
    
    async def _generate_activity_data(self) -> Dict:
        """Генерация данных активности для графика"""
        # Простая заглушка - в реальности здесь будет анализ логов
        now = datetime.now()
        labels = [(now - timedelta(hours=i)).strftime("%H:%M") for i in range(24, 0, -1)]
        
        # Примерные данные (в реальности - из логов транзакций)
        eth_data = [0, 1, 0, 2, 1, 0, 3, 1, 2, 0, 1, 4, 2, 1, 0, 2, 3, 1, 0, 1, 2, 0, 1, 0]
        solana_data = [1, 2, 1, 3, 2, 1, 4, 3, 2, 1, 2, 5, 3, 2, 1, 3, 4, 2, 1, 2, 3, 1, 2, 1]
        
        return {
            "labels": labels,
            "eth_transactions": eth_data,
            "solana_transactions": solana_data
        }
    
    async def _get_recent_transactions(self, limit: int) -> List[Dict]:
        """Получить последние транзакции"""
        transactions = []
        
        # Пытаемся загрузить из файлов логов
        try:
            # ETH транзакции
            whale_follower = app_state["services"]["whale_follower"]
            if whale_follower and hasattr(whale_follower, 'trade_log_file'):
                if os.path.exists(whale_follower.trade_log_file):
                    with open(whale_follower.trade_log_file, 'r') as f:
                        eth_trades = json.load(f)
                        for trade in eth_trades[-limit//2:]:
                            transactions.append({
                                "type": "ETH Trade",
                                "description": f"Trade {trade.get('status', 'unknown')}",
                                "timestamp": trade.get('timestamp', '')[:16],
                                "wallet": trade.get('original_tx', 'N/A')[:16],
                                "chain": "Ethereum"
                            })
            
            # Solana транзакции
            dex_follower = app_state["services"]["dex_follower"]
            if dex_follower and hasattr(dex_follower, 'trade_analysis_file'):
                if os.path.exists(dex_follower.trade_analysis_file):
                    with open(dex_follower.trade_analysis_file, 'r') as f:
                        solana_trades = json.load(f)
                        for trade in solana_trades[-limit//2:]:
                            tx = trade.get('transaction', {})
                            transactions.append({
                                "type": "Solana DEX",
                                "description": f"Swap detected: {tx.get('swap_type', 'unknown')}",
                                "timestamp": trade.get('analyzed_at', '')[:16],
                                "wallet": tx.get('wallet_address', 'N/A')[:16],
                                "chain": "Solana"
                            })
        
        except Exception as e:
            self._add_log(f"Ошибка загрузки транзакций: {e}", "error")
        
        # Сортируем по времени и возвращаем последние
        transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return transactions[:limit]
    
    def _count_meme_purchases(self) -> int:
        """Подсчет покупок мем-коинов"""
        try:
            dex_follower = app_state["services"]["dex_follower"]
            if dex_follower and hasattr(dex_follower, 'trade_analysis_file'):
                if os.path.exists(dex_follower.trade_analysis_file):
                    with open(dex_follower.trade_analysis_file, 'r') as f:
                        analysis_data = json.load(f)
                        return sum(1 for d in analysis_data if d.get('analysis', {}).get('meme_coin_purchase'))
        except:
            pass
        return 0
    
    async def _get_current_settings(self) -> Dict:
        """Получить текущие настройки всех сервисов"""
        settings = {
            "processing_speed": {
                "dex_data_collector": {
                    "rpc_delay": getattr(app_state["services"]["data_collector"], 'rpc_delay', 0.2),
                    "batch_delay": 1.0,
                    "batch_size": 50
                },
                "dex_follower": {
                    "monitor_interval": 20,
                    "rpc_delay": 0.1
                },
                "whale_follower": {
                    "monitor_interval": 30,
                    "etherscan_delay": 0.25
                }
            },
            "data_collection": {
                "max_transactions_per_wallet": 1000,
                "days_back": 30,
                "skip_errors": True,
                "retry_failed": True
            }
        }
        
        # Загружаем актуальные настройки из конфиг файлов
        try:
            config_files = [
                ("config/dex_config.json", "dex_follower"),
                ("config/data_collection_config.json", "data_collector"),
                ("config/trading_config.json", "whale_follower")
            ]
            
            for config_file, service_key in config_files:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        if service_key in settings["processing_speed"]:
                            settings["processing_speed"][service_key].update(config)
        except Exception as e:
            self._add_log(f"Ошибка загрузки настроек: {e}", "warning")
        
        return settings
    
    async def _update_processing_speed(self, new_settings: Dict):
        """Обновить настройки скорости обработки"""
        # Обновляем настройки Data Collector
        if "data_collector" in new_settings:
            collector = app_state["services"]["data_collector"]
            if collector:
                dc_settings = new_settings["data_collector"]
                
                # Обновляем RPC delay
                if "rpc_delay" in dc_settings:
                    collector.rpc_delay = float(dc_settings["rpc_delay"])
                
                # Обновляем настройки в конфиге
                config_file = "config/data_collection_config.json"
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Обновляем конфигурацию
                    if "batch_delay" in dc_settings:
                        config["collection_settings"]["delay_between_batches"] = float(dc_settings["batch_delay"])
                    if "batch_size" in dc_settings:
                        config["collection_settings"]["batch_size"] = int(dc_settings["batch_size"])
                    if "max_transactions" in dc_settings:
                        config["collection_settings"]["max_transactions_per_wallet"] = int(dc_settings["max_transactions"])
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
        
        # Обновляем настройки DEX Follower
        if "dex_follower" in new_settings:
            dex_follower = app_state["services"]["dex_follower"]
            if dex_follower:
                df_settings = new_settings["dex_follower"]
                
                if "rpc_delay" in df_settings:
                    dex_follower.rpc_delay = float(df_settings["rpc_delay"])
                if "monitor_interval" in df_settings:
                    dex_follower.config["monitor_interval"] = int(df_settings["monitor_interval"])
        
        # Обновляем настройки Whale Follower
        if "whale_follower" in new_settings:
            whale_follower = app_state["services"]["whale_follower"]
            if whale_follower:
                wf_settings = new_settings["whale_follower"]
                
                if "etherscan_delay" in wf_settings:
                    whale_follower.etherscan_delay = float(wf_settings["etherscan_delay"])
                if "monitor_interval" in wf_settings:
                    whale_follower.config["monitor_interval"] = int(wf_settings["monitor_interval"])
    
    async def _add_wallet(self, chain: str, wallet_data: Dict) -> Dict:
        """Добавить новый кошелек в соответствующий файл"""
        if chain == "eth":
            file_path = "config/wallets.json"
        elif chain == "solana":
            file_path = "config/dex_wallets.json"
        else:
            raise ValueError(f"Неподдерживаемая сеть: {chain}")
        
        # Валидация данных кошелька
        required_fields = ["address", "name"]
        for field in required_fields:
            if field not in wallet_data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")
        
        # Загружаем существующие кошельки
        wallets = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                wallets = json.load(f)
        
        # Проверяем на дубликаты
        existing_addresses = {w.get("address") for w in wallets}
        if wallet_data["address"] in existing_addresses:
            raise ValueError("Кошелек с таким адресом уже существует")
        
        # Добавляем метаданные
        wallet_data["added_at"] = datetime.now().isoformat()
        wallet_data["added_via"] = "web_interface"
        
        # Устанавливаем значения по умолчанию для Solana
        if chain == "solana":
            wallet_data.setdefault("category", "trader")
            wallet_data.setdefault("estimated_win_rate", 0.5)
            wallet_data.setdefault("specialization", ["general"])
        
        # Добавляем кошелек
        wallets.append(wallet_data)
        
        # Создаем директорию если не существует
        os.makedirs("config", exist_ok=True)
        
        # Сохраняем
        with open(file_path, 'w') as f:
            json.dump(wallets, f, indent=2, ensure_ascii=False)
        
        # Обновляем сервисы
        self._reload_services_wallets()
        
        return {
            "status": "success",
            "message": f"Кошелек добавлен в {chain}",
            "wallet": wallet_data,
            "total_wallets": len(wallets)
        }
    
    async def _remove_wallet(self, chain: str, wallet_address: str) -> Dict:
        """Удалить кошелек из файла"""
        if chain == "eth":
            file_path = "config/wallets.json"
        elif chain == "solana":
            file_path = "config/dex_wallets.json"
        else:
            raise ValueError(f"Неподдерживаемая сеть: {chain}")
        
        if not os.path.exists(file_path):
            raise ValueError("Файл кошельков не найден")
        
        # Загружаем кошельки
        with open(file_path, 'r') as f:
            wallets = json.load(f)
        
        # Находим и удаляем кошелек
        original_count = len(wallets)
        wallets = [w for w in wallets if w.get("address") != wallet_address]
        
        if len(wallets) == original_count:
            raise ValueError("Кошелек не найден")
        
        # Сохраняем обновленный список
        with open(file_path, 'w') as f:
            json.dump(wallets, f, indent=2, ensure_ascii=False)
        
        # Обновляем сервисы
        self._reload_services_wallets()
        
        return {
            "status": "success",
            "message": f"Кошелек удален из {chain}",
            "total_wallets": len(wallets)
        }
    
    def _reload_services_wallets(self):
        """Перезагрузить кошельки в сервисах"""
        try:
            # Перезагружаем кошельки в DEX Follower
            if app_state["services"]["dex_follower"]:
                app_state["services"]["dex_follower"].monitored_wallets = app_state["services"]["dex_follower"]._load_dex_wallets()
            
            # Перезагружаем кошельки в Whale Follower
            if app_state["services"]["whale_follower"]:
                app_state["services"]["whale_follower"].monitored_wallets = app_state["services"]["whale_follower"]._load_wallets()
            
            # Перезагружаем кошельки в Data Collector
            if app_state["services"]["data_collector"]:
                app_state["services"]["data_collector"].monitored_wallets = app_state["services"]["data_collector"]._load_dex_wallets()
            
            self._add_log("Кошельки перезагружены во всех сервисах", "info")
        except Exception as e:
            self._add_log(f"Ошибка перезагрузки кошельков: {e}", "error")
    
    async def _get_collection_progress(self) -> Dict:
        """Получить детальный прогресс сбора данных"""
        collector = app_state["services"]["data_collector"]
        if not collector:
            return {"status": "service_not_available"}
        
        progress = {
            "is_collecting": "data_collection" in app_state["running_tasks"],
            "total_wallets": len(collector.monitored_wallets),
            "completed_wallets": 0,
            "total_transactions": 0,
            "current_wallet": None,
            "current_progress": 0,
            "estimated_time_remaining": None,
            "collection_status": {}
        }
        
        if hasattr(collector, 'collection_status'):
            status = collector.collection_status
            progress["completed_wallets"] = len([w for w in status.values() if w.get('analysis_completed')])
            progress["total_transactions"] = sum(w.get('successful_transactions', 0) for w in status.values())
            progress["collection_status"] = status
            
            # Прогресс в процентах
            if progress["total_wallets"] > 0:
                progress["current_progress"] = (progress["completed_wallets"] / progress["total_wallets"]) * 100
        
        return progress
        """Добавить запись в лог"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "type": log_type
        }
        
        app_state["logs"].append(log_entry)
        
        # Ограничиваем размер лога
        if len(app_state["logs"]) > 1000:
            app_state["logs"] = app_state["logs"][-500:]
        
        # Также логгируем в основной логгер
        self.logger.info(f"[{log_type.upper()}] {message}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Запуск веб-сервера"""
        self._add_log(f"Запуск веб-сервера на {host}:{port}", "info")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def main():
    """Основная функция запуска веб-сервера"""
    server = WhaleWebServer()
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n👋 Веб-сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")

if __name__ == "__main__":
    main()