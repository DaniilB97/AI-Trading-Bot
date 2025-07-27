#!/usr/bin/env python3
"""
Общий модуль логгирования для всех сервисов
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Настройка логгера для сервиса
    
    Args:
        name: Имя логгера (обычно имя сервиса)
        level: Уровень логгирования (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Сохранять ли логи в файл
    
    Returns:
        Настроенный логгер
    """
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Формат сообщений
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый обработчик
    if log_to_file:
        # Создаем директорию для логов
        os.makedirs("logs", exist_ok=True)
        
        # Имя файла с датой
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = f"logs/{name}_{date_str}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ServiceLogger:
    """
    Специализированный логгер для сервисов мониторинга
    Добавляет методы для логгирования специфичных событий
    """
    
    def __init__(self, service_name: str):
        self.logger = setup_logger(service_name)
        self.service_name = service_name
    
    def wallet_scan_start(self, wallet_address: str, wallet_name: str = "Unknown"):
        """Логгирование начала сканирования кошелька"""
        self.logger.info(f"🔍 Сканирование кошелька: {wallet_name} ({wallet_address[:10]}...)")
    
    def transaction_found(self, tx_hash: str, tx_type: str = "swap"):
        """Логгирование найденной транзакции"""
        self.logger.info(f"💰 Найдена транзакция {tx_type}: {tx_hash}")
    
    def trade_opportunity(self, from_token: str, to_token: str, amount: str = "N/A"):
        """Логгирование торговой возможности"""
        self.logger.info(f"🎯 Торговая возможность: {from_token} → {to_token} (сумма: {amount})")
    
    def trade_executed(self, tx_hash: str, status: str = "success"):
        """Логгирование выполненной торговли"""
        if status == "success":
            self.logger.info(f"✅ Торговля выполнена: {tx_hash}")
        else:
            self.logger.error(f"❌ Ошибка торговли: {tx_hash} - {status}")
    
    def rate_limit_warning(self, api_name: str, wait_time: float):
        """Логгирование ограничений скорости API"""
        self.logger.warning(f"⏳ Rate limit {api_name}: ожидание {wait_time:.1f}с")
    
    def connection_error(self, service: str, error: str):
        """Логгирование ошибок подключения"""
        self.logger.error(f"🔌 Ошибка подключения к {service}: {error}")
    
    def data_saved(self, file_path: str, record_count: int = None):
        """Логгирование сохранения данных"""
        count_str = f" ({record_count} записей)" if record_count else ""
        self.logger.info(f"💾 Данные сохранены: {file_path}{count_str}")
    
    def analysis_complete(self, analysis_type: str, results_count: int):
        """Логгирование завершения анализа"""
        self.logger.info(f"📊 Анализ завершен: {analysis_type} - {results_count} результатов")
    
    def debug(self, message: str):
        """Отладочное сообщение"""
        self.logger.debug(f"🔧 {message}")
    
    def info(self, message: str):
        """Информационное сообщение"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Предупреждение"""
        self.logger.warning(f"⚠️  {message}")
    
    def error(self, message: str):
        """Ошибка"""
        self.logger.error(f"❌ {message}")
    
    def critical(self, message: str):
        """Критическая ошибка"""
        self.logger.critical(f"🚨 {message}")
        