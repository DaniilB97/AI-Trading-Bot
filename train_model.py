#!/usr/bin/env python3
"""
Скрипт для тренировки GRU модели предсказания ETH
Оптимизирован для Mac M3 Pro (Metal Performance Shaders)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import argparse
from tqdm import tqdm

# Проверка и настройка для Mac M3
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🎉 Используем Apple M3 Pro GPU (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("💻 Используем CPU")

# Импорт компонентов системы
from config_loader import config
from eth_analysis_system import (
    TelegramParser, 
    NewsRelevanceClassifier,
    TechnicalAnalyzer,
    ETHPricePredictor,
    PricePredictor
)

class GRUTrainer:
    """Класс для тренировки GRU модели"""
    
    def __init__(self):
        self.config = config
        self.device = device
        self.logger = config.logger
        
        # Создаем директории
        self.setup_directories()
        
    def setup_directories(self):
        """Создание необходимых директорий"""
        dirs = ['models', 'data/raw', 'data/processed', 'logs/training']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    async def collect_telegram_data(self, days_back=30):
        """Сбор данных из Telegram"""
        self.logger.info("📱 Начинаем сбор данных из Telegram...")
        
        # Проверяем наличие сохраненных данных
        news_cache_path = Path(f"data/raw/telegram_news_{days_back}d.csv")
        
        if news_cache_path.exists():
            self.logger.info("📂 Найден кэш новостей, загружаем...")
            news_df = pd.read_csv(news_cache_path)
            news_df['date'] = pd.to_datetime(news_df['date'])
            return news_df
        
        # Собираем новые данные
        telegram_config = self.config.get_telegram_config()
        channels = self.config.get_telegram_channels()
        
        parser = TelegramParser(**telegram_config)
        await parser.connect()
        
        try:
            news_df = await parser.parse_multiple_channels(
                channels, 
                days_back=days_back,
                limit=500  # Ограничиваем для первого запуска
            )
            
            # Сохраняем кэш
            news_df.to_csv(news_cache_path, index=False)
            self.logger.info(f"💾 Сохранено {len(news_df)} новостей в кэш")
            
            return news_df
            
        finally:
            await parser.close()
    
    def prepare_technical_data(self, days_back=60):
        """Подготовка технических данных"""
        self.logger.info("📊 Загружаем технические данные ETH...")
        
        # Проверяем кэш
        tech_cache_path = Path(f"data/raw/eth_technical_{days_back}d.csv")
        
        if tech_cache_path.exists():
            self.logger.info("📂 Найден кэш технических данных, загружаем...")
            df = pd.read_csv(tech_cache_path, index_col=0, parse_dates=True)
            return df
        
        # Загружаем новые данные
        analyzer = TechnicalAnalyzer()
        df = analyzer.fetch_eth_data(period=f"{days_back}d")
        df = analyzer.calculate_indicators(df)
        
        # Сохраняем кэш
        df.to_csv(tech_cache_path)
        self.logger.info(f"💾 Сохранены технические данные: {len(df)} записей")
        
        return df
    
    def create_training_dataset(self, tech_df, news_df=None):
        """Создание датасета для тренировки"""
        self.logger.info("🔧 Подготавливаем данные для обучения...")
        
        # Выбираем технические признаки
        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'RSI', 
            'MACD', 'ATR', 'OBV', 'Price_change', 'Volatility',
            'BB_width', 'Volume_ratio'
        ]
        
        # Убираем NaN
        df = tech_df[tech_features].dropna()
        
        # Создаем PricePredictor для подготовки данных
        predictor = PricePredictor()
        
        # Если есть новости, подготавливаем их
        if news_df is not None and not news_df.empty:
            self.logger.info("📰 Интегрируем новостные данные...")
            X_tech, X_news, y = predictor.prepare_training_data(df, news_df)
        else:
            self.logger.info("⚠️ Обучаем только на технических данных")
            # Упрощенная подготовка без новостей
            X_tech, y = self._prepare_tech_only_data(df, predictor)
            X_news = None
        
        self.logger.info(f"✅ Подготовлено {len(X_tech)} последовательностей")
        self.logger.info(f"   Размер последовательности: {X_tech.shape[1]}")
        self.logger.info(f"   Количество признаков: {X_tech.shape[2]}")
        
        return X_tech, X_news, y, predictor
    
    def _prepare_tech_only_data(self, df, predictor):
        """Подготовка только технических данных"""
        # Нормализация
        features = predictor.technical_scaler.fit_transform(df.values)
        target = predictor.price_scaler.fit_transform(df[['Close']].values)
        
        # Создание последовательностей
        sequence_length = 30
        X, y = [], []
        
        for i in range(sequence_length, len(features) - 1):
            X.append(features[i-sequence_length:i])
            y.append(target[i + 1, 0])  # Предсказываем следующий день
        
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def train_gru_model(self, X_tech, X_news, y, predictor):
        """Тренировка GRU модели"""
        self.logger.info("🧠 Начинаем тренировку GRU модели...")
        
        # Настройки из конфига
        model_config = self.config.get_model_config()
        
        # Разделение данных
        train_size = int(0.7 * len(X_tech))
        val_size = int(0.15 * len(X_tech))
        
        X_tech_train = X_tech[:train_size]
        X_tech_val = X_tech[train_size:train_size + val_size]
        X_tech_test = X_tech[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        if X_news is not None:
            X_news_train = X_news[:train_size]
            X_news_val = X_news[train_size:train_size + val_size]
            X_news_test = X_news[train_size + val_size:]
        else:
            # Создаем фиктивные новостные данные
            X_news_train = torch.zeros(len(X_tech_train), 5)
            X_news_val = torch.zeros(len(X_tech_val), 5)
            X_news_test = torch.zeros(len(X_tech_test), 5)
        
        self.logger.info(f"📊 Размеры данных:")
        self.logger.info(f"   Train: {len(X_tech_train)}")
        self.logger.info(f"   Val: {len(X_tech_val)}")
        self.logger.info(f"   Test: {len(X_tech_test)}")
        
        # Обучаем модель
        predictor.model = None  # Сбрасываем модель
        train_losses, val_losses = predictor.train(
            X_tech_train, X_news_train, y_train,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size']
        )
        
        # Оценка на тестовых данных
        self.evaluate_model(predictor, X_tech_test, X_news_test, y_test)
        
        return predictor, train_losses, val_losses
    
    def evaluate_model(self, predictor, X_tech_test, X_news_test, y_test):
        """Оценка модели"""
        self.logger.info("📈 Оценка модели на тестовых данных...")
        
        predictor.model.eval()
        with torch.no_grad():
            X_tech_test = X_tech_test.to(self.device)
            X_news_test = X_news_test.to(self.device)
            y_test = y_test.to(self.device)
            
            predictions = predictor.model(X_tech_test, X_news_test)
            
            # Метрики
            mse = torch.nn.functional.mse_loss(predictions.squeeze(), y_test)
            mae = torch.nn.functional.l1_loss(predictions.squeeze(), y_test)
            
            # Обратная трансформация для интерпретации
            y_test_orig = predictor.price_scaler.inverse_transform(
                y_test.cpu().numpy().reshape(-1, 1)
            )
            pred_orig = predictor.price_scaler.inverse_transform(
                predictions.cpu().numpy()
            )
            
            # Directional accuracy
            y_diff = np.diff(y_test_orig.flatten())
            pred_diff = np.diff(pred_orig.flatten())
            dir_acc = np.mean(np.sign(y_diff) == np.sign(pred_diff)) * 100
            
            self.logger.info(f"✅ Результаты на тестовых данных:")
            self.logger.info(f"   MSE: {mse:.6f}")
            self.logger.info(f"   MAE: {mae:.6f}")
            self.logger.info(f"   MAE в $: ${np.mean(np.abs(y_test_orig - pred_orig)):.2f}")
            self.logger.info(f"   Directional Accuracy: {dir_acc:.1f}%")
    
    def save_model(self, predictor, metrics=None):
        """Сохранение модели"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(f"models/eth_gru_model_{timestamp}.pth")
        
        save_dict = {
            'model_state_dict': predictor.model.state_dict(),
            'technical_scaler': predictor.technical_scaler,
            'price_scaler': predictor.price_scaler,
            'config': self.config.get_model_config(),
            'timestamp': timestamp,
            'device': str(self.device),
            'metrics': metrics
        }
        
        torch.save(save_dict, model_path)
        self.logger.info(f"💾 Модель сохранена: {model_path}")
        
        # Сохраняем как последнюю модель
        latest_path = Path("models/eth_gru_model_latest.pth")
        torch.save(save_dict, latest_path)
        self.logger.info(f"💾 Обновлена последняя модель: {latest_path}")
        
        return model_path
    
    def plot_training_history(self, train_losses, val_losses):
        """Визуализация процесса обучения"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.title('GRU Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Сохраняем график
        plot_path = Path("logs/training/training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 График обучения сохранен: {plot_path}")

async def main():
    """Основная функция тренировки"""
    parser = argparse.ArgumentParser(description='Train ETH GRU Model')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days of historical data to use')
    parser.add_argument('--skip-telegram', action='store_true',
                       help='Skip Telegram data collection')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs from config')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════╗
    ║     ETH GRU Model Training             ║
    ║       Mac M3 Pro Optimized             ║
    ╚════════════════════════════════════════╝
    """)
    
    trainer = GRUTrainer()
    
    try:
        # 1. Сбор данных
        print("\n📊 Этап 1: Сбор данных")
        print("=" * 50)
        
        # Технические данные
        tech_df = trainer.prepare_technical_data(days_back=args.days_back * 2)
        
        # Telegram данные
        news_df = None
        if not args.skip_telegram:
            try:
                news_df = await trainer.collect_telegram_data(days_back=args.days_back)
            except Exception as e:
                trainer.logger.warning(f"⚠️ Не удалось собрать данные Telegram: {e}")
                trainer.logger.info("Продолжаем только с техническими данными...")
        
        # 2. Подготовка данных
        print("\n🔧 Этап 2: Подготовка данных")
        print("=" * 50)
        
        X_tech, X_news, y, predictor = trainer.create_training_dataset(
            tech_df, news_df
        )
        
        # 3. Тренировка модели
        print("\n🧠 Этап 3: Тренировка GRU модели")
        print("=" * 50)
        
        if args.epochs:
            # Переопределяем количество эпох
            config.env_config['model']['epochs'] = args.epochs
        
        predictor, train_losses, val_losses = trainer.train_gru_model(
            X_tech, X_news, y, predictor
        )
        
        # 4. Сохранение результатов
        print("\n💾 Этап 4: Сохранение результатов")
        print("=" * 50)
        
        # Сохраняем модель
        model_path = trainer.save_model(predictor, {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        })
        
        # Визуализация
        trainer.plot_training_history(train_losses, val_losses)
        
        print("\n✅ Тренировка завершена успешно!")
        print(f"📊 Финальная train loss: {train_losses[-1]:.6f}")
        print(f"📊 Финальная val loss: {val_losses[-1]:.6f}")
        print(f"💾 Модель сохранена: {model_path}")
        
        # 5. Быстрый тест предсказания
        print("\n🔮 Тестовое предсказание на завтра:")
        print("=" * 50)
        
        # Берем последние данные
        last_sequence = X_tech[-1:].to(trainer.device)
        last_news = X_news[-1:].to(trainer.device) if X_news is not None else torch.zeros(1, 5).to(trainer.device)
        
        predictor.model.eval()
        with torch.no_grad():
            prediction = predictor.model(last_sequence, last_news)
            pred_price = predictor.price_scaler.inverse_transform(
                prediction.cpu().numpy()
            )[0, 0]
            
            current_price = tech_df['Close'].iloc[-1]
            change_pct = ((pred_price - current_price) / current_price) * 100
            
            print(f"💰 Текущая цена ETH: ${current_price:.2f}")
            print(f"📈 Прогноз на завтра: ${pred_price:.2f}")
            print(f"📊 Ожидаемое изменение: {change_pct:+.2f}%")
        
    except Exception as e:
        trainer.logger.error(f"❌ Ошибка при тренировке: {e}")
        raise

if __name__ == "__main__":
    # Для Mac M3 Pro оптимизация
    if torch.backends.mps.is_available():
        # Включаем оптимизации для Metal
        torch.backends.mps.allow_tf32 = True
    
    asyncio.run(main())