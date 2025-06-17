"""
Configuration loader for ETH Analysis System
Loads settings from .env and config.yaml files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.env_config = self._load_env()
        self.yaml_config = self._load_yaml()
        self.logger = self._setup_logging()
        
    def _load_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            # Telegram
            'telegram': {
                'api_id': os.getenv('TELEGRAM_API_ID'),
                'api_hash': os.getenv('TELEGRAM_API_HASH'),
                'phone': os.getenv('TELEGRAM_PHONE'),
                'channels': [ch.strip() for ch in os.getenv('TELEGRAM_CHANNELS', '').split(',') if ch.strip()]
            },
            
            # Database
            'database': {
                'url': os.getenv('DATABASE_URL', 'sqlite:///eth_analysis.db')
            },
            
            # Model
            'model': {
                'sequence_length': int(os.getenv('MODEL_SEQUENCE_LENGTH', 60)),
                'prediction_days': int(os.getenv('MODEL_PREDICTION_DAYS', 7)),
                'batch_size': int(os.getenv('MODEL_BATCH_SIZE', 32)),
                'epochs': int(os.getenv('MODEL_EPOCHS', 50)),
                'learning_rate': float(os.getenv('MODEL_LEARNING_RATE', 0.001))
            },
            
            # Analysis
            'analysis': {
                'days_back': int(os.getenv('ANALYSIS_DAYS_BACK', 30)),
                'min_news_impact': float(os.getenv('ANALYSIS_MIN_NEWS_IMPACT', 2.0)),
                'confidence_threshold': float(os.getenv('ANALYSIS_CONFIDENCE_THRESHOLD', 0.6))
            },
            
            # API
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', 8000)),
                'workers': int(os.getenv('API_WORKERS', 4))
            },
            
            # Logging
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': os.getenv('LOG_FILE', 'eth_analysis.log')
            },
            
            # Notifications
            'notifications': {
                'enabled': os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true',
                'webhook': os.getenv('NOTIFICATION_WEBHOOK'),
                'threshold': os.getenv('NOTIFICATION_THRESHOLD', 'high')
            }
        }
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = self.env_config['logging']['level']
        log_file = self.env_config['logging']['file']
        
        # Create logger
        logger = logging.getLogger('ETHAnalysis')
        logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        # First check env config
        keys = key.split('.')
        value = self.env_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Fall back to yaml config
                value = self.yaml_config
                for k2 in keys:
                    if isinstance(value, dict) and k2 in value:
                        value = value[k2]
                    else:
                        return default
                break
        
        return value
    
    def get_telegram_config(self) -> Dict[str, str]:
        """Get Telegram configuration"""
        return {
            'api_id': self.env_config['telegram']['api_id'],
            'api_hash': self.env_config['telegram']['api_hash'],
            'phone': self.env_config['telegram']['phone']
        }
    
    def get_telegram_channels(self) -> List[str]:
        """Get list of Telegram channels"""
        return self.env_config['telegram']['channels']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        yaml_model = self.yaml_config.get('model', {})
        env_model = self.env_config['model']
        
        return {
            'sequence_length': env_model['sequence_length'],
            'prediction_days': env_model['prediction_days'],
            'batch_size': env_model['batch_size'],
            'epochs': env_model['epochs'],
            'learning_rate': env_model['learning_rate'],
            'architecture': yaml_model.get('architecture', {}),
            'training': yaml_model.get('training', {})
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        yaml_analysis = self.yaml_config.get('analysis', {})
        env_analysis = self.env_config['analysis']
        
        return {
            'days_back': env_analysis['days_back'],
            'min_news_impact': env_analysis['min_news_impact'],
            'confidence_threshold': env_analysis['confidence_threshold'],
            'technical': yaml_analysis.get('technical', {}),
            'news': yaml_analysis.get('news', {})
        }
    
    def validate_config(self) -> bool:
        """Validate required configuration"""
        required_env_vars = [
            'TELEGRAM_API_ID',
            'TELEGRAM_API_HASH',
            'TELEGRAM_PHONE'
        ]
        
        missing = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            self.logger.error(f"Missing required environment variables: {missing}")
            return False
        
        if not self.get_telegram_channels():
            self.logger.error("No Telegram channels specified in TELEGRAM_CHANNELS")
            return False
        
        return True
    
    def save_example_env(self, path: str = ".env.example"):
        """Save example .env file"""
        example_content = """# Telegram API credentials
# Get these from https://my.telegram.org
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE=+1234567890

# Telegram channels to parse (comma-separated)
TELEGRAM_CHANNELS=@eth_news,@crypto_signals,@defi_updates

# Database configuration (optional)
DATABASE_URL=sqlite:///eth_analysis.db

# Model configuration
MODEL_SEQUENCE_LENGTH=60
MODEL_PREDICTION_DAYS=7
MODEL_BATCH_SIZE=32
MODEL_EPOCHS=50
MODEL_LEARNING_RATE=0.001

# Analysis settings
ANALYSIS_DAYS_BACK=30
ANALYSIS_MIN_NEWS_IMPACT=2.0
ANALYSIS_CONFIDENCE_THRESHOLD=0.6

# Logging
LOG_LEVEL=INFO
LOG_FILE=eth_analysis.log
"""
        
        with open(path, 'w') as f:
            f.write(example_content)
        
        self.logger.info(f"Example .env file saved to {path}")

# Global config instance
config = Config()

# Example usage in main system
if __name__ == "__main__":
    # Validate configuration
    if not config.validate_config():
        print("❌ Configuration validation failed!")
        print("Please check your .env file")
        config.save_example_env()
        exit(1)
    
    # Get configurations
    telegram_config = config.get_telegram_config()
    channels = config.get_telegram_channels()
    model_config = config.get_model_config()
    
    print("✅ Configuration loaded successfully!")
    print(f"📱 Telegram channels: {channels}")
    print(f"🧠 Model epochs: {model_config['epochs']}")
    print(f"📊 Analysis days back: {config.get('analysis.days_back')}")