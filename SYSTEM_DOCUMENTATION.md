# ETH Analysis System - Documentation

## 1. Overview

The ETH Analysis System is a Python-based application designed to collect, analyze, and predict Ethereum (ETH) price movements. It integrates technical analysis, news sentiment analysis from Telegram channels, and machine learning models (currently a GRU-based neural network and a Random Forest model) to generate trading insights and signals.

## 2. System Architecture

The system is modular, consisting of several key Python scripts and components:

*   **Configuration**:
    *   `config_loader.py`: Loads settings from `.env` (environment variables) and `config.yaml`. Manages global configuration access.
    *   `.env`: Stores sensitive information (API keys, phone numbers) and environment-specific settings.
    *   `config.yaml`: Stores detailed configuration for different modules (Telegram parsing, technical analysis, news analysis, model architecture, training parameters, output settings, etc.).
*   **Core Logic**:
    *   `eth_analysis_system.py`: Defines the main classes and logic for the system:
        *   `TechnicalIndicators`: Manual implementations of TA indicators (SMA, EMA, RSI, MACD, BB, ATR, OBV, etc.).
        *   `TelegramParser`: Connects to Telegram (using Telethon) and fetches messages.
        *   `NewsRelevanceClassifier`: Uses a BERT model to classify news relevance and perform basic keyword-based sentiment/impact analysis.
        *   `TechnicalAnalyzer`: Fetches ETH price data (using yfinance) and calculates technical indicators.
        *   `ETHPricePredictor`: A GRU-based neural network model for price prediction (PyTorch).
        *   `PricePredictor`: A wrapper for training and using the `ETHPricePredictor` model, including data preparation.
        *   `ETHAnalysisSystem`: Orchestrates the overall analysis pipeline (technical analysis, news collection, signal generation).
*   **Main Execution Scripts**:
    *   `main.py`: The main entry point for running a full analysis cycle. It uses `ETHAnalysisSystem` to perform analysis, saves reports, and can send notifications. Supports scheduled runs.
    *   `train_model.py`: Script for training the GRU model (`ETHPricePredictor`). Handles data collection (technical & news), preprocessing, model training, evaluation, and saving.
    *   `train_random_forest.py`: Script for training a Random Forest model. Handles data collection (technical & news, including BTC correlation and time features), preprocessing, model training with GridSearchCV, evaluation, and saving.
*   **Utility Scripts**:
    *   `collect_and_save_news.py`: Likely for ad-hoc news collection and saving.
    *   `analyze_news_unsupervised.py`: For unsupervised analysis of news data (e.g., clustering).
    *   `find_channels_id.py`: Utility to find Telegram channel IDs.
    *   `setup.py`: Standard Python package setup file.
*   **Data Storage**:
    *   `data/raw/`: Stores raw fetched data (e.g., cached Telegram news, technical data CSVs).
    *   `data/processed/`: Stores processed data (e.g., `clustered_news_sbert_kmeans.csv`).
*   **Model Storage**:
    *   `models/`: Stores trained machine learning models (e.g., `.pth` for PyTorch GRU, `.joblib` for Random Forest).
*   **Reporting & Logging**:
    *   `reports/`: Stores generated analysis reports (JSON, HTML charts).
    *   `logs/`: Stores log files, including training history images.
    *   `eth_analysis.log`: Main application log file.

## 3. Data Flow (Typical Analysis Run via `main.py`)

1.  **Initialization (`ETHAnalysisRunner` in `main.py`):**
    *   Configuration is loaded.
    *   `ETHAnalysisSystem` is instantiated.
2.  **`ETHAnalysisSystem.run_full_analysis()` is called:**
    *   **Technical Data Collection & Analysis (`TechnicalAnalyzer`):** Fetches ETH price data, calculates indicators, detects patterns.
    *   **Trading Signal Generation (`ETHAnalysisSystem.generate_trading_signals()`):** Uses technicals and patterns for rule-based signals.
    *   **News Data Collection & Analysis (Optional):** Parses Telegram, analyzes sentiment/impact.
    *   **Report Aggregation:** Combines results into a `report` dictionary.
3.  **Post-Analysis (`ETHAnalysisRunner` in `main.py`):** Saves report, sends notifications, displays console summary.

## 4. Model Training Flow

### 4.1. GRU Model (`train_model.py`)

1.  **Data Collection**: Fetches/loads technical and (optional) news data.
2.  **Data Preparation**: Selects features, aggregates news, scales data, creates sequences for RNN. Uses dummy news if real news is skipped.
3.  **Training**: Initializes `ETHPricePredictor`, splits data, trains using PyTorch loop (AdamW, ReduceLROnPlateau, HuberLoss).
4.  **Evaluation**: Calculates MSE, MAE, directional accuracy on test set.
5.  **Saving**: Saves model, scalers, config, metrics; plots training history.

### 4.2. Random Forest Model (`train_random_forest.py`)

1.  **Data Collection**: Fetches/loads ETH technicals, calculates indicators (VWAP, RSI, BB, ATR via `pandas_ta`), custom signals (`VWAPSignal`, `TotalSignal`), BTC data, ETH-BTC correlation, time features. Optionally fetches news.
2.  **Data Preparation**: Aggregates news per hour (currently zeroed out to focus on technicals). Joins features. Defines target (1-hour price direction). Handles NaNs.
3.  **Training & Evaluation**: Splits data, uses `RandomForestClassifier` with `GridSearchCV` for tuning. Evaluates (accuracy, classification report, confusion matrix), shows feature importances.
4.  **Saving**: Saves model using `joblib`.

## 5. Dependencies (`requirements.txt`)

*   **Core ML/DL**: `torch>=2.0.0`, `transformers>=4.3.0`, `sentence-transformers>=2.2.0`, `scikit-learn>=1.3.0`, `optuna>=3.2.0`
*   **Telegram**: `telethon>=1.28.0`
*   **Data Handling**: `pandas>=2.0.0`, `numpy~=1.24.0`, `python-dateutil>=2.8.2`, `nltk>=3.8.1`
*   **Financial Data & TA**: `yfinance>=0.2.18`, `pandas-ta>=0.3.14b`, `TA-Lib>=0.4.25` (optional)
*   **Visualization**: `plotly>=5.14.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.2`
*   **Async**: `aiohttp>=3.8.4`, `asyncio-throttle>=1.0.2`
*   **Utilities**: `python-dotenv>=1.0.0`, `pyyaml>=6.0`, `tqdm>=4.65.0`, `colorlog>=6.7.0`
*   (Other optional dependencies for DB, API, Testing, Linting)

## 6. Key Files and Their Roles

(As listed in section 2)

## 7. Potential Future Upgrades and Scalability

*   **Enhanced News Analysis**: Twitter integration, advanced NLP for sentiment/topic/event detection, multi-source news aggregation.
*   **Advanced Financial Data**: Whale wallet tracking, order book data, derivatives data, macroeconomic data.
*   **Model Improvements**: Extensive hyperparameter optimization, proper time series CV, model ensembling, advanced feature engineering, online learning/retraining pipelines.
*   **Backtesting Framework**: Robust engine with detailed metrics and strategy parameterization.
*   **Scalability & Performance**: Optimized database usage (time-series DBs), distributed task queues (Celery), scalable API service (Docker, Kubernetes), expanded caching.
*   **User Interface**: Develop a dynamic web dashboard for real-time monitoring, configuration, and visualization.

## 8. Real-time HTML Interface (Conceptual)

A real-time HTML dashboard would require:
*   **Backend Modifications**: Python scripts emitting status/data via WebSockets, message queues, or a database. A web server (FastAPI/Flask) to serve data and handle WebSocket connections.
*   **Frontend Development**: HTML for structure, CSS for styling, JavaScript for dynamic updates (consuming data from backend), interactivity, and visualizations. Frameworks like React/Vue could be beneficial.
*   **"AI-Driven" Aspects**: Would need specific AI logic for intelligent display or summarization, separate from the core analysis models.

(The basic HTML outline provided previously can serve as a very rough starting point for layout ideas.)

---

This documentation provides a high-level overview. Each component could be documented in much greater detail.
