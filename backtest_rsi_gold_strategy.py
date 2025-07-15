#!/usr/bin/env python3
"""
Backtests an RSI Mean Reversion strategy for GOLD (XAU/USD) on 4-hour data using Capital.com API.
Strategy:
- Long: RSI(20) < 35, wait for first bullish confirmation candle, SL at candle low, TP = 4*Risk.
- Short: RSI(20) > 68, wait for first bearish confirmation candle, SL at candle high, TP = 4*Risk.
- Re-entries allowed if RSI condition persists after trade close.
"""
import pandas as pd
import pandas_ta as ta
import http.client # For Capital.com API
import json
import os
from dotenv import load_dotenv
from backtesting import Backtest, Strategy
# from backtesting.lib import crossover # Not used in this strategy
from pathlib import Path
from datetime import datetime, timedelta # Keep for default date ranges
import argparse
from typing import Dict, List, Optional

from main_files.config_loader import config 
logger = config.logger

load_dotenv() 

class CapitalComAPIAdapter:
    def __init__(self, api_key: str, identifier: str, password: str, demo_mode: bool = True):
        self.base_url = "demo-api-capital.backend-capital.com" if demo_mode else "api-capital.backend-capital.com"
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.security_token = None
        self.cst_token = None
        self._login_and_get_tokens()

    def _login_and_get_tokens(self) -> bool:
        try:
            logger.info(f"🔐 Logging in to Capital.com ({self.base_url})...")
            conn = http.client.HTTPSConnection(self.base_url)
            payload = json.dumps({"identifier": self.identifier, "password": self.password})
            headers = {'Content-Type': 'application/json', 'X-CAP-API-KEY': self.api_key}
            
            conn.request("POST", "/api/v1/session", payload, headers)
            response = conn.getresponse()
            data = response.read() # Read before accessing headers for some http.client versions
            
            logger.debug(f"Login response status: {response.status}")
            
            if response.status == 200:
                self.cst_token = response.getheader('CST')
                self.security_token = response.getheader('X-SECURITY-TOKEN')
                if self.cst_token and self.security_token:
                    logger.info("✅ Login successful! Tokens obtained.")
                    conn.close()
                    return True
                else:
                    logger.error(f"❌ Login successful but tokens not found. Headers: {dict(response.getheaders())}")
            else:
                logger.error(f"❌ Login failed with status {response.status}. Response: {data.decode('utf-8', errors='ignore')}")
            
            conn.close()
            raise ConnectionError("Failed to login and get authentication tokens from Capital.com")
            
        except Exception as e:
            logger.error(f"❌ Login error: {str(e)}")
            raise ConnectionError(f"Capital.com login error: {e}")

    def get_historical_ohlcv(self, 
                             epic: str,
                             resolution: str = "HOUR_4", # Capital.com format e.g. HOUR_4, DAY
                             from_date_str: Optional[str] = None, # YYYY-MM-DDTHH:MM:SS
                             to_date_str: Optional[str] = None,   # YYYY-MM-DDTHH:MM:SS
                             max_points: Optional[int] = None) -> pd.DataFrame:
        if not self.security_token or not self.cst_token:
            logger.error("❌ Not authenticated. Please login first.")
            raise PermissionError("Not authenticated with Capital.com")

        try:
            # Construct endpoint and params
            # The endpoint /api/v1/prices/{epic} is used based on capital_request.py
            # The /snapshots endpoint might be for single snapshots, not series.
            endpoint = f"/api/v1/prices/{epic}"
            params_list = [f"resolution={resolution}"]
            if from_date_str: params_list.append(f"from={from_date_str}")
            if to_date_str: params_list.append(f"to={to_date_str}")
            if max_points: params_list.append(f"max={max_points}")
            
            full_endpoint = f"{endpoint}?{'&'.join(params_list)}"
            
            logger.info(f"Fetching historical data for {epic} from {self.base_url}{full_endpoint}")
            
            conn = http.client.HTTPSConnection(self.base_url)
            headers = {
                'X-SECURITY-TOKEN': self.security_token,
                'CST': self.cst_token
                # 'Content-Type': 'application/json' # Not strictly needed for GET
            }
            
            conn.request("GET", full_endpoint, "", headers)
            response = conn.getresponse()
            raw_response_data = response.read()
            conn.close()
            
            logger.debug(f"Historical prices response status: {response.status}")

            if response.status == 200:
                response_data = json.loads(raw_response_data.decode("utf-8"))
                
                if 'prices' not in response_data or not response_data['prices']:
                    logger.warning(f"No 'prices' data found in response for {epic}. Response: {response_data}")
                    return pd.DataFrame()

                ohlcv_data = []
                for candle in response_data['prices']:
                    dt_str = candle.get('snapshotTime') # Assuming 'snapshotTime' based on typical API responses
                    if not dt_str: continue # Skip if no timestamp

                    # Try to parse datetime, handling potential 'Z' for UTC
                    if dt_str.endswith('Z'): dt_str = dt_str[:-1] # Remove Z if present
                    try:
                        dt = pd.to_datetime(dt_str, errors='coerce')
                        if pd.isna(dt): 
                            logger.warning(f"Could not parse datetime: {candle.get('snapshotTime')}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error parsing datetime {candle.get('snapshotTime')}: {e}")
                        continue
                    
                    # Helper to extract price, preferring 'bid' if available
                    def get_price(price_field_data):
                        if isinstance(price_field_data, dict):
                            return price_field_data.get('bid', price_field_data.get('ask')) # Fallback to ask
                        return price_field_data

                    open_price = get_price(candle.get('openPrice'))
                    high_price = get_price(candle.get('highPrice'))
                    low_price = get_price(candle.get('lowPrice'))
                    close_price = get_price(candle.get('closePrice'))
                    volume = candle.get('lastTradedVolume', 0) # Default to 0 if not present

                    if all(v is not None for v in [open_price, high_price, low_price, close_price]):
                        ohlcv_data.append([dt, float(open_price), float(high_price), float(low_price), float(close_price), float(volume)])
                
                if not ohlcv_data:
                    logger.warning(f"Parsed OHLCV data is empty for {epic}. Raw prices received: {len(response_data['prices'])}")
                    return pd.DataFrame()

                df = pd.DataFrame(ohlcv_data, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df.set_index('Datetime', inplace=True)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep='first')] # Remove duplicate index entries if any
                return df
            else:
                logger.error(f"API Error fetching prices: Status {response.status}, Response: {raw_response_data.decode('utf-8', errors='ignore')}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching or parsing price data for {epic}: {str(e)}", exc_info=True)
            return pd.DataFrame()

def fetch_formatted_data(epic="GOLD", interval="4h", days_back=729, demo_mode=True):
    api_key = os.getenv("CAPITAL_HEADER_API_KEY") # Key for X-CAP-API-KEY header
    identifier = os.getenv("CAPITAL_IDENTIFIER") # Login email/username
    password = os.getenv("CAPITAL_PASSWORD")

    if not all([api_key, identifier, password]):
        raise ValueError("Missing Capital.com credentials in .env file (CAPITAL_HEADER_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD).")

    client = CapitalComAPIAdapter(api_key=api_key, identifier=identifier, password=password, demo_mode=demo_mode)
    
    # Define date range for fetching data
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=days_back)
    
    # Capital.com API expects specific string formats for dates and resolution
    # Resolution mapping for Capital.com
    resolution_map_capital = {
        "1m": "MINUTE", "5m": "MINUTE_5", "15m": "MINUTE_15", "30m": "MINUTE_30",
        "1h": "HOUR", "4h": "HOUR_4", "1d": "DAY", "1w": "WEEK", "1M": "MONTH"
    }
    capital_resolution = resolution_map_capital.get(interval.lower(), "HOUR_4") # Default to 4H

    # Date format: YYYY-MM-DDTHH:MM:SS (seems to be what their API might prefer for query params)
    # The API might also accept just YYYY-MM-DD for daily resolution.
    # This needs to be confirmed from Java samples if issues persist.
    from_date_str = from_date.strftime("%Y-%m-%dT%H:%M:%S")
    to_date_str = to_date.strftime("%Y-%m-%dT%H:%M:%S")

    df = client.get_historical_ohlcv(
        epic=epic, 
        resolution=capital_resolution,
        from_date_str=from_date_str,
        to_date_str=to_date_str
        # max_points could be added if 'from'/'to' is not sufficient or for capping
    )

    if df.empty:
        raise ValueError(f"No data fetched for {epic} from Capital.com with specified parameters.")
    
    df['RSI'] = ta.rsi(df['Close'], length=20)
    df.dropna(inplace=True)
    logger.info(f"Data for {epic} fetched and RSI calculated. Shape: {df.shape}")
    return df

class RsiMeanReversionGold(Strategy):
    rsi_period = 20 
    rsi_oversold = 35
    rsi_overbought = 68
    tp_rr = 4.0 

    waiting_for_long_confirmation = False
    waiting_for_short_confirmation = False

    def init(self):
        pass 

    def next(self):
        if len(self.data.Close) < self.rsi_period: # Ensure enough data for RSI
            return

        current_rsi = self.data.RSI[-1]
        current_close = self.data.Close[-1]
        current_open = self.data.Open[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        is_bullish_candle = current_close > current_open
        is_bearish_candle = current_close < current_open

        if not self.position and not self.waiting_for_long_confirmation and current_rsi < self.rsi_oversold:
            self.waiting_for_long_confirmation = True
            # logger.debug(f"{self.data.index[-1]}: RSI < {self.rsi_oversold} ({current_rsi:.2f}). Waiting for bullish confirmation.")

        if self.waiting_for_long_confirmation and is_bullish_candle:
            entry_price = current_close 
            sl_price = current_low     
            risk = entry_price - sl_price
            if risk <= 0 or pd.isna(risk): 
                self.waiting_for_long_confirmation = False 
                return
            tp_price = entry_price + (risk * self.tp_rr)
            # logger.info(f"{self.data.index[-1]}: LONG ENTRY. RSI: {current_rsi:.2f}. E:{entry_price:.2f} SL:{sl_price:.2f} TP:{tp_price:.2f}")
            self.buy(sl=sl_price, tp=tp_price)
            self.waiting_for_long_confirmation = False 

        if not self.position and not self.waiting_for_short_confirmation and current_rsi > self.rsi_overbought:
            self.waiting_for_short_confirmation = True
            # logger.debug(f"{self.data.index[-1]}: RSI > {self.rsi_overbought} ({current_rsi:.2f}). Waiting for bearish confirmation.")

        if self.waiting_for_short_confirmation and is_bearish_candle:
            entry_price = current_close 
            sl_price = current_high    
            risk = sl_price - entry_price
            if risk <= 0 or pd.isna(risk): 
                self.waiting_for_short_confirmation = False 
                return
            tp_price = entry_price - (risk * self.tp_rr)
            # logger.info(f"{self.data.index[-1]}: SHORT ENTRY. RSI: {current_rsi:.2f}. E:{entry_price:.2f} SL:{sl_price:.2f} TP:{tp_price:.2f}")
            self.sell(sl=sl_price, tp=tp_price)
            self.waiting_for_short_confirmation = False 

        if self.waiting_for_long_confirmation and current_rsi >= self.rsi_oversold:
            self.waiting_for_long_confirmation = False
        
        if self.waiting_for_short_confirmation and current_rsi <= self.rsi_overbought:
            self.waiting_for_short_confirmation = False
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest RSI Mean Reversion Strategy for GOLD using Capital.com API")
    parser.add_argument("--epic", type=str, default="GOLD", help="Capital.com EPIC for Gold (e.g., GOLD, XAUUSD)")
    parser.add_argument("--days_back", type=int, default=729, help="Number of past days of data to fetch (e.g., 729 for ~2 years)")
    parser.add_argument("--interval", type=str, default="4h", help="Data interval (e.g., 1d, 4h, 1h, 5m)")
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash for backtest")
    parser.add_argument("--commission", type=float, default=0.0002, help="Commission per trade")
    parser.add_argument("--demo", action='store_true', help="Use Capital.com demo environment")
    parser.add_argument("--no-demo", dest='demo', action='store_false', help="Use Capital.com live environment")
    parser.set_defaults(demo=True)
    
    args = parser.parse_args()

    try:
        data = fetch_formatted_data(epic=args.epic, interval=args.interval, days_back=args.days_back, demo_mode=args.demo)
        
        if data.empty or len(data) < 50: # Need some data to run a meaningful backtest
            logger.error("Not enough data fetched from Capital.com, cannot run backtest.")
        else:
            logger.info(f"Running backtest for {args.epic} from {data.index[0]} to {data.index[-1]} using Capital.com data...")
            bt = Backtest(data, RsiMeanReversionGold, cash=args.cash, commission=args.commission)
            stats = bt.run()
            
            logger.info("\n--- Capital.com Backtest Stats ---")
            logger.info(stats)
            
            logger.info(f"\nReturn [%]: {stats['Return [%]']:.2f}")
            logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
            logger.info(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
            logger.info(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}")
            logger.info(f"# Trades: {stats['# Trades']}")

            plot_filename = f"reports/backtest_capital_rsi_{args.epic}_{args.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path("reports").mkdir(parents=True, exist_ok=True)
            bt.plot(filename=plot_filename, open_browser=False)
            logger.info(f"📈 Backtest plot saved to {plot_filename}")

    except ValueError as e:
        logger.error(f"ValueError during backtest setup or run: {e}")
    except ConnectionError as e: # Catch specific connection/auth errors
        logger.error(f"Connection or Authentication Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
