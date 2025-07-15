# -*- coding: utf-8 -*-
# solana_dashboard.py
import asyncio
import logging
import os
import time
import json
from collections import defaultdict
from typing import Dict, Set, Optional, List, Any
import datetime

# --- UI & Threading Libraries ---
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QListWidget, QListWidgetItem, QPlainTextEdit, QGridLayout, QFrame,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
                             QDialog, QSpinBox, QDialogButtonBox)
from PyQt6.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt6.QtGui import QColor, QFont
import webbrowser

# --- Solana & Telegram Libraries ---
import telegram
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
from solders.signature import Signature

# --- Helper Library ---
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WALLETS_FILE = "wallets.json"
POLLING_INTERVAL = 15
SIGNATURE_FETCH_LIMIT = 20
MAX_TRANSACTION_AGE = 300
PROCESSED_SIGNATURES_MAX_SIZE = 2000
SOLSCAN_BASE_URL = "https://solscan.io/tx/"

KNOWN_DEX_PROGRAM_IDS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", "routeXXXXXXX11111111111111111111111111111111",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3ekubTQ", "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
}
KNOWN_TOKENS = {
    "So11111111111111111111111111111111111111112": {"symbol": "SOL", "decimals": 9},
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {"symbol": "USDC", "decimals": 6},
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": {"symbol": "USDT", "decimals": 6},
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Models & Logic ---

class PaperPortfolio:
    """Manages the state of the paper trading portfolio."""
    def __init__(self, initial_usd_value=10000.0):
        self.balances: Dict[str, float] = {"USDC": initial_usd_value}
    def update_from_trade(self, balance_changes: Dict[str, Dict[str, Any]]):
        for change_info in balance_changes.values():
            symbol = change_info["symbol"]
            change_amount = change_info["change"]
            self.balances[symbol] = self.balances.get(symbol, 0.0) + change_amount
            if abs(self.balances[symbol]) < 1e-9: del self.balances[symbol]
    def get_display_string(self) -> str:
        lines = ["\n*Paper Portfolio Balances:*"]
        sorted_symbols = sorted(self.balances.keys(), key=lambda s: (s not in ["USDC", "SOL"], s))
        for symbol in sorted_symbols:
            lines.append(f"  `{self.balances[symbol]:,.8f}` {escape_markdown_v2(symbol)}")
        return "\n".join(lines)

# --- Solana Parsing & Helper Functions ---
# Note: These are integrated directly into the script for self-containment.

def load_wallets_from_json(file_path: str) -> Dict[str, str]:
    try:
        with open(file_path, 'r') as f:
            return {item['name']: item['address'] for item in json.load(f)}
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}!.'
    return "".join(f'\\{char}' if char in escape_chars else char for char in str(text))

async def get_transaction_details(client: AsyncClient, signature: Signature) -> Optional[Any]:
    for _ in range(3):
        try:
            resp = await client.get_transaction(signature, encoding="jsonParsed", max_supported_transaction_version=0)
            if resp.value: return resp.value
            await asyncio.sleep(0.5)
        except Exception: pass
    return None

def parse_balance_changes(tx_result: Any, wallet_pubkey: Pubkey) -> Dict[str, Dict[str, Any]]:
    changes = {}
    meta = getattr(tx_result.transaction, 'meta', None)
    if not meta: return changes
    pre_b, post_b = meta.pre_token_balances or [], meta.post_token_balances or []
    pre_map = {str(b.mint): b.ui_token_amount for b in pre_b if b.owner == wallet_pubkey}
    post_map = {str(b.mint): b.ui_token_amount for b in post_b if b.owner == wallet_pubkey}
    for mint in set(pre_map.keys()) | set(post_map.keys()):
        pre_amt = int(getattr(pre_map.get(mint), 'amount', '0'))
        post_amt = int(getattr(post_map.get(mint), 'amount', '0'))
        if post_amt - pre_amt != 0:
            info = KNOWN_TOKENS.get(str(mint), {"symbol": f"{str(mint)[:4]}...", "decimals": 9})
            changes[str(mint)] = {"change": (post_amt - pre_amt) / (10 ** info['decimals']), "symbol": info['symbol']}
    # SOL Balance (simplified)
    try:
        acc_keys = tx_result.transaction.transaction.message.account_keys
        wallet_idx = acc_keys.index(wallet_pubkey)
        sol_change = meta.post_balances[wallet_idx] - meta.pre_balances[wallet_idx]
        if sol_change != 0:
            changes["SOL"] = {"change": sol_change / 1e9, "symbol": "SOL"}
    except (ValueError, IndexError):
        pass
    return changes

def detect_dex_interaction(tx_result: Any) -> bool:
    try:
        for inst in tx_result.transaction.transaction.message.instructions:
            if str(inst.program_id) in KNOWN_DEX_PROGRAM_IDS: return True
    except AttributeError: return False
    return False

def format_swap_details(balance_changes: Dict[str, Dict[str, Any]]) -> Optional[str]:
    sold = next((v for v in balance_changes.values() if v['change'] < 0), None)
    bought = next((v for v in balance_changes.values() if v['change'] > 0), None)
    if sold and bought: return f"Swapped {abs(sold['change']):.4f} {sold['symbol']} for {bought['change']:.4f} {bought['symbol']}"
    return None

# --- Worker Threads for Background Tasks ---

class MonitorWorker(QObject):
    """Worker for real-time monitoring."""
    new_trade_signal = pyqtSignal(dict)
    
    def __init__(self, client, wallets):
        super().__init__()
        self.client = client
        self.wallets = wallets
        self.is_running = True
        self.processed_signatures = defaultdict(set)

    def run(self):
        asyncio.run(self.main_loop())

    async def main_loop(self):
        tasks = [self.monitor_wallet(name, address) for name, address in self.wallets.items()]
        await asyncio.gather(*tasks)

    async def monitor_wallet(self, name, address):
        wallet_pubkey = Pubkey.from_string(address)
        while self.is_running:
            try:
                current_time = time.time()
                signatures = (await self.client.get_signatures_for_address(wallet_pubkey, limit=SIGNATURE_FETCH_LIMIT)).value
                if signatures:
                    for sig_data in reversed(signatures):
                        sig = sig_data.signature
                        if sig in self.processed_signatures[address]: continue
                        self.processed_signatures[address].add(sig)

                        if sig_data.block_time and (current_time - sig_data.block_time <= MAX_TRANSACTION_AGE):
                            tx = await get_transaction_details(self.client, sig)
                            if tx and detect_dex_interaction(tx) and not getattr(getattr(tx.transaction, 'meta', None), 'err', None):
                                changes = parse_balance_changes(tx, wallet_pubkey)
                                details = format_swap_details(changes)
                                if details:
                                    self.new_trade_signal.emit({"wallet_name": name, "details": details, "changes": changes})
            except Exception as e:
                logger.error(f"Error in monitor for {name}: {e}")
            await asyncio.sleep(POLLING_INTERVAL)

    def stop(self):
        self.is_running = False

class AnalyzerWorker(QObject):
    """Worker for historical analysis with corrected logic."""
    analysis_complete = pyqtSignal(str, list) # NEW: Sends report AND list of trades
    progress_update = pyqtSignal(int, str)

    def __init__(self, client, wallet_address, limit=None):
        super().__init__()
        self.client = client
        self.wallet_address = wallet_address
        self.limit = limit
    
    def run(self): asyncio.run(self.run_analysis())

    async def _get_all_signatures(self) -> List[Any]:
        all_signatures = []
        last_signature = None
        while True:
            try:
                self.progress_update.emit(0, f"Fetching signature batch... (found {len(all_signatures)})")
                batch = (await self.client.get_signatures_for_address(self.wallet_pubkey, before=last_signature, limit=1000)).value
                if not batch: break
                all_signatures.extend(batch)
                last_signature = batch[-1].signature
                if self.limit and len(all_signatures) >= self.limit:
                    return all_signatures[:self.limit]
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Stopping signature fetch: {e}"); break
        return all_signatures

    async def run_analysis(self):
        self.wallet_pubkey = Pubkey.from_string(self.wallet_address)
        signatures_data = await self._get_all_signatures()
        if not signatures_data:
            self.analysis_complete.emit("No transactions found for this wallet.", [])
            return

        all_swaps = []
        for i, sig_data in enumerate(signatures_data):
            self.progress_update.emit(int((i + 1) / len(signatures_data) * 100), f"Analyzing tx {i+1}/{len(signatures_data)}")
            tx = await get_transaction_details(self.client, sig_data.signature)
            if tx and detect_dex_interaction(tx) and not getattr(getattr(tx.transaction, 'meta', None), 'err', None):
                changes = parse_balance_changes(tx, self.wallet_pubkey)
                sold = next((v for v in changes.values() if v['change'] < 0), None)
                bought = next((v for v in changes.values() if v['change'] > 0), None)
                if sold and bought:
                    all_swaps.append({
                        "timestamp": datetime.datetime.fromtimestamp(tx.block_time, tz=datetime.timezone.utc),
                        "signature": str(sig_data.signature),
                        "sold_symbol": sold["symbol"], "sold_amount": abs(sold["change"]),
                        "bought_symbol": bought["symbol"], "bought_amount": bought["change"],
                    })
        
        # --- NEW: Corrected P&L Calculation ---
        pnl_by_token = defaultdict(lambda: {"profit": 0, "loss": 0, "wins": 0, "losses": 0})
        
        # This is a complex problem, here's a more robust, simplified approach
        # We find round trips: BUY token with SOL/USDC, then SELL that token for SOL/USDC
        token_buys = defaultdict(list)
        for swap in sorted(all_swaps, key=lambda x: x['timestamp']):
            if swap['sold_symbol'] in ['SOL', 'USDC']: # This is a BUY of another token
                token_buys[swap['bought_symbol']].append(swap)
            elif swap['bought_symbol'] in ['SOL', 'USDC']: # This is a SELL of another token
                # Try to match with the earliest buy of this token
                if token_buys[swap['sold_symbol']]:
                    buy_trade = token_buys[swap['sold_symbol']].pop(0) # FIFO
                    
                    # Assume cost basis and proceeds are in the same currency (e.g., USDC or SOL)
                    pnl = 0
                    if buy_trade['sold_symbol'] == swap['bought_symbol']:
                        pnl = swap['bought_amount'] - buy_trade['sold_amount']

                    if pnl > 0:
                        pnl_by_token[swap['sold_symbol']]['wins'] += 1
                        pnl_by_token[swap['sold_symbol']]['profit'] += pnl
                    else:
                        pnl_by_token[swap['sold_symbol']]['losses'] += 1
                        pnl_by_token[swap['sold_symbol']]['loss'] += abs(pnl)

        total_wins = sum(d['wins'] for d in pnl_by_token.values())
        total_losses = sum(d['losses'] for d in pnl_by_token.values())
        total_trades = total_wins + total_losses
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(d['profit'] for d in pnl_by_token.values())
        total_loss = sum(d['loss'] for d in pnl_by_token.values())

        report = (
            f"--- Analysis Report for {self.wallet_address} ---\n"
            f"Total Txs Analyzed: {len(signatures_data)}\n"
            f"Identified Swaps: {len(all_swaps)}\n"
            f"Completed Round-Trip Trades: {total_trades}\n"
            f"--------------------------------------------\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total Profit from Wins: {total_profit:.4f} (SOL/USDC)\n"
            f"Total Loss from Losses: {total_loss:.4f} (SOL/USDC)\n"
            f"NET P/L: {total_profit - total_loss:+.4f} (SOL/USDC)"
        )
        self.analysis_complete.emit(report, all_swaps)

class AnalysisOptionsDialog(QDialog):
    """A dialog to get analysis options from the user."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Options")
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("How many of the latest transactions do you want to analyze?"))
        self.tx_limit_input = QSpinBox()
        self.tx_limit_input.setRange(100, 50000)
        self.tx_limit_input.setSingleStep(100)
        self.tx_limit_input.setValue(1000)
        layout.addWidget(self.tx_limit_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_limit(self):
        return self.tx_limit_input.value()

class TradesWindow(QDialog):
    """A new window to display all historical trades in a table."""
    def __init__(self, trades: List[Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Historical Swaps")
        self.setGeometry(150, 150, 1000, 600)
        
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Sold", "Bought", "Signature", "Link"])
        self.table.setRowCount(len(trades))
        
        for i, trade in enumerate(trades):
            self.table.setItem(i, 0, QTableWidgetItem(trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')))
            self.table.setItem(i, 1, QTableWidgetItem(f"{trade['sold_amount']:.4f} {trade['sold_symbol']}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{trade['bought_amount']:.4f} {trade['bought_symbol']}"))
            self.table.setItem(i, 3, QTableWidgetItem(trade['signature']))
            
            link_btn = QPushButton("View on Solscan")
            link_btn.clicked.connect(lambda _, s=trade['signature']: webbrowser.open(f"{SOLSCAN_BASE_URL}{s}"))
            self.table.setCellWidget(i, 4, link_btn)
            
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(3, 300)

        layout.addWidget(self.table)

# --- Main Application Window ---
class MainDashboard(QMainWindow):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.setWindowTitle("Solana Wallet Monitor & Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_stylesheet())

        self.paper_portfolio = PaperPortfolio()
        self.monitor_worker = None
        self.monitor_thread = None

        # Main layout
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.monitor_tab = QWidget()
        self.analyzer_tab = QWidget()

        self.tabs.addTab(self.monitor_tab, "Real-Time Monitor")
        self.tabs.addTab(self.analyzer_tab, "Historical Analyzer")

        self.init_monitor_ui()
        self.init_analyzer_ui()

        self.start_monitor()

    def get_stylesheet(self):
        return """
            QWidget { background-color: #111827; color: #e5e7eb; font-size: 14px; }
            QTabWidget::pane { border: 1px solid #374151; }
            QTabBar::tab { background: #1f2937; color: #9ca3af; padding: 10px; border-top-left-radius: 6px; border-top-right-radius: 6px;}
            QTabBar::tab:selected { background: #111827; color: white; }
            QLabel#Title { font-size: 18px; font-weight: bold; }
            QListWidget, QPlainTextEdit { background-color: #1f2937; border-radius: 6px; }
            QPushButton { background-color: #4f46e5; border-radius: 6px; padding: 10px; font-weight: bold; }
            QPushButton:hover { background-color: #4338ca; }
            QLineEdit { background-color: #374151; border: 1px solid #4b5563; border-radius: 6px; padding: 8px; }
            QProgressBar { border: 1px solid #4b5563; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background-color: #4f46e5; }
        """

    # --- Monitor Tab UI and Logic ---
    def init_monitor_ui(self):
        layout = QGridLayout(self.monitor_tab)
        
        # Portfolio Balances
        balance_title = QLabel("Paper Portfolio"); balance_title.setObjectName("Title")
        self.balance_table = QTableWidget()
        self.balance_table.setColumnCount(2)
        self.balance_table.setHorizontalHeaderLabels(["Token", "Amount"])
        self.balance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Live Trades Log
        trades_title = QLabel("Live Trade Feed"); trades_title.setObjectName("Title")
        self.trades_list = QListWidget()

        layout.addWidget(balance_title, 0, 0)
        layout.addWidget(self.balance_table, 1, 0)
        layout.addWidget(trades_title, 0, 1)
        layout.addWidget(self.trades_list, 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        self.update_portfolio_table()

    def start_monitor(self):
        wallets = load_wallets_from_json(WALLETS_FILE)
        if not wallets:
            self.trades_list.addItem("ERROR: Could not load wallets from wallets.json")
            return
        
        self.monitor_thread = QThread()
        self.monitor_worker = MonitorWorker(self.client, wallets)
        self.monitor_worker.moveToThread(self.monitor_thread)
        self.monitor_thread.started.connect(self.monitor_worker.run)
        self.monitor_worker.new_trade_signal.connect(self.on_new_trade)
        self.monitor_thread.start()
        
    def on_new_trade(self, trade_data: dict):
        self.paper_portfolio.update_from_trade(trade_data['changes'])
        self.update_portfolio_table()
        
        item_text = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {trade_data['wallet_name']}: {trade_data['details']}"
        self.trades_list.insertItem(0, item_text)
        if self.trades_list.count() > 100:
            self.trades_list.takeItem(100)

    def update_portfolio_table(self):
        self.balance_table.setRowCount(0)
        for symbol, amount in self.paper_portfolio.balances.items():
            row_pos = self.balance_table.rowCount()
            self.balance_table.insertRow(row_pos)
            self.balance_table.setItem(row_pos, 0, QTableWidgetItem(symbol))
            self.balance_table.setItem(row_pos, 1, QTableWidgetItem(f"{amount:,.8f}"))

    # --- Analyzer Tab UI and Logic ---
    def init_analyzer_ui(self):
        layout = QVBoxLayout(self.analyzer_tab)
        
        # Input section
        input_layout = QHBoxLayout()
        self.analyzer_input = QLineEdit()
        self.analyzer_input.setPlaceholderText("Enter Solana Wallet Address to Analyze...")
        self.analyze_btn = QPushButton("Analyze History")
        self.analyze_btn.clicked.connect(self.start_analysis)
        input_layout.addWidget(self.analyzer_input)
        input_layout.addWidget(self.analyze_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Output section
        self.analysis_report = QPlainTextEdit()
        self.analysis_report.setReadOnly(True)
        
        layout.addLayout(input_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.analysis_report)
    
    def show_analysis_options(self):
        """NEW: Opens the options dialog before starting analysis."""
        dialog = AnalysisOptionsDialog(self)
        if dialog.exec():
            limit = dialog.get_limit()
            wallet_address = self.analyzer_input.text().strip()
            if not wallet_address:
                self.analysis_report.setPlainText("Please enter a wallet address.")
                return
            self.start_analysis(wallet_address, limit)
    
    def start_analysis(self):
        wallet_address = self.analyzer_input.text().strip()
        if not wallet_address:
            self.analysis_report.setPlainText("Please enter a wallet address.")
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        self.progress_bar.setVisible(True)
        self.analysis_report.setPlainText("Starting analysis... This may take several minutes.")

        self.analyzer_thread = QThread()
        self.analyzer_worker = AnalyzerWorker(self.client, wallet_address)
        self.analyzer_worker.moveToThread(self.analyzer_thread)
        
        self.analyzer_thread.started.connect(self.analyzer_worker.run)
        self.analyzer_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analyzer_worker.progress_update.connect(self.on_progress_update)
        
        self.analyzer_thread.start()

    def on_progress_update(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.analysis_report.setPlainText(message) # Show live status

    def on_analysis_complete(self, report: str, trades: list):
        self.analysis_report.setPlainText(report)
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True); self.analyze_btn.setText("Analyze History")
        
        # NEW: Show the all trades window if trades were found
        if trades:
            self.trades_window = TradesWindow(trades, self)
            self.trades_window.show()

        self.analyzer_thread.quit()
        self.analyzer_thread.wait()

    def closeEvent(self, event):
        """Cleanly stop background threads on exit."""
        if self.monitor_worker:
            self.monitor_worker.stop()
            self.monitor_thread.quit()
            self.monitor_thread.wait()
        event.accept()

# --- Application Entry Point ---
async def async_main():
    # Setup command-line parsing inside async main
    parser = argparse.ArgumentParser(description="Solana Wallet Monitor and Analyzer.")
    parser.add_argument("--analyze", type=str, help="Run historical analysis on a specific wallet address instead of monitoring.")
    args = parser.parse_args()

    # Common initialization
    if not SOLANA_RPC_URL:
        logger.error("Missing SOLANA_RPC_URL environment variable. Exiting.")
        return
    
    solana_client = AsyncClient(SOLANA_RPC_URL, commitment=Confirmed)
    if not await solana_client.is_connected():
        logger.error(f"Failed to connect to Solana RPC: {SOLANA_RPC_URL}")
        return
    logger.info("✅ Connected to Solana RPC.")

    # --- Mode 1: Historical Analysis (Command-Line) ---
    if args.analyze:
        logger.info(f"--- Starting HISTORICAL ANALYSIS mode for wallet: {args.analyze} ---")
        analyzer = WalletAnalyzer(solana_client, args.analyze)
        await analyzer.run_analysis()
    
    # --- Mode 2: Launch PyQt6 Dashboard ---
    else:
        logger.info("--- Launching PyQt6 Dashboard for Real-Time Monitoring & Analysis ---")
        app = QApplication(sys.argv)
        # We need to pass the already connected client to the dashboard
        window = MainDashboard(solana_client)
        window.show()
        sys.exit(app.exec())
        
    # This part will be reached only after the GUI closes or analysis finishes
    await solana_client.close()

if __name__ == "__main__":
    try:
        # We need to run the async_main within an asyncio event loop
        # But PyQt also has its own event loop. The simplest way for this app
        # is to let PyQt run the main loop and handle async tasks in threads.
        # So we switch to a synchronous-style entry point for the GUI.
        app = QApplication(sys.argv)

        # We must create an event loop to run our initial async connection test
        loop = asyncio.get_event_loop()
        client = loop.run_until_complete(AsyncClient(SOLANA_RPC_URL).__aenter__())
        is_connected = loop.run_until_complete(client.is_connected())

        if not is_connected:
            logger.error("Failed to connect to Solana RPC.")
            sys.exit(1)
            
        logger.info("✅ Connected to Solana RPC (sync check).")

        window = MainDashboard(client)
        window.show()
        
        app_exit_code = app.exec()

        # Cleanly close the client session
        loop.run_until_complete(client.__aexit__(None, None, None))
        sys.exit(app_exit_code)

    except KeyboardInterrupt:
        logger.info("Script stopped by user.")
