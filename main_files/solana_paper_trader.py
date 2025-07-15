# -*- coding: utf-8 -*-
import asyncio
import logging
import os
import time
import json
from collections import defaultdict
from typing import Dict, Set, Optional, List, Any
import datetime

# Telegram libraries
import telegram
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

# Solana libraries
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.exceptions import SolanaRpcException
from solders.pubkey import Pubkey
from solders.signature import Signature

# Helper libraries
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WALLETS_FILE = "wallets.json"

# Polling interval in seconds
POLLING_INTERVAL = 15
SIGNATURE_FETCH_LIMIT = 15

# Filtering Settings
MAX_TRANSACTION_AGE = 300  # Skip transactions older than 5 minutes
PROCESSED_SIGNATURES_MAX_SIZE = 2000

# Explorer URL
SOLSCAN_BASE_URL = "https://solscan.io/tx/"

# Known Program IDs for filtering
KNOWN_DEX_PROGRAM_IDS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM/Liquidity Pool V4
    "routeXXXXXXX11111111111111111111111111111111",  # Raydium Router
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3ekubTQ",   # Orca Whirlpool
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",    # Jupiter v6 Aggregator
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",    # Pump.fun
}

# Common Token Info
KNOWN_TOKENS = {
    "So11111111111111111111111111111111111111112": {"symbol": "SOL", "decimals": 9},
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {"symbol": "USDC", "decimals": 6},
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": {"symbol": "USDT", "decimals": 6},
}

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
processed_signatures: Dict[str, Set[Signature]] = defaultdict(set)

# --- Paper Trading Portfolio ---
class PaperPortfolio:
    def __init__(self, initial_usd_value=10000.0):
        # We track balances by token symbol for simplicity
        self.balances: Dict[str, float] = {"USDC": initial_usd_value}
        self.initial_value = initial_usd_value
        logger.info(f"Paper portfolio initialized with ${initial_usd_value:,.2f} USDC.")

    def update_from_trade(self, balance_changes: Dict[str, Dict[str, Any]]):
        """Mirrors the token changes from a target wallet's trade."""
        logger.info("Updating paper portfolio based on detected swap...")
        for change_info in balance_changes.values():
            symbol = change_info["symbol"]
            change_amount = change_info["change"]
            
            # Add or subtract the token amount from our balance
            self.balances[symbol] = self.balances.get(symbol, 0.0) + change_amount
            
            # Clean up if balance is near zero
            if abs(self.balances[symbol]) < 1e-9:
                del self.balances[symbol]
        logger.info("Paper portfolio update complete.")

    def get_display_string(self) -> str:
        """Formats the portfolio for display in a Telegram message."""
        if not self.balances:
            return "Paper portfolio is empty."
            
        lines = ["\n*Paper Portfolio Balances:*"]
        # Sort by symbol, with USDC and SOL first if they exist
        sorted_symbols = sorted(
            self.balances.keys(),
            key=lambda s: (s not in ["USDC", "SOL"], s)
        )

        for symbol in sorted_symbols:
            amount = self.balances[symbol]
            # Use a reasonable number of decimal places for display
            amount_str = f"{amount:,.8f}".rstrip('0').rstrip('.')
            lines.append(f"  `{amount_str}` {escape_markdown_v2(symbol)}")
            
        return "\n".join(lines)

# --- Utility & Helper Functions ---
def load_wallets_from_json(file_path: str) -> Dict[str, str]:
    """Loads a list of wallets from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            wallets_list = json.load(f)
        # Convert list of dicts to a single dict
        return {item['name']: item['address'] for item in wallets_list}
    except FileNotFoundError:
        logger.error(f"Error: The wallet file '{file_path}' was not found.")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing {file_path}. Ensure it's a valid JSON list of objects with 'name' and 'address' keys. Error: {e}")
        return {}

def escape_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}!.'
    return "".join(f'\\{char}' if char in escape_chars else char for char in str(text))

async def send_telegram_message(bot: Bot, message: str):
    """Sends a message to the configured Telegram chat."""
    if not bot or not TELEGRAM_CHAT_ID:
        return
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True
        )
    except TelegramError as e:
        logger.error(f"Telegram API Error: {e}")
        logger.debug(f"Failed message content:\n---\n{message}\n---")

# --- Solana Interaction & Parsing Functions ---
# (These functions are based on the robust logic from your blueprint)
async def get_latest_signatures(client: AsyncClient, wallet_address: Pubkey) -> Optional[List[Any]]:
    try:
        return (await client.get_signatures_for_address(wallet_address, limit=SIGNATURE_FETCH_LIMIT, commitment=Confirmed)).value
    except Exception as e:
        logger.error(f"Error fetching signatures for {wallet_address}: {e}")
        return None

async def get_transaction_details(client: AsyncClient, signature: Signature) -> Optional[Any]:
    for attempt in range(3):
        try:
            resp = await client.get_transaction(signature, encoding="jsonParsed", max_supported_transaction_version=0, commitment=Confirmed)
            if resp.value: return resp.value
            await asyncio.sleep(2 * (attempt + 1))
        except Exception as e:
            logger.error(f"Error fetching tx {signature} on attempt {attempt+1}: {e}")
    return None

def get_token_info(mint_address: Pubkey, accounts: List[Any]) -> Dict[str, Any]:
    mint_str = str(mint_address)
    if mint_str in KNOWN_TOKENS: return KNOWN_TOKENS[mint_str]
    
    # Fallback to finding decimals from token accounts in the transaction
    for acc in accounts:
        if isinstance(acc, dict) and acc.get('mint') == mint_str and 'tokenAmount' in acc:
            return {"symbol": f"{mint_str[:4]}...{mint_str[-4:]}", "decimals": acc['tokenAmount']['decimals']}
            
    return {"symbol": f"{mint_str[:4]}...{mint_str[-4:]}", "decimals": 9} # Default to 9

def parse_balance_changes(tx_result: Any, wallet_pubkey: Pubkey) -> Dict[str, Dict[str, Any]]:
    changes: Dict[str, Dict[str, Any]] = {}
    meta = getattr(tx_result.transaction, 'meta', None)
    if not meta: return changes

    all_accounts = tx_result.transaction.transaction.message.account_keys
    
    # Token Balances
    pre_token_balances = meta.pre_token_balances or []
    post_token_balances = meta.post_token_balances or []
    wallet_pre_map = {str(b.mint): b for b in pre_token_balances if b.owner == wallet_pubkey}
    wallet_post_map = {str(b.mint): b for b in post_token_balances if b.owner == wallet_pubkey}
    all_mints = set(wallet_pre_map.keys()) | set(wallet_post_map.keys())

    for mint_str in all_mints:
        pre_amount = int(getattr(wallet_pre_map.get(mint_str, {}), 'ui_token_amount', {}).get('amount', '0'))
        post_amount = int(getattr(wallet_post_map.get(mint_str, {}), 'ui_token_amount', {}).get('amount', '0'))
        raw_change = post_amount - pre_amount

        if raw_change != 0:
            token_info = get_token_info(Pubkey.from_string(mint_str), all_accounts)
            changes[mint_str] = {"change": raw_change / (10 ** token_info["decimals"]), "symbol": token_info["symbol"]}
    
    # SOL Balance
    try:
        wallet_idx = all_accounts.index(wallet_pubkey)
        sol_change = meta.post_balances[wallet_idx] - meta.pre_balances[wallet_idx]
        if sol_change != 0:
            changes["SOL"] = {"change": sol_change / 1e9, "symbol": "SOL"}
    except (ValueError, IndexError) as e:
        logger.debug(f"Could not determine SOL change for {wallet_pubkey}: {e}")

    return changes

def detect_dex_interaction(tx_result: Any) -> bool:
    try:
        instructions = tx_result.transaction.transaction.message.instructions
        for inst in instructions:
            if str(inst.program_id) in KNOWN_DEX_PROGRAM_IDS:
                return True
    except AttributeError:
        return False
    return False

def format_swap_details(balance_changes: Dict[str, Dict[str, Any]]) -> Optional[str]:
    sold = next((item for item in balance_changes.values() if item['change'] < 0), None)
    bought = next((item for item in balance_changes.values() if item['change'] > 0), None)

    if sold and bought:
        return (f"🔄 Swapped `{abs(sold['change']):.4f}` {escape_markdown_v2(sold['symbol'])} "
                f"for `{bought['change']:.4f}` {escape_markdown_v2(bought['symbol'])}")
    return None

# --- Main Monitoring Loop ---
async def monitor_wallet(client: AsyncClient, bot: Bot, paper_portfolio: PaperPortfolio, wallet_name: str, wallet_address: str):
    logger.info(f"Starting monitoring for {wallet_name} ({wallet_address})")
    wallet_pubkey = Pubkey.from_string(wallet_address)
    
    while True:
        try:
            current_time = time.time()
            latest_signatures_data = await get_latest_signatures(client, wallet_pubkey)

            if latest_signatures_data:
                for sig_data in reversed(latest_signatures_data):
                    signature = sig_data.signature
                    if signature in processed_signatures[wallet_address]:
                        continue
                    
                    processed_signatures[wallet_address].add(signature)
                    if len(processed_signatures[wallet_address]) > PROCESSED_SIGNATURES_MAX_SIZE:
                        processed_signatures[wallet_address].pop()

                    if sig_data.block_time and (current_time - sig_data.block_time > MAX_TRANSACTION_AGE):
                        continue

                    logger.info(f"[{wallet_name}] Processing new signature: {signature}")
                    tx_result = await get_transaction_details(client, signature)

                    if not tx_result or getattr(getattr(tx_result.transaction, 'meta', None), 'err', None):
                        logger.info(f"[{wallet_name}] Tx failed or details unavailable. Skipping.")
                        continue

                    is_swap = detect_dex_interaction(tx_result)
                    if not is_swap:
                        logger.info(f"[{wallet_name}] Tx is not a DEX interaction. Skipping.")
                        continue
                        
                    balance_changes = parse_balance_changes(tx_result, wallet_pubkey)
                    swap_details = format_swap_details(balance_changes)

                    if swap_details:
                        # --- COPY TRADE LOGIC ---
                        paper_portfolio.update_from_trade(balance_changes)
                        
                        # --- NOTIFICATION LOGIC ---
                        message = (
                            f"🎯 *Copy Trade Alert for {escape_markdown_v2(wallet_name)}*\n\n"
                            f"{swap_details}\n\n"
                            f"[View on Solscan]({SOLSCAN_BASE_URL}{signature})"
                            f"{paper_portfolio.get_display_string()}" # Add portfolio status to message
                        )
                        await send_telegram_message(bot, message)

        except Exception as e:
            logger.error(f"Error in monitoring loop for {wallet_name}: {e}", exc_info=True)
            await asyncio.sleep(POLLING_INTERVAL * 2) # Wait longer on error

        await asyncio.sleep(POLLING_INTERVAL)

async def main():
    """Main function to initialize clients and start monitoring tasks."""
    if not all([SOLANA_RPC_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("Missing required environment variables. Exiting.")
        return

    wallets_to_monitor = load_wallets_from_json(WALLETS_FILE)
    if not wallets_to_monitor:
        logger.error(f"No valid wallets found in {WALLETS_FILE}. Exiting.")
        return

    solana_client = AsyncClient(SOLANA_RPC_URL, commitment=Confirmed)
    telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
    paper_portfolio = PaperPortfolio()

    try:
        if not await solana_client.is_connected():
            logger.error(f"Failed to connect to Solana RPC: {SOLANA_RPC_URL}")
            return
        logger.info("✅ Connected to Solana RPC.")
        
        bot_info = await telegram_bot.get_me()
        logger.info(f"✅ Connected to Telegram Bot: {bot_info.username}")

        tasks = [monitor_wallet(solana_client, telegram_bot, paper_portfolio, name, address) for name, address in wallets_to_monitor.items()]
        logger.info(f"🚀 Starting monitoring for {len(tasks)} wallet(s)...")
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.critical(f"A critical error occurred in main: {e}", exc_info=True)
    finally:
        await solana_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user.")
