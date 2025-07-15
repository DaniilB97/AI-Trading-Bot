# -*- coding: utf-8 -*-
# test_analyzer.py v2
import asyncio
import logging
import os
import json
from collections import defaultdict
from typing import Dict, Optional, List, Any
import datetime
import argparse
import requests # NEW: Added for making HTTP requests to price APIs

# Solana & Helper Libraries
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
from solders.signature import Signature
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")
SOLSCAN_BASE_URL = "https://solscan.io/tx/"

JUPITER_V6_PROGRAM_ID = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
KNOWN_DEX_PROGRAM_IDS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3ekubTQ",
    JUPITER_V6_PROGRAM_ID, "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
}

KNOWN_TOKENS = {
    "So11111111111111111111111111111111111111112": {"symbol": "SOL", "decimals": 9},
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {"symbol": "USDC", "decimals": 6},
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": {"symbol": "USDT", "decimals": 6},
}
BASE_CURRENCIES = {"SOL", "USDC", "USDT"}
token_cache = {} # NEW: Cache for token symbols

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- NEW: Token Information and Pricing Functions ---
def get_token_symbol(mint_address: str) -> str:
    """Get token symbol from mint address, using cache and APIs."""
    if mint_address in token_cache: return token_cache[mint_address]
    if mint_address in KNOWN_TOKENS:
        symbol = KNOWN_TOKENS[mint_address]["symbol"]
        token_cache[mint_address] = symbol
        return symbol
    try:
        response = requests.get("https://token.jup.ag/all")
        if response.status_code == 200:
            for token in response.json():
                if token.get("address") == mint_address:
                    symbol = token.get("symbol", f"{mint_address[:4]}...{mint_address[-4:]}")
                    token_cache[mint_address] = symbol
                    return symbol
    except Exception as e: logger.error(f"Error fetching token list from Jupiter: {e}")
    fallback_symbol = f"{mint_address[:4]}...{mint_address[-4:]}"
    token_cache[mint_address] = fallback_symbol
    return fallback_symbol

def get_token_price(mint_address: str) -> float:
    """Get current token price in USD from Jupiter Price API."""
    if mint_address == "So11111111111111111111111111111111111111112":
        # Using a reliable endpoint for SOL price
        try:
            response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd")
            return response.json().get("solana", {}).get("usd", 0)
        except Exception as e: logger.error(f"Error fetching SOL price from CoinGecko: {e}"); return 0
    try:
        response = requests.get(f"https://price.jup.ag/v4/price?ids={mint_address}")
        if response.status_code == 200:
            data = response.json()
            return data.get("data", {}).get(mint_address, {}).get("price", 0)
    except Exception as e: logger.error(f"Error fetching token price from Jupiter: {e}")
    return 0


# --- Solana Parsing & Helper Functions (Updated) ---
async def get_transaction_details(client: AsyncClient, signature: Signature) -> Optional[Any]:
    for _ in range(3):
        try:
            resp = await client.get_transaction(signature, encoding="jsonParsed", max_supported_transaction_version=0)
            if resp.value: return resp.value
            await asyncio.sleep(0.5)
        except Exception as e: logger.debug(f"Attempt failed for {signature}: {e}")
    logger.error(f"Failed to fetch details for {signature} after retries.")
    return None

def parse_balance_changes(tx_data: Any, wallet_pubkey: Pubkey) -> Dict[str, Dict[str, Any]]:
    """Parses pre/post token balances for the wallet, including SOL."""
    changes = {}
    meta = getattr(tx_data.transaction, 'meta', None)
    if not meta: return changes
    
    pre_token_balances = meta.pre_token_balances or []
    post_token_balances = meta.post_token_balances or []
    pre_map = {str(b.mint): b.ui_token_amount for b in pre_token_balances if hasattr(b, 'owner') and b.owner == wallet_pubkey}
    post_map = {str(b.mint): b.ui_token_amount for b in post_token_balances if hasattr(b, 'owner') and b.owner == wallet_pubkey}
    all_mints = set(pre_map.keys()) | set(post_map.keys())

    for mint_str in all_mints:
        pre_info, post_info = pre_map.get(mint_str), post_map.get(mint_str)
        decimals = getattr(pre_info, 'decimals', getattr(post_info, 'decimals', 9))
        pre_amt, post_amt = int(getattr(pre_info, 'amount', '0')), int(getattr(post_info, 'amount', '0'))
        
        if (raw_change := post_amt - pre_amt) != 0:
            symbol = get_token_symbol(mint_str)
            changes[mint_str] = {"change": raw_change / (10 ** decimals), "symbol": symbol}
            
    try:
        acc_keys = tx_data.transaction.transaction.message.account_keys
        wallet_idx = acc_keys.index(wallet_pubkey)
        sol_change = meta.post_balances[wallet_idx] - meta.pre_balances[wallet_idx]
        if sol_change != 0:
            sol_mint = "So11111111111111111111111111111111111111112"
            changes[sol_mint] = {"change": sol_change / 1e9, "symbol": "SOL"}
    except (ValueError, IndexError): pass
    return changes

def is_dex_swap(tx_data: Any) -> bool:
    """Checks if the transaction is a DEX swap, with a focus on Jupiter."""
    try:
        log_messages = getattr(getattr(tx_data.transaction, 'meta', None), 'log_messages', [])
        if any("Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 invoke" in log for log in log_messages):
            return True
        for inner_inst in getattr(getattr(tx_data.transaction, 'meta', None), 'inner_instructions', []) or []:
            for inst in inner_inst.instructions:
                if str(inst.program_id) in KNOWN_DEX_PROGRAM_IDS: return True
    except AttributeError: return False
    return False

# --- Main Wallet History Analyzer ---
class WalletAnalyzer:
    def __init__(self, client: AsyncClient, wallet_address: str):
        self.client = client
        self.wallet_address = wallet_address
        self.wallet_pubkey = Pubkey.from_string(wallet_address)
        self.swaps = []
        logger.info(f"Initialized analyzer for wallet: {wallet_address}")

    async def fetch_transactions(self, limit: int) -> List[Any]:
        logger.info(f"Fetching last {limit} transaction signatures...")
        signatures_data = (await self.client.get_signatures_for_address(self.wallet_pubkey, limit=limit)).value
        if not signatures_data:
            logger.warning("No signatures found."); return []
        
        logger.info(f"Found {len(signatures_data)} signatures. Fetching details sequentially...")
        transactions = []
        for i, sig_data in enumerate(signatures_data):
            print(f"\rFetching transaction details {i+1}/{len(signatures_data)}...", end="")
            tx = await get_transaction_details(self.client, sig_data.signature)
            if tx: transactions.append(tx)
            await asyncio.sleep(0.1)
        print("\nDone fetching details.")
        return transactions

    def analyze_transactions(self, transactions: List[Any]):
        """Processes transactions to find swaps and enrich them with price data."""
        logger.info(f"Analyzing {len(transactions)} transactions...")
        for tx in transactions:
            if tx and is_dex_swap(tx) and not getattr(getattr(tx.transaction, 'meta', None), 'err', None):
                changes = parse_balance_changes(tx, self.wallet_pubkey)
                if len(changes) >= 2:
                    sold = next((v for v in changes.values() if v['change'] < 0), None)
                    bought = next((v for v in changes.values() if v['change'] > 0), None)
                    if sold and bought:
                        # --- NEW: Fetch price for both tokens ---
                        sold_price_usd = get_token_price(list(changes.keys())[list(changes.values()).index(sold)])
                        bought_price_usd = get_token_price(list(changes.keys())[list(changes.values()).index(bought)])
                        
                        self.swaps.append({
                            "timestamp": datetime.datetime.fromtimestamp(tx.block_time, tz=datetime.timezone.utc),
                            "signature": str(tx.transaction.signatures[0]),
                            "sold_symbol": sold["symbol"], "sold_amount": abs(sold["change"]),
                            "bought_symbol": bought["symbol"], "bought_amount": bought["change"],
                            "sold_usd_value": abs(sold["change"]) * sold_price_usd,
                            "bought_usd_value": bought["change"] * bought_price_usd,
                        })
        self.swaps.sort(key=lambda x: x['timestamp'])
        logger.info(f"Found {len(self.swaps)} valid DEX swaps in the analyzed transactions.")

    def calculate_statistics(self):
        """Calculates and prints statistics based on the found swaps."""
        if not self.swaps:
            print("\nNo DEX swaps found to calculate statistics.")
            return

        # P&L logic remains simplified for now, as historical prices are needed for true P&L.
        # But we can display the USD value of each swap.

        print("\n" + "="*50)
        print(f"📊 Wallet Analysis Report for: {self.wallet_address}")
        print("="*50)
        print(f"Total DEX Swaps Identified: {len(self.swaps)}")
        print("="*50)
        
        print("\n--- DETAILED SWAPS (with approximate USD values) ---\n")
        for swap in self.swaps:
            print(
                f"[{swap['timestamp'].strftime('%Y-%m-%d %H:%M')}] "
                f"Swapped {swap['sold_amount']:.4f} {swap['sold_symbol']} (~${swap['sold_usd_value']:.2f}) "
                f"for {swap['bought_amount']:.4f} {swap['bought_symbol']} (~${swap['bought_usd_value']:.2f})"
            )


# --- Main Execution Block ---
async def main():
    parser = argparse.ArgumentParser(description="Solana Wallet Swap Analyzer.")
    parser.add_argument("wallet", type=str, help="The Solana wallet address to analyze.")
    parser.add_argument("--limit", type=int, default=100, help="Number of recent transactions to analyze.")
    args = parser.parse_args()

    if not SOLANA_RPC_URL:
        logger.error("Missing SOLANA_RPC_URL environment variable. Exiting.")
        return
    
    async with AsyncClient(SOLANA_RPC_URL) as client:
        if not await client.is_connected():
            logger.error("Failed to connect to Solana RPC.")
            return
        
        analyzer = WalletAnalyzer(client, args.wallet)
        transactions = await analyzer.fetch_transactions(args.limit)
        analyzer.analyze_transactions(transactions)
        analyzer.calculate_statistics()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Analysis stopped by user.")
