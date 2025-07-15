import asyncio
import json
import time
import datetime
import requests
from typing import Dict, List, Set, Optional, Any
import logging
import os

# For telegram bot v20+
from telegram import Bot
from telegram.constants import ParseMode

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
RPC_ENDPOINT = "https://white-light-sky.solana-mainnet.quiknode.pro/fd5499d7bcba97dd228efa87a0610324f5ae01bd"
TELEGRAM_BOT_TOKEN = "8045923523:AAE2xf4Z2BvAtu701vA8gHLoknGt8rnWuzA"
TELEGRAM_CHAT_ID = "417977974"
POLLING_INTERVAL = 10  # seconds

# Filter settings
MAX_TRANSACTION_AGE = 300  # Skip transactions older than 5 minutes (300 seconds)
SKIP_FAILED_TRANSACTIONS = True  # Skip transactions that failed to execute
SKIP_EXISTING_TRANSACTIONS = True  # Skip transactions that existed before script started

# Known DEX program IDs
RAYDIUM_SWAP_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
ORCA_SWAP_PROGRAM_ID = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"
JUPITER_PROGRAM_ID = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"

# Wallets to monitor
WALLETS = {
    #"God's Country": "J3JeorQjFNLpP1Te1shKDs2CGnqVma1iaxJxPKUaCxKh",
    "IFR": "B9znJmM44AnG5hnXdbuZvrpWFxFe73Uwapyr3msWujVj",
    "God's Country2": "GPavhMZpz1TEd6Zdyt3mv1JFMmupF12pCLmEVzQA8vzd",
}

# Store processed transaction signatures
processed_signatures: Set[str] = set()

token_symbol_cache = {}


def get_token_symbol(mint_address: str) -> str:
    """Get token symbol from mint address using external API if needed."""
    # Check cache first
    if mint_address in token_symbol_cache:
        return token_symbol_cache[mint_address]

    # Common tokens we already know
    known_tokens = {
        "So11111111111111111111111111111111111111112": "SOL",
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
    }

    # Return known token if available
    if mint_address in known_tokens:
        token_symbol_cache[mint_address] = known_tokens[mint_address]
        return known_tokens[mint_address]

    # Otherwise try to fetch from an API
    try:
        # Example of fetching from Jupiter API
        response = requests.get(f"https://token.jup.ag/all")
        if response.status_code == 200:
            tokens = response.json()
            for token in tokens:
                if token.get("address") == mint_address:
                    symbol = token.get("symbol", f"{mint_address[:4]}...{mint_address[-4:]}")
                    token_symbol_cache[mint_address] = symbol
                    return symbol
    except Exception as e:
        logger.debug(f"Error fetching token data: {e}")

    # Fallback to shortened address
    shortened = f"{mint_address[:4]}...{mint_address[-4:]}"
    token_symbol_cache[mint_address] = shortened
    return shortened


# Initialize Telegram bot
async def setup_telegram():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not provided. Notifications will be disabled.")
        return None

    return Bot(token=TELEGRAM_BOT_TOKEN)


# Solana RPC helper functions
def solana_rpc_call(method: str, params: List[Any]) -> Dict:
    """Make a call to the Solana JSON RPC API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }

    try:
        response = requests.post(RPC_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"RPC call failed: {e}")
        return {"error": str(e)}


def get_signatures_for_address(wallet_address: str, limit: int = 10) -> List[Dict]:
    """Get recent transaction signatures for a wallet address."""
    result = solana_rpc_call(
        "getSignaturesForAddress",
        [wallet_address, {"limit": limit}]
    )

    if "result" in result:
        return result["result"]

    logger.error(f"Failed to get signatures: {result.get('error')}")
    return []


def get_transaction(signature: str) -> Optional[Dict]:
    """Get detailed information about a transaction."""
    result = solana_rpc_call(
        "getTransaction",
        [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
    )

    if "result" in result and result["result"]:
        return result["result"]

    logger.error(f"Failed to get transaction {signature}: {result.get('error')}")
    return None


def is_transaction_successful(tx_data: Dict) -> bool:
    """Check if a transaction executed successfully."""
    # Check for the "err" field in meta
    if "meta" in tx_data and tx_data["meta"] is not None:
        return tx_data["meta"].get("err") is None

    # If we can't determine status, assume successful
    return True


def is_transaction_recent(tx_data: Dict) -> bool:
    """Check if a transaction is recent (within MAX_TRANSACTION_AGE seconds)."""
    if "blockTime" not in tx_data:
        return False

    block_time = tx_data.get("blockTime", 0)
    current_time = int(time.time())

    # Check if transaction is within the allowed age
    return (current_time - block_time) <= MAX_TRANSACTION_AGE


# Token balance and DEX detection functions
def extract_token_balances(transaction: Dict) -> List[Dict]:
    """Extract token balance changes from a transaction."""
    balance_changes = []

    if not transaction or "meta" not in transaction:
        return balance_changes

    pre_balances = transaction["meta"].get("preTokenBalances", [])
    post_balances = transaction["meta"].get("postTokenBalances", [])

    # Create maps for easier lookup
    pre_map = {f"{b.get('accountIndex')}_{b.get('mint', '')}": b for b in pre_balances}
    post_map = {f"{b.get('accountIndex')}_{b.get('mint', '')}": b for b in post_balances}

    # Find all unique account+mint combinations
    all_keys = set(list(pre_map.keys()) + list(post_map.keys()))

    for key in all_keys:
        pre = pre_map.get(key)
        post = post_map.get(key)

        if not pre and not post:
            continue

        # Get token information
        mint = pre.get("mint") if pre else post.get("mint")
        if not mint:
            continue

        account_index = int(pre.get("accountIndex", 0)) if pre else int(post.get("accountIndex", 0))

        # Calculate balance changes
        pre_amount = float(pre.get("uiTokenAmount", {}).get("uiAmount") or 0) if pre else 0
        post_amount = float(post.get("uiTokenAmount", {}).get("uiAmount") or 0) if post else 0
        change = post_amount - pre_amount

        # Skip if no change
        if change == 0:
            continue

        # Get token info
        token_info = None
        if pre and "uiTokenAmount" in pre:
            token_info = pre["uiTokenAmount"]
        elif post and "uiTokenAmount" in post:
            token_info = post["uiTokenAmount"]

        # Get the token symbol or use the mint address as a fallback
        symbol = "Unknown"
        if token_info and token_info.get("symbol"):
            symbol = token_info.get("symbol")
        else:
            # Try to find token info from API
            symbol = get_token_symbol(mint)

        if token_info:
            balance_changes.append({
                "mint": mint,
                "symbol": symbol,  # Using our looked-up symbol instead of the original line
                "account_index": account_index,
                "pre_amount": pre_amount,
                "post_amount": post_amount,
                "change": change,
                "decimals": token_info.get("decimals", 0)
            })

    return balance_changes


def is_dex_swap(transaction: Dict) -> bool:
    """Check if a transaction is a DEX swap."""
    if not transaction or "transaction" not in transaction:
        return False

    # Try to get program IDs from different possible formats
    program_ids = []

    # Check in account keys
    try:
        account_keys = transaction.get("transaction", {}).get("message", {}).get("accountKeys", [])

        # Handle different formats
        if isinstance(account_keys, list):
            if isinstance(account_keys[0], str):
                # Simple string list format
                program_ids.extend(account_keys)
            elif isinstance(account_keys[0], dict):
                # Object format with pubkey field
                program_ids.extend([key.get("pubkey") for key in account_keys])
    except (IndexError, KeyError, TypeError):
        pass

    # Also check in instructions
    try:
        instructions = transaction.get("transaction", {}).get("message", {}).get("instructions", [])
        for instruction in instructions:
            if "programId" in instruction:
                program_ids.append(instruction["programId"])
    except (KeyError, TypeError):
        pass

    # Check if any known DEX program IDs are in the transaction
    return any(
        program_id in [RAYDIUM_SWAP_PROGRAM_ID, ORCA_SWAP_PROGRAM_ID, JUPITER_PROGRAM_ID] for program_id in program_ids)


def get_tx_link(signature: str) -> str:
    """Generate a Solscan link for a transaction."""
    return f"https://solscan.io/tx/{signature}"


async def send_telegram_notification(bot, wallet_name: str, wallet_address: str, tx_data: Dict,
                                     token_changes: List[Dict]):
    """Send a notification to Telegram with transaction details."""
    if not bot or not TELEGRAM_CHAT_ID:
        logger.info("Telegram notifications disabled")
        return

    # Get transaction signature
    signature = None
    if "transaction" in tx_data and "signatures" in tx_data["transaction"]:
        signature = tx_data["transaction"]["signatures"][0]
    else:
        signature = tx_data.get("transaction", {}).get("signatures", ["Unknown"])[0]

    tx_link = get_tx_link(signature)

    # Determine if this is a DEX swap
    is_swap = is_dex_swap(tx_data)
    tx_type = "DEX Swap" if is_swap else "Transaction"

    # Get timestamp
    timestamp = tx_data.get("blockTime", None)
    time_str = ""
    if timestamp:
        dt = datetime.datetime.fromtimestamp(timestamp)
        time_str = f" at {dt.strftime('%Y-%m-%d %H:%M:%S')}"

    # Get transaction status
    is_successful = is_transaction_successful(tx_data)
    status_str = "✅ SUCCESS" if is_successful else "❌ FAILED"

    # Create message
    message = (
        f"🔔 New {tx_type} detected for {wallet_name}{time_str}\n"
        f"Status: {status_str}\n\n"
        f"👛 Wallet: {wallet_address[:6]}...{wallet_address[-4:]}\n"
        f"🔗 Transaction: <a href='{tx_link}'>{signature[:6]}...{signature[-4:]}</a>\n\n"
    )

    # Add token changes
    if token_changes:
        message += "💰 Token Changes:\n"
        for change in token_changes:
            direction = "+" if change["change"] > 0 else "-"
            amount = abs(change["change"])
            symbol = change["symbol"] or "Unknown"
            message += f"{direction} {amount:.6f} {symbol}\n"
    else:
        message += "No token balance changes detected.\n"

    # Send message
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        logger.info(f"Notification sent for transaction {signature}")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")


async def initialize_wallet_monitoring(wallet_address: str) -> Set[str]:
    """Get initial transactions to ignore."""
    if not SKIP_EXISTING_TRANSACTIONS:
        return set()

    signatures_to_ignore = set()
    try:
        transactions = get_signatures_for_address(wallet_address, limit=50)  # Increased from 20 to 50
        signatures_to_ignore = {tx["signature"] for tx in transactions}
        logger.info(f"Initialized with {len(signatures_to_ignore)} existing transactions to skip")
    except Exception as e:
        logger.error(f"Error initializing wallet monitoring: {e}")

    return signatures_to_ignore


async def monitor_wallet(bot, wallet_name: str, wallet_address: str):
    """Monitor a single wallet for new transactions."""
    global processed_signatures

    logger.info(f"Starting to monitor wallet: {wallet_name} ({wallet_address})")

    # Initialize by getting existing transactions to ignore
    signatures_to_ignore = await initialize_wallet_monitoring(wallet_address)
    processed_signatures.update(signatures_to_ignore)

    while True:
        try:
            # Get recent transactions
            transactions = get_signatures_for_address(wallet_address)

            # Process new transactions
            for tx in transactions:
                signature = tx["signature"]

                # Skip if already processed
                if signature in processed_signatures:
                    continue

                # Mark as processed immediately to prevent duplicates
                processed_signatures.add(signature)

                # Get transaction details
                tx_data = get_transaction(signature)
                if not tx_data:
                    continue

                # Skip old transactions
                if not is_transaction_recent(tx_data):
                    logger.info(f"Skipping old transaction: {signature}")
                    continue

                # Skip failed transactions if configured
                if SKIP_FAILED_TRANSACTIONS and not is_transaction_successful(tx_data):
                    logger.info(f"Skipping failed transaction: {signature}")
                    continue

                # Extract token balance changes
                token_changes = extract_token_balances(tx_data)

                # Skip transactions with no detectable token changes
                if not token_changes:
                    logger.info(f"Skipping transaction with no token changes: {signature}")
                    continue

                logger.info(f"New valid transaction found: {signature}")

                # Send notification
                await send_telegram_notification(bot, wallet_name, wallet_address, tx_data, token_changes)

            # Prevent the set from growing too large
            if len(processed_signatures) > 1000:
                processed_signatures = set(list(processed_signatures)[-500:])

        except Exception as e:
            logger.error(f"Error monitoring wallet {wallet_name}: {e}")

        # Wait before next poll
        await asyncio.sleep(POLLING_INTERVAL)


async def main():
    """Main function to run the wallet monitor."""
    logger.info("Starting Solana wallet monitor with improved filtering...")
    logger.info(
        f"Filtering settings: Skip failed: {SKIP_FAILED_TRANSACTIONS}, Skip older than {MAX_TRANSACTION_AGE} seconds: True")

    # Setup Telegram bot
    bot = await setup_telegram()

    # Start monitoring tasks for each wallet
    tasks = []
    for wallet_name, wallet_address in WALLETS.items():
        if wallet_address:
            task = asyncio.create_task(monitor_wallet(bot, wallet_name, wallet_address))
            tasks.append(task)

    if not tasks:
        logger.error("No valid wallet addresses configured. Exiting.")
        return

    logger.info(f"Monitoring {len(tasks)} wallets with polling interval of {POLLING_INTERVAL} seconds")

    # Run forever
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Wallet monitor stopped by user")
    except Exception as e:
        logger.error(f"Wallet monitor crashed: {e}")