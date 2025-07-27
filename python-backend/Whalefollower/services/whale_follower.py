#!/usr/bin/env python3
"""
Whale Trade Follower - Real-time Transaction Monitoring & Copying
Monitors whale wallets and executes similar trades automatically
"""

import os
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
from eth_account import Account
from dotenv import load_dotenv
import logging
from decimal import Decimal

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whale_follower.log'),
        logging.StreamHandler()
    ]
)

class WhaleFollower:
    def __init__(self):
        # API Keys and connections
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.infura_api_key = os.getenv('INFURA_API_KEY')
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        
        # Private key for executing trades
        self.private_key = os.getenv('PRIVATE_KEY')
        self.account = Account.from_key(self.private_key) if self.private_key else None
        
        # Web3 connection
        self.w3 = self._setup_web3()
        
        # Configuration
        self.config = self._load_config()
        self.wallets_file = "wallets.json"
        self.processed_txs_file = "processed_transactions.json"
        self.trade_log_file = "trade_log.json"
        
        # State
        self.monitored_wallets = self._load_wallets()
        self.processed_txs = self._load_processed_txs()
        self.known_tokens = {}
        
        # Rate limiting
        self.etherscan_delay = 0.25
        self.last_etherscan_call = 0
        
    def _setup_web3(self) -> Web3:
        """Setup Web3 connection with fallback providers"""
        providers = []
        
        if self.infura_api_key:
            providers.append(f"https://mainnet.infura.io/v3/{self.infura_api_key}")
        
        if self.alchemy_api_key:
            providers.append(f"https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}")
        
        providers.append("https://eth.llamarpc.com")  # Free backup
        
        for provider_url in providers:
            try:
                w3 = Web3(Web3.HTTPProvider(provider_url))
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                if w3.is_connected():
                    logging.info(f"Connected to Web3 provider: {provider_url[:30]}...")
                    return w3
            except Exception as e:
                logging.warning(f"Failed to connect to {provider_url[:30]}...: {e}")
        
        raise Exception("Failed to connect to any Web3 provider")
    
    def _load_config(self) -> Dict:
        """Load trading configuration"""
        default_config = {
            "max_position_size_eth": 1.0,
            "max_gas_price_gwei": 100,
            "slippage_tolerance": 0.03,
            "min_profit_threshold": 0.02,
            "copy_percentage": 0.1,  # Copy 10% of whale's trade size
            "token_whitelist": [],   # Empty = all tokens
            "token_blacklist": [],
            "min_liquidity_usd": 50000,
            "enable_auto_trade": False,
            "dry_run": True,  # Simulate trades without executing
            "monitor_interval": 30,  # seconds
            "dex_routers": {
                "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "uniswap_v3": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
                "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
            }
        }
        
        try:
            with open('trading_config.json', 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            with open('trading_config.json', 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _load_wallets(self) -> List[Dict]:
        """Load whale wallets to monitor"""
        try:
            with open(self.wallets_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _load_processed_txs(self) -> set:
        """Load already processed transaction hashes"""
        try:
            with open(self.processed_txs_file, 'r') as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def _save_processed_tx(self, tx_hash: str):
        """Save processed transaction hash"""
        self.processed_txs.add(tx_hash)
        with open(self.processed_txs_file, 'w') as f:
            json.dump(list(self.processed_txs), f)
    
    def _log_trade(self, trade_data: Dict):
        """Log executed trades"""
        try:
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
        except FileNotFoundError:
            trades = []
        
        trades.append(trade_data)
        
        with open(self.trade_log_file, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
    
    def _rate_limit_etherscan(self):
        """Ensure we don't exceed Etherscan rate limits"""
        elapsed = time.time() - self.last_etherscan_call
        if elapsed < self.etherscan_delay:
            time.sleep(self.etherscan_delay - elapsed)
        self.last_etherscan_call = time.time()
    
    def get_recent_transactions(self, address: str, hours: int = 1) -> List[Dict]:
        """Get recent transactions for a wallet"""
        self._rate_limit_etherscan()
        
        # Calculate block range
        current_block = self.w3.eth.block_number
        blocks_per_hour = 300  # ~12 sec per block
        start_block = current_block - (blocks_per_hour * hours)
        
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': current_block,
            'sort': 'desc',
            'apikey': self.etherscan_api_key
        }
        
        try:
            response = requests.get('https://api.etherscan.io/api', params=params)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    return data['result']
        except Exception as e:
            logging.error(f"Error fetching transactions: {e}")
        
        return []
    
    def decode_swap_transaction(self, tx: Dict) -> Optional[Dict]:
        """Decode DEX swap transactions"""
        if tx['isError'] == '1':
            return None
        
        # Common DEX method IDs
        swap_methods = {
            '0x7ff36ab5': 'swapExactETHForTokens',
            '0x18cbafe5': 'swapExactTokensForETH',
            '0x38ed1739': 'swapExactTokensForTokens',
            '0xfb3bdb41': 'swapETHForExactTokens',
            '0x5c11d795': 'swapExactTokensForTokensSupportingFeeOnTransferTokens'
        }
        
        method_id = tx['input'][:10] if len(tx['input']) >= 10 else None
        
        if method_id not in swap_methods:
            return None
        
        # Get transaction receipt for logs
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx['hash'])
            
            # Parse Transfer events to determine tokens involved
            transfer_topic = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
            transfers = [log for log in receipt.logs if log.topics[0].hex() == transfer_topic]
            
            if len(transfers) >= 2:
                # Determine token flow
                swap_data = {
                    'tx_hash': tx['hash'],
                    'block': tx['blockNumber'],
                    'timestamp': tx['timeStamp'],
                    'from_address': tx['from'],
                    'to_address': tx['to'],
                    'method': swap_methods[method_id],
                    'gas_price': tx['gasPrice'],
                    'gas_used': tx['gasUsed'],
                    'router': tx['to']
                }
                
                # Identify tokens and amounts
                if method_id == '0x7ff36ab5':  # ETH -> Token
                    swap_data['token_in'] = 'ETH'
                    swap_data['amount_in'] = tx['value']
                    if transfers:
                        swap_data['token_out'] = transfers[-1].address
                        swap_data['amount_out'] = int(transfers[-1].data, 16)
                
                elif method_id == '0x18cbafe5':  # Token -> ETH
                    if transfers:
                        swap_data['token_in'] = transfers[0].address
                        swap_data['amount_in'] = int(transfers[0].data, 16)
                    swap_data['token_out'] = 'ETH'
                
                else:  # Token -> Token
                    if len(transfers) >= 2:
                        swap_data['token_in'] = transfers[0].address
                        swap_data['amount_in'] = int(transfers[0].data, 16)
                        swap_data['token_out'] = transfers[-1].address
                        swap_data['amount_out'] = int(transfers[-1].data, 16)
                
                return swap_data
                
        except Exception as e:
            logging.error(f"Error decoding swap: {e}")
        
        return None
    
    def get_token_info(self, token_address: str) -> Dict:
        """Get token information"""
        if token_address in self.known_tokens:
            return self.known_tokens[token_address]
        
        if token_address == 'ETH':
            return {'symbol': 'ETH', 'decimals': 18, 'name': 'Ethereum'}
        
        try:
            # ERC20 ABI for basic functions
            erc20_abi = [
                {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"}
            ]
            
            contract = self.w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=erc20_abi)
            
            token_info = {
                'symbol': contract.functions.symbol().call(),
                'decimals': contract.functions.decimals().call(),
                'name': contract.functions.name().call(),
                'address': token_address
            }
            
            self.known_tokens[token_address] = token_info
            return token_info
            
        except Exception as e:
            logging.warning(f"Could not get info for token {token_address}: {e}")
            return {'symbol': 'UNKNOWN', 'decimals': 18, 'address': token_address}
    
    def should_copy_trade(self, swap_data: Dict) -> bool:
        """Determine if a whale trade should be copied"""
        # Check if already processed
        if swap_data['tx_hash'] in self.processed_txs:
            return False
        
        # Check token whitelist/blacklist
        token_in = swap_data.get('token_in', '')
        token_out = swap_data.get('token_out', '')
        
        if self.config['token_whitelist']:
            if token_in not in self.config['token_whitelist'] and token_out not in self.config['token_whitelist']:
                return False
        
        if self.config['token_blacklist']:
            if token_in in self.config['token_blacklist'] or token_out in self.config['token_blacklist']:
                return False
        
        # Check gas price
        gas_price_gwei = int(swap_data['gas_price']) / 1e9
        if gas_price_gwei > self.config['max_gas_price_gwei']:
            logging.info(f"Skipping trade due to high gas: {gas_price_gwei:.1f} gwei")
            return False
        
        return True
    
    async def execute_copy_trade(self, original_swap: Dict) -> Optional[Dict]:
        """Execute a copy of the whale's trade"""
        if not self.account:
            logging.error("No private key configured for trading")
            return None
        
        if self.config['dry_run']:
            logging.info("DRY RUN MODE - Trade would be executed:")
            logging.info(f"  Original: {original_swap}")
            return {'status': 'dry_run', 'original': original_swap}
        
        try:
            # Calculate copy trade size
            if original_swap.get('token_in') == 'ETH':
                original_amount = int(original_swap['amount_in'])
                copy_amount = int(original_amount * self.config['copy_percentage'])
                
                # Check max position size
                max_wei = Web3.to_wei(self.config['max_position_size_eth'], 'ether')
                copy_amount = min(copy_amount, max_wei)
                
            else:
                # For token swaps, calculate proportionally
                token_info = self.get_token_info(original_swap['token_in'])
                original_amount = int(original_swap['amount_in'])
                copy_amount = int(original_amount * self.config['copy_percentage'])
            
            # Build transaction based on swap type
            router_address = original_swap['router']
            
            # Get router ABI (simplified)
            router_abi = self._get_router_abi()
            router = self.w3.eth.contract(address=Web3.to_checksum_address(router_address), abi=router_abi)
            
            # Set deadline (10 minutes from now)
            deadline = int(time.time()) + 600
            
            # Build path
            if original_swap['token_in'] == 'ETH':
                path = [
                    self.w3.to_checksum_address('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'),  # WETH
                    self.w3.to_checksum_address(original_swap['token_out'])
                ]
            else:
                path = [
                    self.w3.to_checksum_address(original_swap['token_in']),
                    self.w3.to_checksum_address(original_swap['token_out'])
                ]
            
            # Calculate minimum output with slippage
            if 'amount_out' in original_swap:
                expected_out = int(original_swap['amount_out'] * self.config['copy_percentage'])
                min_out = int(expected_out * (1 - self.config['slippage_tolerance']))
            else:
                min_out = 0  # Use 0 for safety, but risky
            
            # Prepare transaction
            if original_swap.get('method') == 'swapExactETHForTokens':
                tx = router.functions.swapExactETHForTokens(
                    min_out,
                    path,
                    self.account.address,
                    deadline
                ).build_transaction({
                    'from': self.account.address,
                    'value': copy_amount,
                    'gas': 300000,
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address)
                })
            
            elif original_swap.get('method') == 'swapExactTokensForETH':
                # Need to approve router first
                token_contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(original_swap['token_in']),
                    abi=self._get_erc20_abi()
                )
                
                # Check allowance
                allowance = token_contract.functions.allowance(
                    self.account.address,
                    router_address
                ).call()
                
                if allowance < copy_amount:
                    # Approve router
                    approve_tx = token_contract.functions.approve(
                        router_address,
                        copy_amount
                    ).build_transaction({
                        'from': self.account.address,
                        'gas': 100000,
                        'gasPrice': self.w3.eth.gas_price,
                        'nonce': self.w3.eth.get_transaction_count(self.account.address)
                    })
                    
                    signed_approve = self.w3.eth.account.sign_transaction(approve_tx, self.private_key)
                    approve_hash = self.w3.eth.send_raw_transaction(signed_approve.rawTransaction)
                    logging.info(f"Approval tx: {approve_hash.hex()}")
                    
                    # Wait for approval
                    self.w3.eth.wait_for_transaction_receipt(approve_hash)
                
                tx = router.functions.swapExactTokensForETH(
                    copy_amount,
                    min_out,
                    path,
                    self.account.address,
                    deadline
                ).build_transaction({
                    'from': self.account.address,
                    'gas': 300000,
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address)
                })
            
            else:
                logging.warning(f"Unsupported swap method: {original_swap.get('method')}")
                return None
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logging.info(f"Copy trade sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            trade_result = {
                'status': 'success' if receipt.status == 1 else 'failed',
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt.gasUsed,
                'block': receipt.blockNumber,
                'original_tx': original_swap['tx_hash'],
                'timestamp': datetime.now().isoformat()
            }
            
            self._log_trade(trade_result)
            return trade_result
            
        except Exception as e:
            logging.error(f"Error executing copy trade: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_router_abi(self) -> List:
        """Get simplified router ABI"""
        return [
            {
                "inputs": [
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForETH",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    def _get_erc20_abi(self) -> List:
        """Get ERC20 ABI"""
        return [
            {
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "amount", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    async def monitor_wallet(self, wallet: Dict):
        """Monitor a single wallet for new trades"""
        address = wallet['address']
        name = wallet.get('name', 'Unknown')
        
        logging.info(f"Checking wallet: {name} ({address[:10]}...)")
        
        # Get recent transactions
        recent_txs = self.get_recent_transactions(address, hours=1)
        
        for tx in recent_txs:
            if tx['hash'] in self.processed_txs:
                continue
            
            # Try to decode as swap
            swap_data = self.decode_swap_transaction(tx)
            
            if swap_data:
                # Get token info
                if swap_data.get('token_in') and swap_data['token_in'] != 'ETH':
                    token_in_info = self.get_token_info(swap_data['token_in'])
                    swap_data['token_in_symbol'] = token_in_info['symbol']
                else:
                    swap_data['token_in_symbol'] = 'ETH'
                
                if swap_data.get('token_out') and swap_data['token_out'] != 'ETH':
                    token_out_info = self.get_token_info(swap_data['token_out'])
                    swap_data['token_out_symbol'] = token_out_info['symbol']
                else:
                    swap_data['token_out_symbol'] = 'ETH'
                
                logging.info(f"🐋 Whale swap detected: {swap_data['token_in_symbol']} -> {swap_data['token_out_symbol']}")
                logging.info(f"   TX: {swap_data['tx_hash']}")
                
                # Check if we should copy
                if self.should_copy_trade(swap_data):
                    if self.config['enable_auto_trade']:
                        logging.info("📋 Copying trade...")
                        result = await self.execute_copy_trade(swap_data)
                        if result:
                            logging.info(f"✅ Trade result: {result['status']}")
                    else:
                        logging.info("⚠️  Auto-trading disabled. Enable in config to copy trades.")
                
                # Mark as processed
                self._save_processed_tx(tx['hash'])
    
    async def monitor_all_wallets(self):
        """Monitor all tracked wallets"""
        while True:
            try:
                logging.info(f"\n{'='*60}")
                logging.info(f"Monitoring {len(self.monitored_wallets)} wallets...")
                
                for wallet in self.monitored_wallets:
                    await self.monitor_wallet(wallet)
                
                logging.info(f"Sleeping for {self.config['monitor_interval']} seconds...")
                await asyncio.sleep(self.config['monitor_interval'])
                
            except KeyboardInterrupt:
                logging.info("Monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def print_status(self):
        """Print current status and configuration"""
        print("\n🐋 WHALE FOLLOWER STATUS")
        print(f"{'='*60}")
        print(f"Monitoring: {len(self.monitored_wallets)} wallets")
        print(f"Auto-trading: {'ENABLED' if self.config['enable_auto_trade'] else 'DISABLED'}")
        print(f"Dry run: {'YES' if self.config['dry_run'] else 'NO'}")
        print(f"Copy percentage: {self.config['copy_percentage']*100}%")
        print(f"Max position: {self.config['max_position_size_eth']} ETH")
        print(f"Slippage tolerance: {self.config['slippage_tolerance']*100}%")
        
        if self.account:
            balance = self.w3.eth.get_balance(self.account.address)
            print(f"\nTrading wallet: {self.account.address}")
            print(f"Balance: {Web3.from_wei(balance, 'ether'):.4f} ETH")
        else:
            print("\n⚠️  No trading wallet configured!")
        
        print(f"\nProcessed transactions: {len(self.processed_txs)}")
        
        # Show recent trades
        try:
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
                if trades:
                    print(f"\nRecent trades: {len(trades)}")
                    for trade in trades[-5:]:
                        print(f"  - {trade['timestamp']}: {trade['status']} ({trade.get('tx_hash', 'N/A')[:10]}...)")
        except:
            pass

async def main():
    follower = WhaleFollower()
    
    print("\n🐋 WHALE TRADE FOLLOWER")
    print("1. Start monitoring")
    print("2. Show status")
    print("3. Toggle auto-trading")
    print("4. Configure settings")
    print("5. Exit")
    
    choice = input("\nSelect option: ")
    
    if choice == '1':
        follower.print_status()
        print("\nStarting monitoring... Press Ctrl+C to stop")
        await follower.monitor_all_wallets()
    
    elif choice == '2':
        follower.print_status()
    
    elif choice == '3':
        follower.config['enable_auto_trade'] = not follower.config['enable_auto_trade']
        with open('trading_config.json', 'w') as f:
            json.dump(follower.config, f, indent=2)
        print(f"Auto-trading: {'ENABLED' if follower.config['enable_auto_trade'] else 'DISABLED'}")
    
    elif choice == '4':
        print("\nEdit trading_config.json to change settings")
        print("Current config:")
        print(json.dumps(follower.config, indent=2))
    
    elif choice == '5':
        print("Goodbye! 🐋")
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    asyncio.run(main())