#!/usr/bin/env python3
"""
Whale Wallet Analyzer - Historical Data Collection & Analysis
Collects and analyzes all transactions from specified Ethereum wallets
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

class WalletAnalyzer:
    def __init__(self):
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.etherscan_base_url = "https://api.etherscan.io/api"
        self.w3 = Web3()
        self.wallets_file = "wallets.json"
        self.rate_limit_delay = 0.25  # Etherscan rate limit: 5 calls/sec
        
    def load_wallets(self) -> List[Dict[str, str]]:
        """Load wallet addresses from JSON file"""
        try:
            with open(self.wallets_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default wallets file
            default_wallets = [
                {
                    "address": "0x2ba553d9f990a3b66b03b2dc0d030dfc1c061036",
                    "name": "Main Whale",
                    "notes": "Primary tracking wallet"
                }
            ]
            self.save_wallets(default_wallets)
            return default_wallets
    
    def save_wallets(self, wallets: List[Dict[str, str]]):
        """Save wallets to JSON file"""
        with open(self.wallets_file, 'w') as f:
            json.dump(wallets, f, indent=2)
    
    def add_wallet(self, address: str, name: str = "", notes: str = ""):
        """Add new wallet to tracking list"""
        wallets = self.load_wallets()
        
        # Validate address
        if not Web3.is_address(address):
            raise ValueError(f"Invalid Ethereum address: {address}")
        
        # Check if already exists
        checksum_address = Web3.to_checksum_address(address)
        if any(w['address'].lower() == checksum_address.lower() for w in wallets):
            print(f"Wallet {checksum_address} already in tracking list")
            return
        
        wallets.append({
            "address": checksum_address,
            "name": name or f"Wallet_{len(wallets)+1}",
            "notes": notes,
            "added_date": datetime.now().isoformat()
        })
        
        self.save_wallets(wallets)
        print(f"Added wallet: {checksum_address} ({name})")
    
    def get_transactions(self, address: str, start_block: int = 0) -> List[Dict]:
        """Get all transactions for a wallet address"""
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': self.etherscan_api_key
        }
        
        response = requests.get(self.etherscan_base_url, params=params)
        time.sleep(self.rate_limit_delay)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                return data['result']
            else:
                print(f"Error fetching transactions: {data.get('message', 'Unknown error')}")
                return []
        else:
            print(f"HTTP Error: {response.status_code}")
            return []
    
    def get_token_transfers(self, address: str) -> List[Dict]:
        """Get all ERC20 token transfers for a wallet"""
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': self.etherscan_api_key
        }
        
        response = requests.get(self.etherscan_base_url, params=params)
        time.sleep(self.rate_limit_delay)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                return data['result']
            else:
                return []
        else:
            return []
    
    def get_nft_transfers(self, address: str) -> List[Dict]:
        """Get all NFT transfers for a wallet"""
        params = {
            'module': 'account',
            'action': 'tokennfttx',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': self.etherscan_api_key
        }
        
        response = requests.get(self.etherscan_base_url, params=params)
        time.sleep(self.rate_limit_delay)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                return data['result']
            else:
                return []
        else:
            return []
    
    def analyze_transactions(self, transactions: List[Dict], address: str) -> Dict[str, Any]:
        """Analyze transaction patterns and statistics"""
        if not transactions:
            return {"error": "No transactions found"}
        
        df = pd.DataFrame(transactions)
        
        # Convert values safely
        df['value_eth'] = df['value'].apply(lambda x: float(Web3.from_wei(int(x), 'ether')))
        df['timestamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
        df['gas_cost_eth'] = (df['gasUsed'].astype(float) * df['gasPrice'].astype(float)) / 1e18
        
        # Determine if transaction is incoming or outgoing
        df['direction'] = df['from'].apply(lambda x: 'OUT' if x.lower() == address.lower() else 'IN')
        
        # Calculate date range safely
        date_range_days = (df['timestamp'].max() - df['timestamp'].min()).days
        if date_range_days == 0:
            date_range_days = 1
        
        analysis = {
            'total_transactions': len(df),
            'first_transaction': df['timestamp'].min().isoformat(),
            'last_transaction': df['timestamp'].max().isoformat(),
            'total_eth_sent': float(df[df['direction'] == 'OUT']['value_eth'].sum()),
            'total_eth_received': float(df[df['direction'] == 'IN']['value_eth'].sum()),
            'total_gas_spent': float(df['gas_cost_eth'].sum()),
            'unique_addresses_interacted': len(set(df['from'].tolist() + df['to'].tolist())) - 1,
            'failed_transactions': len(df[df['isError'] == '1']),
            'avg_transaction_value': float(df['value_eth'].mean()),
            'max_transaction_value': float(df['value_eth'].max()),
            'transaction_frequency': {
                'daily_avg': len(df) / date_range_days,
                'monthly_breakdown': df.groupby(df['timestamp'].dt.strftime('%Y-%m')).size().to_dict()
            }
        }
        
        # Find most interacted addresses
        all_addresses = df[df['from'] == address.lower()]['to'].tolist() + \
                       df[df['to'] == address.lower()]['from'].tolist()
        if all_addresses:
            address_counts = pd.Series(all_addresses).value_counts()
            analysis['top_interacted_addresses'] = {str(k): int(v) for k, v in address_counts.head(10).items()}
        else:
            analysis['top_interacted_addresses'] = {}
        
        return analysis
    
    def analyze_tokens(self, token_transfers: List[Dict], address: str) -> Dict[str, Any]:
        """Analyze token trading patterns"""
        if not token_transfers:
            return {"error": "No token transfers found"}
        
        df = pd.DataFrame(token_transfers)
        df['timestamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
        df['direction'] = df['from'].apply(lambda x: 'SELL' if x.lower() == address.lower() else 'BUY')
        
        # Group by token
        token_analysis = {}
        for token in df['tokenSymbol'].unique():
            token_df = df[df['tokenSymbol'] == token]
            
            try:
                decimals = int(token_df['tokenDecimal'].iloc[0])
            except:
                decimals = 18  # Default to 18 if parsing fails
            
            token_analysis[str(token)] = {
                'contract': str(token_df['contractAddress'].iloc[0]),
                'total_bought': float(token_df[token_df['direction'] == 'BUY']['value'].sum()) / (10**decimals),
                'total_sold': float(token_df[token_df['direction'] == 'SELL']['value'].sum()) / (10**decimals),
                'transaction_count': int(len(token_df)),
                'first_transaction': token_df['timestamp'].min().isoformat(),
                'last_transaction': token_df['timestamp'].max().isoformat(),
                'unique_addresses': len(set(token_df['from'].tolist() + token_df['to'].tolist())) - 1
            }
            
            # Calculate net position
            token_analysis[str(token)]['net_position'] = token_analysis[str(token)]['total_bought'] - token_analysis[str(token)]['total_sold']
        
        return {
            'total_unique_tokens': len(token_analysis),
            'total_token_transactions': len(df),
            'tokens': token_analysis,
            'most_traded_token': max(token_analysis.items(), key=lambda x: x[1]['transaction_count'])[0] if token_analysis else None
        }
    
    def generate_report(self, wallet_info: Dict, save_to_file: bool = True) -> str:
        """Generate comprehensive analysis report"""
        address = wallet_info['address']
        name = wallet_info.get('name', 'Unknown')
        
        print(f"\n{'='*60}")
        print(f"Analyzing wallet: {name} ({address})")
        print(f"{'='*60}")
        
        # Get all transaction types
        print("Fetching ETH transactions...")
        eth_txs = self.get_transactions(address)
        
        print("Fetching token transfers...")
        token_txs = self.get_token_transfers(address)
        
        print("Fetching NFT transfers...")
        nft_txs = self.get_nft_transfers(address)
        
        # Analyze data
        eth_analysis = self.analyze_transactions(eth_txs, address)
        token_analysis = self.analyze_tokens(token_txs, address)
        
        report = {
            'wallet_info': wallet_info,
            'analysis_date': datetime.now().isoformat(),
            'eth_analysis': eth_analysis,
            'token_analysis': token_analysis,
            'nft_count': len(nft_txs),
            'summary': {
                'total_transactions': len(eth_txs) + len(token_txs) + len(nft_txs),
                'eth_balance_change': eth_analysis.get('total_eth_received', 0) - eth_analysis.get('total_eth_sent', 0),
                'gas_spent_eth': eth_analysis.get('total_gas_spent', 0),
                'tokens_traded': token_analysis.get('total_unique_tokens', 0)
            }
        }
        
        if save_to_file:
            filename = f"analysis_{address[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nAnalysis saved to: {filename}")
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"Total Transactions: {report['summary']['total_transactions']}")
        print(f"ETH In: {eth_analysis.get('total_eth_received', 0):.4f} ETH")
        print(f"ETH Out: {eth_analysis.get('total_eth_sent', 0):.4f} ETH")
        print(f"Net ETH: {report['summary']['eth_balance_change']:.4f} ETH")
        print(f"Gas Spent: {report['summary']['gas_spent_eth']:.4f} ETH")
        print(f"Unique Tokens Traded: {report['summary']['tokens_traded']}")
        print(f"NFT Transfers: {report['nft_count']}")
        
        if token_analysis.get('tokens'):
            print(f"\n🪙 TOP TOKENS BY ACTIVITY:")
            sorted_tokens = sorted(token_analysis['tokens'].items(), 
                                 key=lambda x: x[1]['transaction_count'], 
                                 reverse=True)[:5]
            for token, data in sorted_tokens:
                print(f"  {token}: {data['transaction_count']} txs, Net: {data['net_position']:.2f}")
        
        return json.dumps(report, indent=2, default=str)
    
    def analyze_all_wallets(self):
        """Analyze all tracked wallets"""
        wallets = self.load_wallets()
        all_reports = []
        
        for wallet in wallets:
            report = self.generate_report(wallet)
            all_reports.append(json.loads(report))
            print("\n" + "="*60 + "\n")
        
        # Save combined report
        combined_report = {
            'analysis_date': datetime.now().isoformat(),
            'wallets_analyzed': len(wallets),
            'reports': all_reports
        }
        
        filename = f"combined_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(combined_report, f, indent=2, default=str)
        
        print(f"\nCombined analysis saved to: {filename}")

def main():
    analyzer = WalletAnalyzer()
    
    while True:
        print("\n🐋 WHALE WALLET ANALYZER")
        print("1. Analyze all tracked wallets")
        print("2. Add new wallet")
        print("3. View tracked wallets")
        print("4. Analyze specific wallet")
        print("5. Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == '1':
            analyzer.analyze_all_wallets()
        
        elif choice == '2':
            address = input("Enter wallet address: ")
            name = input("Enter wallet name (optional): ")
            notes = input("Enter notes (optional): ")
            try:
                analyzer.add_wallet(address, name, notes)
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            wallets = analyzer.load_wallets()
            print("\n📋 TRACKED WALLETS:")
            for i, wallet in enumerate(wallets, 1):
                print(f"{i}. {wallet['name']} - {wallet['address']}")
                if wallet.get('notes'):
                    print(f"   Notes: {wallet['notes']}")
        
        elif choice == '4':
            wallets = analyzer.load_wallets()
            for i, wallet in enumerate(wallets, 1):
                print(f"{i}. {wallet['name']} - {wallet['address'][:10]}...")
            
            try:
                idx = int(input("\nSelect wallet number: ")) - 1
                if 0 <= idx < len(wallets):
                    analyzer.generate_report(wallets[idx])
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")
        
        elif choice == '5':
            print("Goodbye! 🐋")
            break
        
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()