# debug_marketaux.py
"""
Debug script to test MarketAux API and find correct parameters
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def test_marketaux_api():
    """Test different MarketAux API parameters to find what works"""
    
    api_token = os.getenv("MARKETAUX_API_TOKEN")
    if not api_token:
        print("❌ MARKETAUX_API_TOKEN not found in environment")
        return
    
    base_url = "https://api.marketaux.com/v1/news/all"
    
    # Test different parameter combinations
    test_cases = [
        {
            "name": "Basic request",
            "params": {
                'api_token': api_token,
                'limit': 5
            }
        },
        {
            "name": "With symbols",
            "params": {
                'symbols': 'XAUUSD',
                'api_token': api_token,
                'limit': 5
            }
        },
        {
            "name": "With search term",
            "params": {
                'search': 'gold',
                'api_token': api_token,
                'limit': 5
            }
        },
        {
            "name": "With date filter",
            "params": {
                'published_after': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'api_token': api_token,
                'limit': 5
            }
        },
        {
            "name": "Simple gold search",
            "params": {
                'search': 'gold',
                'language': 'en',
                'api_token': api_token,
                'limit': 10
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Parameters: {test_case['params']}")
        
        try:
            response = requests.get(base_url, params=test_case['params'], timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                print(f"✅ Success! Retrieved {len(articles)} articles")
                
                if articles:
                    # Show first article as example
                    first_article = articles[0]
                    print(f"   📰 Example: {first_article.get('title', 'No title')}")
                    print(f"   📊 Sentiment: {first_article.get('sentiment', 'No sentiment')}")
                    print(f"   🏢 Source: {first_article.get('source', 'Unknown')}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 50)

def test_working_marketaux_request():
    """Test a simplified MarketAux request that should work"""
    
    api_token = os.getenv("MARKETAUX_API_TOKEN")
    if not api_token:
        print("❌ MARKETAUX_API_TOKEN not found")
        return None
    
    print("🔍 Testing simplified MarketAux request...")
    
    try:
        params = {
            'search': 'gold',
            'language': 'en',
            'limit': 10,
            'api_token': api_token
        }
        
        response = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', [])
            print(f"✅ Success! Found {len(articles)} articles")
            
            for i, article in enumerate(articles[:3], 1):
                title = article.get('title', 'No title')
                sentiment = article.get('sentiment', {})
                source = article.get('source', 'Unknown')
                
                print(f"   {i}. {source}: {title[:60]}...")
                if sentiment:
                    print(f"      📊 Sentiment: {sentiment}")
            
            return articles
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

if __name__ == "__main__":
    print("🔧 MarketAux API Debug Tool")
    print("=" * 50)
    
    # First try simple working request
    working_articles = test_working_marketaux_request()
    
    if working_articles:
        print(f"\n✅ Found working MarketAux configuration!")
    else:
        print(f"\n🧪 Testing different parameter combinations...")
        test_marketaux_api()