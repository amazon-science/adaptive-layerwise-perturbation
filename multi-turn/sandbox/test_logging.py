#!/usr/bin/env python3
"""
Test sandbox API logging functionality
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_sandbox_execution():
    """Test sandbox execution and view logs"""
    
    # Test code
    test_cases = [
        {
            "code": "print('Hello, World!')",
            "language": "python",
            "run_timeout": 30.0
        },
        {
            "code": "print(2 + 2)\nprint('Math is fun!')",
            "language": "python", 
            "run_timeout": 30.0
        },
        {
            "code": "import sys\nprint('Python version:', sys.version)",
            "language": "python",
            "run_timeout": 30.0
        },
        {
            "code": "print('This will cause an error')\nraise ValueError('Test error')",
            "language": "python",
            "run_timeout": 30.0
        }
    ]
    
    print("🧪 Starting sandbox execution tests...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test case {i}:")
        print(f"Code: {test_case['code'][:50]}{'...' if len(test_case['code']) > 50 else ''}")
        
        # Execute code
        response = requests.post(
            f"{API_BASE}/faas/sandbox/",
            json=test_case,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Execution succeeded: {result['status']}")
            print(f"Output: {result['run_result']['stdout'][:100]}{'...' if len(result['run_result']['stdout']) > 100 else ''}")
            if result['run_result']['stderr']:
                print(f"Error: {result['run_result']['stderr'][:100]}{'...' if len(result['run_result']['stderr']) > 100 else ''}")
        else:
            print(f"❌ Execution failed: {response.status_code}")
            print(response.text)
        
        time.sleep(0.5)  # Avoid execution ID conflicts

def test_log_endpoints():
    """Test log viewing endpoints"""
    
    print("\n📋 Testing log endpoints...")
    
    # List all logs
    print("\n1. Listing all log files:")
    response = requests.get(f"{API_BASE}/logs/")
    if response.status_code == 200:
        logs_data = response.json()
        print(f"Found {logs_data['total']} log files")
        
        for log_info in logs_data['logs'][:5]:  # Only show first 5
            print(f"  - {log_info['filename']} ({log_info['size']} bytes)")
            print(f"    Created: {log_info['created']}")
            print(f"    Modified: {log_info['modified']}")
    else:
        print(f"❌ Failed to get log list: {response.status_code}")
        return
    
    # Get latest log details
    if logs_data['logs']:
        latest_log = logs_data['logs'][0]
        execution_id = latest_log['filename'].replace('execution_', '').replace('.json', '')
        
        print(f"\n2. Getting latest log details (ID: {execution_id}):")
        response = requests.get(f"{API_BASE}/logs/{execution_id}")
        if response.status_code == 200:
            log_detail = response.json()
            print(f"Execution ID: {log_detail['execution_id']}")
            print(f"Timestamp: {log_detail['timestamp']}")
            print(f"Code:\n{log_detail['code']}")
            print(f"Input: {log_detail['stdin']}")
            print(f"Result status: {log_detail['result']['status']}")
            print(f"Output: {log_detail['result']['stdout']}")
            if log_detail['result']['stderr']:
                print(f"Error: {log_detail['result']['stderr']}")
        else:
            print(f"❌ Failed to get log details: {response.status_code}")

def main():
    """Main function"""
    print("🚀 Sandbox API Log Functionality Test")
    print("=" * 50)
    
    try:
        # Test sandbox execution
        test_sandbox_execution()
        
        # Test log endpoints
        test_log_endpoints()
        
        print("\n✅ Tests complete!")
        print("\n📁 Log file locations:")
        print("  - Execution logs: ./sandbox_logs/executions.json")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to sandbox API")
        print("Please ensure the API service is running: uvicorn sandbox_api:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main()
