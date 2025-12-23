#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load Testing Script for Quantitative Analytics Dashboard
Tests application under concurrent user load using Locust

Installation:
    pip install locust

Usage:
    locust -f load_test.py --headless -u 10 -r 2 -t 5m
    
    -u 10 = 10 total concurrent users
    -r 2 = Spawn 2 users per second
    -t 5m = Run for 5 minutes

Real-world load test:
    locust -f load_test.py --headless -u 50 -r 5 -t 10m --host=http://localhost:8501

For interactive UI:
    locust -f load_test.py --host=http://localhost:8501
"""

from locust import HttpUser, task, constant, between
import logging
import random
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of popular stocks to test with
TEST_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "WMT",
    "BRK.B", "JNJ", "XOM", "PG", "MA"
]


class QuantEngineUser(HttpUser):
    """Simulates a user of the Quant Engine application"""
    
    # Define wait time between tasks (2-5 seconds, mimics real user)
    wait_time = between(2, 5)
    
    # Track user session
    user_id = None
    session_token = None
    
    def on_start(self):
        """Called when a Locust user starts"""
        self.user_id = f"load_test_user_{random.randint(1000, 9999)}"
        logger.info(f"User started: {self.user_id}")
    
    @task(3)  # Weight: 3 (runs 3x more often than weight-1 tasks)
    def load_dashboard(self):
        """Task: Load the main dashboard"""
        with self.client.get(
            "/",
            catch_response=True,
            headers={"X-User-ID": self.user_id}
        ) as response:
            if response.status_code == 200:
                response.success()
                logger.debug(f"Dashboard loaded for {self.user_id}")
            else:
                response.fail(f"Unexpected status code: {response.status_code}")
    
    @task(5)  # Weight: 5 (most common operation)
    def fetch_stock_data(self):
        """Task: Fetch data for a random stock"""
        ticker = random.choice(TEST_TICKERS)
        
        # Simulate URL parameter (in real Streamlit, this is via st.query_params)
        with self.client.get(
            f"/?ticker={ticker}",
            catch_response=True,
            headers={
                "X-User-ID": self.user_id,
                "User-Agent": f"LoadTest/{self.user_id}"
            }
        ) as response:
            if response.status_code == 200:
                response.success()
                logger.debug(f"Fetched data for {ticker}")
            elif response.status_code == 429:
                response.fail("Rate limit exceeded (429)")
            elif response.status_code == 503:
                response.fail("Service unavailable (503)")
            else:
                response.fail(f"Status {response.status_code}")
    
    @task(2)  # Weight: 2
    def fetch_comparison_data(self):
        """Task: Fetch comparison/benchmark data"""
        ticker1 = random.choice(TEST_TICKERS)
        ticker2 = random.choice(TEST_TICKERS)
        
        with self.client.get(
            f"/?ticker={ticker1}&benchmark={ticker2}",
            catch_response=True,
            headers={"X-User-ID": self.user_id}
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.fail(f"Comparison failed: {response.status_code}")
    
    @task(1)  # Weight: 1 (lowest priority)
    def simulate_navigation(self):
        """Task: Simulate user clicking between different sections"""
        sections = ["overview", "analysis", "comparison", "settings"]
        section = random.choice(sections)
        
        with self.client.get(
            f"/?section={section}",
            catch_response=True,
            headers={"X-User-ID": self.user_id}
        ) as response:
            if response.status_code in [200, 304]:
                response.success()
            else:
                response.fail(f"Navigation failed: {response.status_code}")


class AdminUser(HttpUser):
    """Simulates an admin user (less frequent)"""
    
    wait_time = between(5, 10)
    
    def on_start(self):
        """Called when a Locust user starts"""
        self.admin_id = f"admin_load_test_{random.randint(1, 100)}"
        logger.info(f"Admin user started: {self.admin_id}")
    
    @task(1)
    def check_monitoring(self):
        """Task: Admin checking monitoring dashboard"""
        with self.client.get(
            "/?page=monitoring",
            catch_response=True,
            headers={
                "X-User-ID": self.admin_id,
                "X-Admin": "true"
            }
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.fail(f"Monitoring check failed: {response.status_code}")
    
    @task(1)
    def view_audit_logs(self):
        """Task: Admin viewing audit logs"""
        with self.client.get(
            "/?page=audit_logs",
            catch_response=True,
            headers={"X-Admin": "true"}
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.fail(f"Audit logs failed: {response.status_code}")


# ============================================================================
# EXECUTION INSTRUCTIONS
# ============================================================================
"""
QUICK TEST (2 minutes, 5 users):
    locust -f load_test.py --headless -u 5 -r 1 -t 2m

STANDARD TEST (5 minutes, 10 users):
    locust -f load_test.py --headless -u 10 -r 2 -t 5m

STRESS TEST (10 minutes, 25 users):
    locust -f load_test.py --headless -u 25 -r 3 -t 10m

SCALING TEST (15 minutes, 50 users):
    locust -f load_test.py --headless -u 50 -r 5 -t 15m

INTERACTIVE WEB UI (start Locust web interface):
    locust -f load_test.py --host=http://localhost:8501
    Then navigate to http://localhost:8089 in your browser

EXPECTED RESULTS:
    - Response time p95: < 5 seconds
    - Response time p99: < 10 seconds
    - Error rate: < 1%
    - Failure rate: < 0.5%

WHAT TO MONITOR:
    1. Response time trends (should be stable)
    2. Error rate (should stay <1%)
    3. CPU usage (should not max out)
    4. Memory usage (should be stable)
    5. Database query times (should stay <50ms)

ANALYSIS AFTER TEST:
    1. Review response time percentiles
    2. Check error logs for failures
    3. Verify rate limiting is working
    4. Check quota system is enforcing limits
    5. Measure actual throughput (requests/second)
"""
