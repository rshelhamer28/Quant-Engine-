#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite Runner
Runs all unit tests and integration tests with detailed reporting
"""

import sys
import unittest
import os
from io import StringIO
from datetime import datetime

# Fix Unicode encoding for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() == 'cp1252':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


class TestResults:
    """Track and report test results"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        self.start_time = None
        self.end_time = None
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY".center(70))
        print("=" * 70)
        print(f"Total Tests:    {self.total_tests}")
        print(f"✅ Passed:       {self.passed}")
        print(f"❌ Failed:       {self.failed}")
        print(f"⚠️  Skipped:      {self.skipped}")
        print(f"🔴 Errors:       {self.errors}")
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"Duration:       {duration:.2f}s")
        
        success_rate = (
            (self.passed / self.total_tests * 100)
            if self.total_tests > 0 else 0
        )
        print(f"Success Rate:   {success_rate:.1f}%")
        print("=" * 70)
        
        # Overall status
        if self.failed == 0 and self.errors == 0:
            print("✅ ALL TESTS PASSED!".center(70))
        else:
            print("⚠️ SOME TESTS FAILED - SEE DETAILS ABOVE".center(70))
        print("=" * 70 + "\n")


def run_test_suite():
    """Run complete test suite"""
    
    results = TestResults()
    results.start_time = datetime.now()
    
    # Create test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load all tests
    print("Loading tests...")
    
    try:
        # Load utility tests
        import test_utilities
        utility_tests = loader.loadTestsFromModule(test_utilities)
        suite.addTests(utility_tests)
        print("✓ Loaded test_utilities.py")
    except Exception as e:
        print(f"✗ Failed to load test_utilities: {e}")
    
    try:
        # Load integration tests
        import test_integration
        integration_tests = loader.loadTestsFromModule(test_integration)
        suite.addTests(integration_tests)
        print("✓ Loaded test_integration.py")
    except Exception as e:
        print(f"✗ Failed to load test_integration: {e}")
    
    try:
        # Load data pipeline tests
        import test_data_pipeline
        pipeline_tests = loader.loadTestsFromModule(test_data_pipeline)
        suite.addTests(pipeline_tests)
        print("✓ Loaded test_data_pipeline.py")
    except Exception as e:
        print(f"✗ Failed to load test_data_pipeline: {e}")
    
    # Run tests with detailed output
    print("\n" + "=" * 70)
    print("RUNNING TESTS".center(70))
    print("=" * 70 + "\n")
    
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    # Calculate results
    results.total_tests = test_result.testsRun
    results.passed = test_result.testsRun - (
        len(test_result.failures) +
        len(test_result.errors) +
        len(test_result.skipped)
    )
    results.failed = len(test_result.failures)
    results.errors = len(test_result.errors)
    results.skipped = len(test_result.skipped)
    results.end_time = datetime.now()
    
    # Print summary
    results.print_summary()
    
    # Return exit code
    return 0 if (results.failed == 0 and results.errors == 0) else 1


if __name__ == '__main__':
    exit_code = run_test_suite()
    sys.exit(exit_code)
