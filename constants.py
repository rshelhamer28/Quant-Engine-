# constants.py - Centralized constants to avoid magic strings and magic numbers
# This file defines all hardcoded values and string literals used throughout the app
# Purpose: Enable easy updates and reduce code duplication

import os
from pathlib import Path

# =================================================================
# APP CONFIGURATION
# =================================================================

APP_NAME = "Quantitative Analytics Dashboard"
APP_VERSION = "1.0"

# =================================================================
# DATA SOURCE SCHEMAS - yfinance
# =================================================================

# Expected price column names (in order of preference)
YFINANCE_PRICE_COLUMNS = {
    'primary': 'Adj Close',
    'fallbacks': ['Close', 'Adj_Close', 'CLOSE'],
    'all': ['Adj Close', 'Close', 'Adj_Close', 'CLOSE']
}

# Expected yfinance ticker.info fields (critical vs optional)
YFINANCE_TICKER_FIELDS = {
    'critical': ['currentPrice', 'marketCap', 'sector'],
    'optional': ['trailingPE', 'forwardPE', 'targetMeanPrice', 'shortPercentOfFloat',
                 'beta', 'industry', 'longName', 'shortRatio', 'heldPercentInstitutions']
}

# yfinance schema version tracking
YFINANCE_SCHEMA_VERSION = "0.2.30"

# =================================================================
# DATA SOURCE SCHEMAS - Finnhub
# =================================================================

# Expected Finnhub news response fields
FINNHUB_NEWS_FIELDS = {
    'required': ['headline', 'source', 'datetime'],
    'optional': ['summary', 'url', 'image', 'sentiment']
}

# Finnhub API version
FINNHUB_API_VERSION = "v1"

# =================================================================
# DATA QUALITY THRESHOLDS
# =================================================================

# Minimum data requirements for analysis
MIN_TRADING_DAYS = int(os.getenv('MIN_TRADING_DAYS', '20'))
MIN_DATA_COMPLETENESS = float(os.getenv('MIN_DATA_COMPLETENESS', '0.70'))

# Thresholds for different market cap categories
DATA_QUALITY_BY_MARKET_CAP = {
    'mega': {'min_trading_days': 10, 'min_completeness': 0.60},
    'large': {'min_trading_days': 20, 'min_completeness': 0.70},
    'mid': {'min_trading_days': 50, 'min_completeness': 0.80},
    'small': {'min_trading_days': 100, 'min_completeness': 0.90}
}

# =================================================================
# MARKET & BENCHMARK CONSTANTS
# =================================================================

TRADING_DAYS_PER_YEAR = 252
BENCHMARK_TICKER = os.getenv('BENCHMARK_TICKER', '^GSPC')

# =================================================================
# API TIMEOUTS & RETRIES
# =================================================================

YFINANCE_TIMEOUT = int(os.getenv('YFINANCE_TIMEOUT', '15'))
NEWS_API_TIMEOUT = int(os.getenv('NEWS_API_TIMEOUT', '10'))

RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 0.5
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# =================================================================
# CACHE CONFIGURATION
# =================================================================

CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '300'))
PRICE_HISTORY_PERIOD = os.getenv('PRICE_HISTORY_PERIOD', '5y')

# =================================================================
# MONTE CARLO SIMULATION
# =================================================================

MONTE_CARLO_PATHS = int(os.getenv('MONTE_CARLO_PATHS', '10000'))
MONTE_CARLO_FORECAST_YEARS = [1, 2, 5]
MONTE_CARLO_VOLATILITY_CAP = 0.80  # Don't allow vol > 80%

# =================================================================
# SCORING WEIGHTS & THRESHOLDS
# =================================================================

# Thresholds for conviction scoring (in order of magnitude)
SCORING_THRESHOLDS = {
    'vol_adjusted_return_excellent': 0.8,
    'vol_adjusted_return_good': 0.5,
    'vol_adjusted_return_poor': 0.2,
    
    'upside_ratio_excellent': 1.2,
    'upside_ratio_poor': 0.8,
    
    'beta_volatility_low': 0.2,
    'beta_volatility_high': 0.4,
    
    'short_float_high': 0.10,
    'short_float_low': 0.02,
    
    'peg_ratio_excellent': 0.8,
    'peg_ratio_good': 1.2,
    'peg_ratio_poor': 2.0,
    
    'profit_margin_threshold': 0.15,
    'revenue_growth_threshold': 0.10,
    
    'pe_sector_multiple_low': 0.8,
    'pe_sector_multiple_high': 1.2,
    
    'min_data_years': 2,
    'good_data_years': 8,
}

# VaR confidence level
VAR_CONFIDENCE_LEVEL = 0.95

# =================================================================
# FILE PATHS & DIRECTORIES
# =================================================================

# Base directory
BASE_DIR = Path(__file__).parent

# Logging
LOG_DIR = Path(os.getenv('LOG_DIR', BASE_DIR / 'logs'))
LOG_FILE_TEMPLATE = 'quant_engine_{hostname}_{timestamp}.log'

# Data files
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
SENTIMENT_DICT_PATH = Path(os.getenv('SENTIMENT_DICT_PATH', DATA_DIR / 'sentiment_dict.json'))
SECTOR_PE_PATH = Path(os.getenv('SECTOR_PE_PATH', DATA_DIR / 'sector_pe_latest.json'))
SCORING_WEIGHTS_PATH = Path(os.getenv('SCORING_WEIGHTS_PATH', DATA_DIR / 'scoring_weights.json'))

# Database
CACHE_DB_PATH = Path(os.getenv('CACHE_DB_PATH', DATA_DIR / 'price_cache.db'))

# =================================================================
# API CONFIGURATION
# =================================================================

NEWS_API_KEY = os.getenv('FINNHUB_API_KEY')
NEWS_API_BASE_URL = os.getenv('FINNHUB_API_URL', 'https://finnhub.io/api/v1')
NEWS_API_ENDPOINT = f"{NEWS_API_BASE_URL}/company-news"

# =================================================================
# STREAMLIT CONFIGURATION
# =================================================================

STREAMLIT_PAGE_TITLE = "Quantitative Analytics Dashboard"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_SIDEBAR_STATE = "collapsed"

# =================================================================
# ERROR MESSAGES & FALLBACKS
# =================================================================

ERROR_INSUFFICIENT_DATA = "Insufficient trading history for analysis"
ERROR_DATA_FETCH_FAILED = "Failed to fetch data from source"
ERROR_SCHEMA_MISMATCH = "Data source schema has changed. Using fallback data."

FALLBACK_DATA_MESSAGE = "Using cached/sample data due to API unavailability"

# =================================================================
# DATA SOURCE FIELD NAMES
# =================================================================

# Column names used internally (canonical names)
PRICE_COLUMN_CANONICAL = 'Price'
DATE_COLUMN_CANONICAL = 'Date'

# Mapping of external field names to internal names
PRICE_DATA_FIELD_MAPPING = {
    'Adj Close': PRICE_COLUMN_CANONICAL,
    'Close': PRICE_COLUMN_CANONICAL,
    'price': PRICE_COLUMN_CANONICAL,
    'Price': PRICE_COLUMN_CANONICAL
}

FINNHUB_FIELD_MAPPING = {
    'headline': 'title',
    'title': 'title',
    'source': 'publisher',
    'publisher': 'publisher',
    'datetime': 'date',
    'date': 'date',
    'summary': 'summary',
    'content': 'summary',
    'url': 'url',
    'image': 'image'
}

# =================================================================
# UTILITY CONSTANTS
# =================================================================

# Market cap sizes for categorization
MARKET_CAP_SIZES = {
    'mega': {'min': 200e9, 'label': 'Mega-cap (>$200B)'},
    'large': {'min': 10e9, 'label': 'Large-cap (>$10B)'},
    'mid': {'min': 2e9, 'label': 'Mid-cap (>$2B)'},
    'small': {'min': 300e6, 'label': 'Small-cap (>$300M)'},
    'micro': {'min': 0, 'label': 'Micro-cap (<$300M)'}
}

# =================================================================
# DOCUMENTATION & COMMENTS
# =================================================================

"""
HOW TO USE THIS FILE:

1. Import constants instead of using magic strings:
   
   FROM (bad):
   hist.columns = ['Adj Close', 'Close', 'Adj_Close', 'CLOSE']
   
   TO (good):
   from constants import YFINANCE_PRICE_COLUMNS
   hist.columns = YFINANCE_PRICE_COLUMNS['all']

2. Update configuration:
   
   FROM (bad): 
   if price < 100: ...  # Magic number
   
   TO (good):
   from constants import SCORING_THRESHOLDS
   if price < SCORING_THRESHOLDS['price_threshold']: ...

3. Add new constants here, not scattered throughout code

4. Document the source/reasoning for each value

5. Centralize environment variable loading here
"""
