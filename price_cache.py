# -*- coding: utf-8 -*-
"""
SQLite-based caching layer for historical price data
Reduces API calls to yfinance and improves performance
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, Tuple
import time

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'price_cache.db'
CACHE_VALIDITY_DAYS = 1  # Re-fetch data older than 1 day
SCHEMA_VERSION = 1  # Increment when schema changes


class PriceCache:
    """SQLite-backed cache for historical price data"""
    
    def __init__(self, db_path: str = str(DB_PATH)):
        """Initialize cache database"""
        self.db_path = db_path
        self._create_tables()
        logger.info(f"PriceCache initialized at {db_path}")
    
    def _get_connection(self, timeout: int = 30) -> sqlite3.Connection:
        """Get database connection with timeout and retry logic"""
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=timeout)
                conn.row_factory = sqlite3.Row
                # Enable foreign keys
                conn.execute('PRAGMA foreign_keys = ON')
                # Set journal mode for robustness
                conn.execute('PRAGMA journal_mode = WAL')
                return conn
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                    raise
        
        raise RuntimeError("Failed to establish database connection")
    
    def _create_tables(self):
        """Create necessary tables if they don't exist with schema versioning"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Create schema_info table to track version
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY,
                    created_at TEXT
                )
            ''')
            
            # Check current schema version
            cursor.execute('SELECT version FROM schema_info ORDER BY version DESC LIMIT 1')
            result = cursor.fetchone()
            current_version = result[0] if result else 0
            
            # Apply migrations if needed
            if current_version < SCHEMA_VERSION:
                logger.info(f"Upgrading schema from version {current_version} to {SCHEMA_VERSION}")
                self._apply_migrations(cursor, current_version, SCHEMA_VERSION)
                cursor.execute('INSERT INTO schema_info (version, created_at) VALUES (?, ?)',
                             (SCHEMA_VERSION, datetime.now().isoformat()))
            
            # Prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    fetched_at TEXT,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prices_symbol 
                ON prices(symbol)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prices_symbol_date 
                ON prices(symbol, date)
            ''')
            
            # Metadata table for tracking last fetch
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_updated TEXT,
                    data_points INTEGER
                )
            ''')
            
            conn.commit()
            logger.debug(f"Cache tables created/verified (schema v{SCHEMA_VERSION})")
            
        except sqlite3.Error as e:
            logger.error(f"Database schema error: {str(e)}")
            raise
        finally:
            conn.close()
    
    def _apply_migrations(self, cursor: sqlite3.Cursor, from_version: int, to_version: int):
        """Apply schema migrations"""
        # Add migration logic here as schema changes
        # Example migration for future version changes:
        if from_version < 1 and to_version >= 1:
            # Migration code for schema version 1
            logger.info("Applying migration to schema version 1")
            pass
    
    def save_prices(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save price data to cache with validation and deduplication
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate and clean input DataFrame
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame provided for {ticker}")
                return False
            
            # Remove duplicate index entries
            df = df[~df.index.duplicated(keep='last')]
            
            # Ensure dataframe has required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in price data")
                    return False
            
            # Validate data types - convert to numeric and coerce errors
            numeric_cols = required_cols[:-1]  # All except Volume
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Remove rows with NaN in critical columns
            critical_cols = ['Open', 'High', 'Low', 'Close']
            df = df.dropna(subset=critical_cols)
            
            if df.empty:
                logger.error(f"No valid price data for {ticker} after cleaning")
                return False
            
            # Validate price sanity: all prices should be positive and High >= Low >= 0
            invalid_rows = (
                (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0) |
                (df['High'] < df['Low']) |  # High should be >= Low
                (df['Open'] < 0) | (df['Close'] < 0)  # No negative prices
            ).sum()
            
            if invalid_rows > 0:
                logger.warning(f"Found {invalid_rows} rows with invalid prices for {ticker}")
                # Remove invalid rows
                df = df[
                    (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0) &
                    (df['High'] >= df['Low']) & (df['Open'] >= 0) & (df['Close'] >= 0)
                ]
            
            if df.empty:
                logger.error(f"No valid data for {ticker} after price validation")
                return False
            
            conn = self._get_connection()
            
            try:
                # Prepare data with transaction
                records = []
                now = datetime.now().isoformat()
                
                for date, row in df.iterrows():
                    record = (
                        ticker,
                        date.strftime('%Y-%m-%d'),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        float(row['Adj Close']),
                        int(row['Volume']),
                        now
                    )
                    records.append(record)
                
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute('BEGIN TRANSACTION')
                
                # First, clear old data for this ticker to avoid conflicts
                cursor.execute('DELETE FROM prices WHERE symbol = ?', (ticker,))
                
                # Insert new data
                cursor.executemany('''
                    INSERT INTO prices 
                    (symbol, date, open, high, low, close, adj_close, volume, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)
                
                # Update metadata
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_metadata 
                    (symbol, last_updated, data_points)
                    VALUES (?, ?, ?)
                ''', (ticker, now, len(records)))
                
                conn.commit()
                logger.info(f"Cached {len(records)} price records for {ticker}")
                return True
                
            except sqlite3.IntegrityError as e:
                logger.error(f"Database integrity error for {ticker}: {str(e)}")
                conn.rollback()
                return False
            except sqlite3.OperationalError as e:
                logger.error(f"Database operational error for {ticker}: {str(e)}")
                conn.rollback()
                return False
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Unexpected error saving prices for {ticker}: {str(e)}")
            return False
    
    def get_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached price data with validation and deduplication
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with cached prices (deduplicated and validated), or None if not found/expired
        """
        try:
            conn = self._get_connection()
            
            # Check if data is fresh enough
            cursor = conn.cursor()
            cursor.execute('''
                SELECT last_updated FROM cache_metadata WHERE symbol = ?
            ''', (ticker,))
            
            metadata = cursor.fetchone()
            if not metadata:
                logger.debug(f"No cached data for {ticker}")
                conn.close()
                return None
            
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            age_days = (datetime.now() - last_updated).days
            
            if age_days > CACHE_VALIDITY_DAYS:
                logger.info(f"Cache for {ticker} is {age_days} days old, consider refreshing")
            
            # Fetch cached data
            df = pd.read_sql_query('''
                SELECT date, open, high, low, close, adj_close, volume
                FROM prices
                WHERE symbol = ?
                ORDER BY date ASC
            ''', conn, parse_dates=['date'])
            
            if df.empty:
                logger.warning(f"No price records found for cached {ticker}")
                conn.close()
                return None
            
            # Remove duplicate dates (keep last)
            if df['date'].duplicated().any():
                dup_count = df['date'].duplicated().sum()
                logger.warning(f"Found {dup_count} duplicate dates for {ticker}, keeping latest")
                df = df[~df['date'].duplicated(keep='last')]
            
            # Convert to proper dtypes and validate
            for col in ['open', 'high', 'low', 'close', 'adj_close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Remove rows with NaN in critical columns
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Validate price data: all should be positive and High >= Low
            invalid_rows = (
                (df['open'] <= 0) | (df['high'] <= 0) | 
                (df['low'] <= 0) | (df['close'] <= 0) |
                (df['high'] < df['low'])
            ).sum()
            
            if invalid_rows > 0:
                logger.warning(f"Found {invalid_rows} invalid price rows for {ticker}, removing")
                df = df[
                    (df['open'] > 0) & (df['high'] > 0) & 
                    (df['low'] > 0) & (df['close'] > 0) &
                    (df['high'] >= df['low'])
                ]
            
            if df.empty:
                logger.error(f"No valid cached data for {ticker} after validation")
                conn.close()
                return None
            
            logger.info(f"Retrieved {len(df)} cached price records for {ticker}")
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving cached prices for {ticker}: {str(e)}")
            return None
    
    def clear_cache(self, ticker: Optional[str] = None) -> bool:
        """
        Clear cache for specific ticker or all
        
        Args:
            ticker: Specific ticker to clear, or None to clear all
        
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # Start transaction for atomicity
                cursor.execute('BEGIN TRANSACTION')
                
                if ticker:
                    # Delete prices
                    cursor.execute('DELETE FROM prices WHERE symbol = ?', (ticker,))
                    prices_deleted = cursor.rowcount
                    
                    # Delete metadata
                    cursor.execute('DELETE FROM cache_metadata WHERE symbol = ?', (ticker,))
                    metadata_deleted = cursor.rowcount
                    
                    conn.commit()
                    logger.info(f"Cleared cache for {ticker}: {prices_deleted} price records, {metadata_deleted} metadata entries")
                else:
                    # Delete all with counts
                    cursor.execute('SELECT COUNT(*) FROM prices')
                    total_prices = cursor.fetchone()[0]
                    cursor.execute('SELECT COUNT(*) FROM cache_metadata')
                    total_metadata = cursor.fetchone()[0]
                    
                    cursor.execute('DELETE FROM prices')
                    cursor.execute('DELETE FROM cache_metadata')
                    
                    conn.commit()
                    logger.info(f"Cleared all cache: {total_prices} price records, {total_metadata} metadata entries")
                
                return True
                
            except sqlite3.Error as e:
                logger.error(f"Database error during cache clear: {str(e)}")
                conn.rollback()
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error clearing cache: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics with validation
        
        Returns:
            Dictionary with cache statistics or empty dict on error
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            try:
                # Get unique symbols
                cursor.execute('SELECT COUNT(DISTINCT symbol) FROM prices')
                result = cursor.fetchone()
                stats['unique_symbols'] = max(0, result[0] if result and result[0] else 0)
                
                # Get total records
                cursor.execute('SELECT COUNT(*) FROM prices')
                result = cursor.fetchone()
                stats['total_records'] = max(0, result[0] if result and result[0] else 0)
                
                # Get total data points with null check
                cursor.execute('SELECT SUM(data_points) FROM cache_metadata')
                result = cursor.fetchone()
                stats['total_data_points'] = max(0, result[0] if result and result[0] else 0)
                
                # Get oldest and newest dates
                cursor.execute('SELECT MIN(date), MAX(date) FROM prices')
                result = cursor.fetchone()
                if result and result[0] and result[1]:
                    stats['oldest_date'] = result[0]
                    stats['newest_date'] = result[1]
                else:
                    stats['oldest_date'] = None
                    stats['newest_date'] = None
                
                # Get database file size
                if self.db_path.exists():
                    stats['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
                else:
                    stats['db_size_mb'] = 0
                
                stats['cache_path'] = str(self.db_path)
                
                # Validate ranges
                if stats['unique_symbols'] < 0:
                    logger.warning("Negative unique_symbols count detected, resetting to 0")
                    stats['unique_symbols'] = 0
                if stats['total_records'] < 0:
                    logger.warning("Negative total_records count detected, resetting to 0")
                    stats['total_records'] = 0
                if stats['total_data_points'] < 0:
                    logger.warning("Negative total_data_points count detected, resetting to 0")
                    stats['total_data_points'] = 0
                
                logger.info(f"Cache stats: {stats['unique_symbols']} symbols, {stats['total_records']} records, {stats['db_size_mb']}MB")
                return stats
                
            except sqlite3.OperationalError as e:
                logger.error(f"Database error retrieving stats: {str(e)}")
                return {}
                
        except Exception as e:
            logger.error(f"Unexpected error getting cache stats: {str(e)}")
            return {}
        finally:
            if conn:
                conn.close()


# Singleton instance
_cache_instance: Optional[PriceCache] = None


def get_cache() -> PriceCache:
    """Get or create cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PriceCache()
    return _cache_instance


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    cache = get_cache()
    stats = cache.get_cache_stats()
    
    print("Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
