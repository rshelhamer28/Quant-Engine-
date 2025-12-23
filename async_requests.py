# -*- coding: utf-8 -*-
"""
Async HTTP utilities for parallel API requests
Enables concurrent data fetching for improved performance on slow networks
"""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


async def fetch_url(session: aiohttp.ClientSession, url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch a single URL asynchronously with error handling
    
    Args:
        session: aiohttp ClientSession
        url: URL to fetch
        params: Query parameters
        timeout: Request timeout in seconds
    
    Returns:
        JSON response or None if error
    """
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                logger.debug(f"Successfully fetched {url}")
                return await response.json()
            else:
                logger.warning(f"HTTP {response.status} from {url}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching {url}")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {str(e)}")
        return None


async def fetch_multiple_urls(urls: List[str], params_list: Optional[List[Dict[str, Any]]] = None, timeout: int = 10) -> List[Optional[Dict[str, Any]]]:
    """
    Fetch multiple URLs concurrently
    
    Args:
        urls: List of URLs to fetch
        params_list: Optional list of parameter dicts (same length as urls)
        timeout: Request timeout in seconds
    
    Returns:
        List of JSON responses (None for failed requests)
    
    Example:
        results = asyncio.run(fetch_multiple_urls([
            'https://api1.com/data',
            'https://api2.com/data'
        ]))
    """
    if params_list is None:
        params_list = [None] * len(urls)
    
    if len(urls) != len(params_list):
        raise ValueError("urls and params_list must have same length")
    
    logger.info(f"Fetching {len(urls)} URLs concurrently")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_url(session, url, params, timeout)
            for url, params in zip(urls, params_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        clean_results = [r if not isinstance(r, Exception) else None for r in results]
        logger.info(f"Completed {len(clean_results)} requests ({sum(1 for r in clean_results if r is not None)} successful)")
        
        return clean_results


# Example usage for Finnhub multiple symbol news fetching
async def fetch_multi_symbol_news(symbols: List[str], api_key: str, days: int = 30) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch news for multiple symbols in parallel
    
    Args:
        symbols: List of ticker symbols
        api_key: Finnhub API key
        days: Number of days of history
    
    Returns:
        Dict mapping symbols to their news data
    
    Example:
        news_data = asyncio.run(fetch_multi_symbol_news(['AAPL', 'MSFT'], api_key='xxx'))
    """
    from datetime import datetime, timedelta
    
    base_url = "https://finnhub.io/api/v1/company-news"
    
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    urls = [base_url] * len(symbols)
    params_list = [
        {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': api_key
        }
        for symbol in symbols
    ]
    
    logger.info(f"Fetching news for {len(symbols)} symbols asynchronously")
    results = await fetch_multiple_urls(urls, params_list, timeout=15)
    
    return {
        symbol: result
        for symbol, result in zip(symbols, results)
    }


if __name__ == '__main__':
    # Example: Fetch news for multiple stocks
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    api_key = 'd51nuu1r01qiituq1060d51nuu1r01qiituq106g'  # Demo API key (rate-limited)
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Run async function
    results = asyncio.run(fetch_multi_symbol_news(symbols, api_key))
    
    for symbol, data in results.items():
        if data:
            print(f"{symbol}: {len(data) if isinstance(data, list) else 'N/A'} articles")
        else:
            print(f"{symbol}: Failed to fetch")
