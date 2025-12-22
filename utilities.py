# utilities.py - Refactored Utility Functions
# Phase 1: Critical Utilities for Immediate Implementation
# Reduces code duplication from 40+ repetitions

import numpy as np
import pandas as pd
from typing import Optional, Union, Callable

# ===== CONFIGURATION =====
TRADING_DAYS_PER_YEAR = 252

# ===== ANNUALIZED VOLATILITY UTILITIES =====

def annualize_volatility(returns_series: Union[pd.Series, float], min_length: int = 0) -> float:
    """Convert daily returns volatility to annual volatility.
    
    Replaces 7+ instances of: .std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    Args:
        returns_series: Series/array of daily returns OR a pre-calculated std value
        min_length: Minimum data points required (return 0 if below)
    
    Returns:
        Annualized volatility (or 0.0 if insufficient data or NaN)
    
    Example:
        # Old: stock_annual_vol = stock_returns_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        # New: stock_annual_vol = annualize_volatility(stock_returns_clean)
        
        # Old: hv_30d = stock_returns_clean.tail(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        # New: hv_30d = annualize_volatility(stock_returns_clean.tail(30))
    """
    if returns_series is None:
        return 0.0
    
    # Handle both Series and scalar (rolling.std() returns scalar)
    if isinstance(returns_series, (int, float)):
        std_dev = returns_series
    else:
        if len(returns_series) <= min_length:
            return 0.0
        std_dev = returns_series.std()
    
    # Check for invalid values
    if pd.isna(std_dev) or std_dev == 0:
        return 0.0
    
    return float(std_dev * np.sqrt(TRADING_DAYS_PER_YEAR))


# ===== SAFE STATISTICS UTILITIES =====

def safe_std(series: Optional[pd.Series], min_length: int = 1, annualize: bool = False, default: float = 0.0) -> float:
    """Calculate standard deviation safely.
    
    Replaces 6+ instances of: .std() * sqrt() if len() > threshold else 0
    
    Args:
        series: Data series
        min_length: Minimum required data points
        annualize: If True, annualize the volatility
        default: Default value if insufficient data or error
    
    Returns:
        Standard deviation (optionally annualized) or default
    
    Example:
        # Old: downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 1 else 0
        # New: downside_vol = safe_std(downside_returns, min_length=1, annualize=True)
    """
    if series is None or len(series) <= min_length:
        return default
    
    try:
        std_val = series.std()
        if pd.isna(std_val) or std_val == 0:
            return default
        
        if annualize:
            std_val *= np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return float(std_val)
    except Exception:
        return default


def safe_var(series: Optional[pd.Series], min_length: int = 1, default: float = 0.0) -> float:
    """Calculate variance safely.
    
    Replaces instances of: .var() if len() > threshold else 0
    
    Args:
        series: Data series
        min_length: Minimum required data points
        default: Default value if insufficient data
    
    Returns:
        Variance or default
    
    Example:
        # Old: market_variance = market_returns_clean.var() if len(market_returns_clean) > 1 else 0
        # New: market_variance = safe_var(market_returns_clean, min_length=1)
    """
    if series is None or len(series) <= min_length:
        return default
    
    try:
        var_val = series.var()
        return default if pd.isna(var_val) else float(var_val)
    except Exception:
        return default


def safe_mean(series: Optional[pd.Series], min_length: int = 1, default: float = 0.0) -> float:
    """Calculate mean safely.
    
    Args:
        series: Data series
        min_length: Minimum required data points
        default: Default value if insufficient data
    
    Returns:
        Mean or default
    
    Example:
        # Old: jump_mean = jumps.mean() if len(jumps) > 0 else 0
        # New: jump_mean = safe_mean(jumps)
    """
    if series is None or len(series) <= min_length:
        return default
    
    try:
        mean_val = series.mean()
        return default if pd.isna(mean_val) else float(mean_val)
    except Exception:
        return default


# ===== DATA CLEANING UTILITIES =====

def clean_returns(returns_series: pd.Series) -> pd.Series:
    """Remove infinite and NaN values from returns series.
    
    Replaces instances of: .replace([np.inf, -np.inf], np.nan).dropna()
    
    Args:
        returns_series: Series of returns
    
    Returns:
        Cleaned series with inf and NaN removed
    
    Example:
        # Old: stock_returns = stock_returns.replace([np.inf, -np.inf], np.nan).dropna()
        # New: stock_returns = clean_returns(stock_returns)
    """
    if returns_series is None:
        return pd.Series()
    
    return returns_series.replace([np.inf, -np.inf], np.nan).dropna()


def fill_missing_values(df: pd.DataFrame, forward_fill: bool = True, backward_fill: bool = True) -> pd.DataFrame:
    """Fill missing values (NaN) in dataframe using forward and backward fill.
    
    Replaces instances of: .ffill().bfill()
    
    Args:
        df: DataFrame with potential missing values
        forward_fill: Whether to forward fill
        backward_fill: Whether to backward fill
    
    Returns:
        DataFrame with filled values
    
    Example:
        # Old: aligned_df = aligned_df.ffill().bfill().dropna()
        # New: aligned_df = fill_missing_values(aligned_df).dropna()
    """
    if df is None or df.empty:
        return df
    
    result = df.copy()
    if forward_fill:
        result = result.ffill()
    if backward_fill:
        result = result.bfill()
    
    return result


# ===== CONDITIONAL CALCULATION UTILITIES =====

def safe_calculate(calculation_func: Callable, *args, min_length: int = 0, default: float = 0.0, **kwargs) -> float:
    """Execute a calculation safely with length validation.
    
    Args:
        calculation_func: Function to execute
        *args: Positional arguments to pass to function
        min_length: Minimum length to require on first Series arg
        default: Default if length insufficient or exception
        **kwargs: Keyword arguments to pass to function
    
    Returns:
        Result of calculation or default
    """
    try:
        # Check first Series argument length if specified
        if min_length > 0 and len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, (pd.Series, list)):
                if len(first_arg) <= min_length:
                    return default
        
        result = calculation_func(*args, **kwargs)
        
        if pd.isna(result) or (isinstance(result, float) and np.isinf(result)):
            return default
        
        return float(result)
    except Exception:
        return default


# ===== DATA VALIDATION UTILITIES =====

class DataValidation:
    """Constants and utilities for consistent data validation.
    
    Replaces 12+ inconsistent threshold checks (2, 15, 20, 30, 100 with no semantic meaning)
    """
    
    # Semantic constants for data validation thresholds
    MIN_RETURNS_FOR_ROLLING = 2              # Absolute minimum for any rolling calculation
    MIN_RETURNS_FOR_VOLATILITY = 20          # Minimum for reasonable volatility estimate
    MIN_RETURNS_FOR_ESTIMATION = 20          # Minimum for parameter estimation
    MIN_RETURNS_FOR_BETA = 252               # One year of trading days (standard)
    MIN_PRICE_DATA = 100                     # Minimum price points for Monte Carlo
    MIN_ALIGNED_DATA = 20                    # Minimum aligned data points
    
    @staticmethod
    def has_sufficient_data(series: Optional[pd.Series], min_length: int) -> bool:
        """Check if series has minimum length without raising.
        
        Args:
            series: Data series to check
            min_length: Minimum required length
        
        Returns:
            True if sufficient data, False otherwise
        
        Example:
            if DataValidation.has_sufficient_data(returns, DataValidation.MIN_RETURNS_FOR_VOLATILITY):
                # Calculate volatility
        """
        return series is not None and len(series) >= min_length
    
    @staticmethod
    def validate_series_length(series: Optional[pd.Series], min_length: int, context: str = "") -> Optional[pd.Series]:
        """Validate series has minimum length. Raise if not.
        
        Args:
            series: Data series to validate
            min_length: Minimum required length
            context: Context string for error message
        
        Returns:
            Series if valid
        
        Raises:
            ValueError if insufficient data
        
        Example:
            DataValidation.validate_series_length(
                stock_returns,
                DataValidation.MIN_RETURNS_FOR_BETA,
                "calculate_advanced_metrics"
            )
        """
        if series is None:
            raise ValueError(f"{context}: Series is None")
        if len(series) < min_length:
            raise ValueError(
                f"{context}: Insufficient data ({len(series)} < {min_length} required)"
            )
        return series


# ===== RISK METRICS UTILITIES =====

class RiskMetrics:
    """Calculate standardized risk metrics and percentiles."""
    
    # Confidence interval percentiles
    CI_95 = {'lower': 2.5, 'upper': 97.5}
    CI_99 = {'lower': 0.5, 'upper': 99.5}
    
    @staticmethod
    def calculate_confidence_interval(
        data: np.ndarray,
        confidence: str = '95'
    ) -> dict:
        """Calculate confidence interval for price distribution.
        
        Args:
            data: Array of prices/returns
            confidence: '95' or '99' for confidence level
        
        Returns:
            Dictionary with 'lower' and 'upper' bounds
        
        Example:
            ci = RiskMetrics.calculate_confidence_interval(final_prices)
            # {'lower': 95.50, 'upper': 104.60}
        """
        conf = RiskMetrics.CI_95 if confidence == '95' else RiskMetrics.CI_99
        return {
            'lower': float(np.percentile(data, conf['lower'])),
            'upper': float(np.percentile(data, conf['upper']))
        }
    
    @staticmethod
    def calculate_var_cvar(
        data: np.ndarray,
        percentile: float = 5.0
    ) -> tuple:
        """Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            data: Array of prices/returns
            percentile: Percentile for VaR (default 5 = 95% confidence)
        
        Returns:
            Tuple of (VaR, CVaR)
        
        Example:
            var_95, cvar_95 = RiskMetrics.calculate_var_cvar(final_prices, percentile=5)
        """
        var = np.percentile(data, percentile)
        cvar = np.mean(data[data <= var]) if len(data[data <= var]) > 0 else var
        return float(var), float(cvar)
    
    @staticmethod
    def calculate_monte_carlo_risks(
        final_prices: np.ndarray,
        last_price: float,
        confidence: str = '95'
    ) -> dict:
        """Calculate all risk metrics from Monte Carlo results.
        
        Replaces the 8-line calculation at lines 977-984
        
        Args:
            final_prices: Array of final prices from simulation
            last_price: Current/initial price
            confidence: '95' or '99' confidence level
        
        Returns:
            Dictionary with all risk metrics
        
        Example:
            risk = RiskMetrics.calculate_monte_carlo_risks(final_prices, current_price)
            # Access: risk['ci_lower'], risk['ci_upper'], risk['var_95'], etc.
        """
        ci = RiskMetrics.calculate_confidence_interval(final_prices, confidence)
        var_95, cvar_95 = RiskMetrics.calculate_var_cvar(final_prices, percentile=5)
        percentile_99 = float(np.percentile(final_prices, 99))
        prob_loss = float(np.mean(final_prices < last_price))
        prob_positive = float(np.mean(final_prices > last_price))
        
        return {
            'ci_lower': ci['lower'],
            'ci_upper': ci['upper'],
            'var_95': var_95,
            'cvar_95': cvar_95,
            'percentile_99': percentile_99,
            'prob_loss': prob_loss,
            'prob_positive_return': prob_positive
        }


# ===== ERROR HANDLING UTILITIES =====

def handle_errors(context_name: str, default_return=None, ticker_param: str = None):
    """Decorator for automatic safe error handling.
    
    Replaces try/except blocks in 5+ locations
    
    Args:
        context_name: Name of operation for logging
        default_return: Value to return on error
        ticker_param: Name of parameter containing ticker (e.g., 'ticker')
    
    Returns:
        Decorator function
    
    Example:
        @handle_errors("get_price_data", default_return=(None, None), ticker_param="ticker")
        def get_price_data(ticker: str):
            # Implementation here
            pass
        
        # Now exceptions are automatically logged and (None, None) is returned
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract ticker if specified
                ticker = None
                if ticker_param:
                    if ticker_param in kwargs:
                        ticker = kwargs[ticker_param]
                    # Could also extract from args if needed
                
                # Import here to avoid circular imports
                from quant_engine import SafeErrorHandler
                SafeErrorHandler.log_exception(e, context_name, ticker or "")
                
                return default_return
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# ===== TEST UTILITIES (Optional) =====

class StatisticsValidator:
    """Validate statistics calculations for correctness."""
    
    @staticmethod
    def validate_volatility(annual_vol: float) -> bool:
        """Check if volatility is reasonable (0-300% annually)."""
        return 0 <= annual_vol <= 3.0
    
    @staticmethod
    def validate_returns(returns: float) -> bool:
        """Check if returns are reasonable (-100% to +500%)."""
        return -1.0 <= returns <= 5.0
    
    @staticmethod
    def validate_correlation(correlation: float) -> bool:
        """Check if correlation is valid (-1 to 1)."""
        return -1.0 <= correlation <= 1.0


# ===== USAGE EXAMPLES =====

"""
# In quant_engine.py, instead of:

    stock_annual_vol = stock_returns_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    hv_30d = stock_returns_clean.tail(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    hv_90d = stock_returns_clean.tail(90).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    upside_vol = upside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(upside_returns) > 1 else 0
    downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 1 else 0

# Use:

    stock_annual_vol = annualize_volatility(stock_returns_clean)
    hv_30d = annualize_volatility(stock_returns_clean.tail(30))
    hv_90d = annualize_volatility(stock_returns_clean.tail(90))
    upside_vol = safe_std(upside_returns, min_length=1, annualize=True)
    downside_vol = safe_std(downside_returns, min_length=1, annualize=True)

# Instead of:

    stock_returns = stock_returns.replace([np.inf, -np.inf], np.nan).dropna()
    market_returns = market_returns.replace([np.inf, -np.inf], np.nan).dropna()

# Use:

    stock_returns = clean_returns(stock_returns)
    market_returns = clean_returns(market_returns)

# Instead of:

    if len(returns) < 100:
        return ...
    if len(aligned_df) < 20:
        return ...
    if len(stock_returns) < 252:
        return ...

# Use:

    DataValidation.validate_series_length(returns, DataValidation.MIN_PRICE_DATA)
    DataValidation.validate_series_length(aligned_df, DataValidation.MIN_ALIGNED_DATA)
    DataValidation.validate_series_length(stock_returns, DataValidation.MIN_RETURNS_FOR_BETA)

# Instead of (lines 977-984):

    ci_95_lower = np.percentile(final_prices, 2.5)
    ci_95_upper = np.percentile(final_prices, 97.5)
    var_95 = np.percentile(final_prices, 5)
    cvar_95 = np.mean(final_prices[final_prices <= var_95])
    percentile_99 = np.percentile(final_prices, 99)
    prob_loss = np.mean(final_prices < last_price)

# Use:

    risks = RiskMetrics.calculate_monte_carlo_risks(final_prices, last_price)
    # Access: risks['ci_lower'], risks['ci_upper'], risks['var_95'], etc.
"""
