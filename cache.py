import os
import pandas as pd
import logging
from datetime import datetime, date
import pytz
from pathlib import Path
import functools
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Path to cache files
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
STOCK_CACHE_FILE = CACHE_DIR / "stock_info.parquet"
FUND_CACHE_FILE = CACHE_DIR / "fund_info.parquet"

# Global cache variables
_STOCK_CACHE = None
_FUND_CACHE = None

def load_all_stock_cache():
    """Load the entire stock cache into memory at once."""
    global _STOCK_CACHE
    
    if not STOCK_CACHE_FILE.exists():
        logging.info("Stock cache file does not exist yet")
        _STOCK_CACHE = pd.DataFrame()
        return _STOCK_CACHE
    
    try:
        _STOCK_CACHE = pd.read_parquet(STOCK_CACHE_FILE)
        logging.info(f"Loaded {len(_STOCK_CACHE)} stocks from cache")
        return _STOCK_CACHE
    except Exception as e:
        logging.warning(f"Error reading stock cache: {e}")
        _STOCK_CACHE = pd.DataFrame()
        return _STOCK_CACHE

def load_all_fund_cache():
    """Load the entire fund cache into memory at once."""
    global _FUND_CACHE
    
    if not FUND_CACHE_FILE.exists():
        logging.info("Fund cache file does not exist yet")
        _FUND_CACHE = pd.DataFrame()
        return _FUND_CACHE
    
    try:
        _FUND_CACHE = pd.read_parquet(FUND_CACHE_FILE)
        logging.info(f"Loaded {len(_FUND_CACHE)} funds from cache")
        return _FUND_CACHE
    except Exception as e:
        logging.warning(f"Error reading fund cache: {e}")
        _FUND_CACHE = pd.DataFrame()
        return _FUND_CACHE

def get_cached_stock_info_fast(symbol: str) -> dict:
    """Get cached stock info using the preloaded cache."""
    global _STOCK_CACHE
    
    if _STOCK_CACHE is None:
        _STOCK_CACHE = load_all_stock_cache()
    
    if symbol not in _STOCK_CACHE.index:
        return None
    
    row = _STOCK_CACHE.loc[symbol]
    if is_stock_info_fresh(row):
        return row.to_dict()
    return None

def get_fund_expense_ratio_fast(ticker: str) -> float:
    """Get fund expense ratio using the preloaded cache."""
    global _FUND_CACHE
    
    if _FUND_CACHE is None:
        _FUND_CACHE = load_all_fund_cache()
    
    if ticker in _FUND_CACHE.index:
        row = _FUND_CACHE.loc[ticker]
        if 'expense_ratio' in row and row['expense_ratio'] is not None:
            return row['expense_ratio']
    
    # Fall back to reasonable defaults
    if ticker.startswith('VO') or ticker.startswith('VT') or ticker.startswith('SP'):
        return 0.002  # 0.2% for passive funds
    else:
        return 0.01   # 1% for active funds

def concat_dataframes_safely(dfs):
    """
    Concatenate a list of DataFrames, safely handling empty or all-NA DataFrames to avoid FutureWarning.
    """
    # Filter out empty or all-NA DataFrames
    filtered_dfs = [df for df in dfs if not df.empty and not df.isna().all().all()]

    if not filtered_dfs:
        return pd.DataFrame()
    elif len(filtered_dfs) == 1:
        return filtered_dfs[0].copy()
    else:
        return pd.concat(filtered_dfs)

def get_pytz_timezone(short_name: str) -> str:
    """
    Convert exchange timezone short name to pytz timezone name.
    
    Args:
        short_name: Short timezone name (e.g., 'EDT', 'EST')
        
    Returns:
        Pytz timezone name (e.g., 'America/New_York')
    """
    # Mapping of common exchange timezone short names to pytz timezone names
    timezone_map = {
        'EDT': 'America/New_York',
        'EST': 'America/New_York',
        'CST': 'America/Chicago',
        'CDT': 'America/Chicago',
        'MST': 'America/Denver',
        'MDT': 'America/Denver',
        'PST': 'America/Los_Angeles',
        'PDT': 'America/Los_Angeles',
        'GMT': 'Europe/London',
        'BST': 'Europe/London',
        'CET': 'Europe/Paris',
        'CEST': 'Europe/Paris',
        'JST': 'Asia/Tokyo',
        'AEST': 'Australia/Sydney',
        'AEDT': 'Australia/Sydney',
    }
    
    return timezone_map.get(short_name, 'UTC')

def is_stock_info_fresh(stock_info: pd.Series) -> bool:
    """
    Check if stock info is from today.
    
    Args:
        stock_info: Series with stock info
        
    Returns:
        True if the data is from today, False otherwise
    """
    try:
        # Get the market time from the stock info
        market_time = stock_info.get('regularMarketTime')
        
        if not market_time:
            logging.info("No regularMarketTime in stock info")
            return False
        
        # Convert the timestamp to a datetime
        market_datetime = datetime.fromtimestamp(market_time)
        
        # Get the timezone from the stock info
        timezone_name = stock_info.get('exchangeTimezoneShortName')
        if timezone_name:
            try:
                # Convert to the exchange timezone
                timezone = pytz.timezone(get_pytz_timezone(timezone_name))
                market_datetime = market_datetime.astimezone(timezone)
            except Exception as e:
                logging.warning(f"Error converting timezone: {e}")
        
        # Check if the date is today
        today = date.today()
        return market_datetime.date() == today
        
    except Exception as e:
        logging.warning(f"Error checking if stock info is fresh: {e}")
        return False
    
def is_fund_info_fresh(fund_info: pd.Series) -> bool:
    """
    Check if fund info is from today or recent enough (within 7 days).
    
    Args:
        fund_info: Series with fund info
        
    Returns:
        True if the data is recent, False otherwise
    """
    try:
        # Get the cache date from the fund info
        cache_date = fund_info.get('cache_date')
        
        if not cache_date:
            logging.info("No cache_date in fund info")
            return False
        
        # Convert the string to a date
        cache_datetime = datetime.strptime(cache_date, '%Y-%m-%d').date()
        
        # Check if the date is within the last 7 days (fund data doesn't change as often)
        today = date.today()
        delta = today - cache_datetime
        return delta.days <= 7
        
    except Exception as e:
        logging.warning(f"Error checking if fund info is fresh: {e}")
        return False
    
# Stock cache functions
def get_cached_stock_info(symbol: str) -> dict:
    """
    Get cached stock info for a symbol if it exists and is fresh (from today).
    
    Args:
        symbol: Stock symbol to look up
        
    Returns:
        Dictionary with stock info or None if not cached or stale
    """
    if not STOCK_CACHE_FILE.exists():
        logging.info("Stock cache file does not exist yet")
        return None
    
    try:
        # Load the cache
        cache_df = pd.read_parquet(STOCK_CACHE_FILE)
        
        # Check if symbol exists in cache
        if symbol not in cache_df.index:
            logging.info(f"Symbol {symbol} not found in cache")
            return None
        
        # Get the row for this symbol
        row = cache_df.loc[symbol]
        
        # Check if the data is fresh
        if is_stock_info_fresh(row):
            logging.info(f"Using cached data for {symbol} from today")
            # Convert the row to a dictionary
            return row.to_dict()
        else:
            logging.info(f"Cached data for {symbol} is stale")
            return None
            
    except Exception as e:
        logging.warning(f"Error reading stock cache: {e}")
        return None

def update_stock_info_cache(symbol: str, stock_info: dict) -> None:
    """
    Update the stock info cache with new data.
    
    Args:
        symbol: Stock symbol
        stock_info: Dictionary with stock info from yfinance
    """
    try:
        # Create a DataFrame from the stock info
        new_row = pd.DataFrame([stock_info], index=[symbol])
        
        if STOCK_CACHE_FILE.exists():
            # Load existing cache
            cache_df = pd.read_parquet(STOCK_CACHE_FILE)
            
            # Update or add the row for this symbol
            cache_df = cache_df.drop(symbol, errors='ignore')
            # Use safe concat instead of direct concatenation
            cache_df = concat_dataframes_safely([cache_df, new_row])
        else:
            # Create new cache
            cache_df = new_row
        
        # Save the updated cache
        cache_df.to_parquet(STOCK_CACHE_FILE)
        logging.info(f"Updated stock cache for {symbol}")
        
    except Exception as e:
        logging.warning(f"Error updating stock cache: {e}")

# Fund cache functions
def get_cached_fund_info(ticker: str) -> dict:
    """
    Get cached fund info for a ticker if it exists and is recent.
    
    Args:
        ticker: Fund ticker to look up
        
    Returns:
        Dictionary with fund info or None if not cached or stale
    """
    if not FUND_CACHE_FILE.exists():
        logging.info("Fund cache file does not exist yet")
        return None
    
    try:
        # Load the cache
        cache_df = pd.read_parquet(FUND_CACHE_FILE)
        
        # Check if ticker exists in cache
        if ticker not in cache_df.index:
            logging.info(f"Fund ticker {ticker} not found in cache")
            return None
        
        # Get the row for this ticker
        row = cache_df.loc[ticker]
        
        # Check if the data is fresh
        if is_fund_info_fresh(row):
            logging.info(f"Using cached fund data for {ticker}")
            # Convert the row to a dictionary
            return row.to_dict()
        else:
            logging.info(f"Cached fund data for {ticker} is stale")
            return None
            
    except Exception as e:
        logging.warning(f"Error reading fund cache: {e}")
        return None

def update_fund_info_cache(ticker: str, fund_info: dict) -> None:
    """
    Update the fund info cache with new data.
    
    Args:
        ticker: Fund ticker
        fund_info: Dictionary with fund info
    """
    try:
        # Add cache date if not present
        if 'cache_date' not in fund_info:
            fund_info['cache_date'] = date.today().strftime('%Y-%m-%d')
            
        # Create a DataFrame from the fund info
        new_row = pd.DataFrame([fund_info], index=[ticker])
        
        if FUND_CACHE_FILE.exists():
            # Load existing cache
            cache_df = pd.read_parquet(FUND_CACHE_FILE)
            
            # Update or add the row for this ticker
            cache_df = cache_df.drop(ticker, errors='ignore')
            # Use safe concat instead of direct concatenation
            cache_df = concat_dataframes_safely([cache_df, new_row])
        else:
            # Create new cache
            cache_df = new_row
        
        # Save the updated cache
        cache_df.to_parquet(FUND_CACHE_FILE)
        logging.info(f"Updated fund cache for {ticker}")
        
    except Exception as e:
        logging.warning(f"Error updating fund cache: {e}")

def get_fund_expense_ratio(ticker: str, update_cache: bool = True) -> float:
    """
    Get the expense ratio for a fund. First tries to fetch from cache,
    then falls back to yfinance, then to reasonable defaults.
    
    Args:
        ticker: Fund ticker symbol
        update_cache: Whether to update the cache with new data (default: True)
        
    Returns:
        Expense ratio as a decimal (e.g., 0.0075 for 0.75%)
    """
    try:
        # First check cache
        fund_info = get_cached_fund_info(ticker)
        if fund_info and 'expense_ratio' in fund_info and fund_info['expense_ratio'] is not None:
            return fund_info['expense_ratio']
        
        # If not in cache or update_cache is True, try to fetch from yfinance
        if update_cache:
            import yfinance as yf
            fund = yf.Ticker(ticker)
            expense_ratio = None
            
            if 'annualReportExpenseRatio' in fund.info and fund.info['annualReportExpenseRatio'] is not None:
                expense_ratio = fund.info['annualReportExpenseRatio']
            
            # If found, update cache and return
            if expense_ratio is not None:
                fund_info = {'expense_ratio': expense_ratio, 'cache_date': date.today().strftime('%Y-%m-%d')}
                update_fund_info_cache(ticker, fund_info)
                return expense_ratio
    
    except Exception as e:
        logging.warning(f"Could not get expense ratio for {ticker}: {e}")
    
    # Fall back to reasonable defaults based on fund type
    # Passive funds ~0.2%, active funds ~1%
    if ticker.startswith('VO') or ticker.startswith('VT') or ticker.startswith('SP'):
        # Likely a passive index fund
        default_ratio = 0.002  # 0.2%
    else:
        # Assume actively managed
        default_ratio = 0.01  # 1%
    
    # Store the default in cache if update_cache is True
    if update_cache:
        fund_info = {'expense_ratio': default_ratio, 'cache_date': date.today().strftime('%Y-%m-%d')}
        update_fund_info_cache(ticker, fund_info)
    
    return default_ratio

def clear_cache(cache_type: str = "all") -> None:
    """
    Clear the specified cache.
    
    Args:
        cache_type: Type of cache to clear ("stock", "fund", or "all")
    """
    if cache_type in ["stock", "all"]:
        if STOCK_CACHE_FILE.exists():
            STOCK_CACHE_FILE.unlink()
            logging.info("Stock cache cleared")
    
    if cache_type in ["fund", "all"]:
        if FUND_CACHE_FILE.exists():
            FUND_CACHE_FILE.unlink()
            logging.info("Fund cache cleared")

# Function decorator for caching
def cache_function(maxsize=128, ttl=3600):
    """
    Decorator to cache function results in memory with a time-to-live.
    
    Args:
        maxsize: Maximum cache size (default: 128)
        ttl: Time to live in seconds (default: 1 hour)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Use LRU cache from functools
        @functools.lru_cache(maxsize=maxsize)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs), time.time()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, timestamp = cached_func(*args, **kwargs)
            # Check if result is still valid
            if time.time() - timestamp > ttl:
                cached_func.cache_clear()
                result, timestamp = cached_func(*args, **kwargs)
            return result
        
        # Add cache_clear method
        wrapper.cache_clear = cached_func.cache_clear
        return wrapper
    
    return decorator