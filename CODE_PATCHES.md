# Code Patches: Multi-User Safety Fixes
## Copy-Paste Ready Solutions for Critical Issues

---

## Patch 1: Session State Initialization
**File:** `quant_engine.py`  
**Location:** After line 667 (`st.set_page_config()`)  
**Lines:** ~40 new lines

```python
# ===== SESSION STATE MANAGEMENT (CRITICAL FOR MULTI-USER) =====
def init_session_state():
    """
    Initialize per-user session state.
    Ensures each user has isolated data structures.
    Called once per session at app startup.
    """
    # Prevent re-initialization
    if 'session_initialized' in st.session_state:
        return
    
    import random
    
    # Generate unique session ID
    timestamp = datetime.now().isoformat()
    random_val = random.randint(100000, 999999)
    session_bytes = f"{timestamp}_{random_val}".encode()
    session_id = hashlib.md5(session_bytes).hexdigest()[:12]
    
    # Initialize per-user structures
    st.session_state.session_id = session_id
    st.session_state.rate_limiter = RequestRateLimiter(
        max_requests=50,
        window_seconds=60
    )
    st.session_state.cache = {}
    st.session_state.cache_timestamps = {}
    st.session_state.request_history = []
    st.session_state.session_start_time = datetime.now()
    st.session_state.current_ticker = None
    st.session_state.error_count = 0
    st.session_state.session_initialized = True
    
    logger.info(
        f"Session initialized - session_id={session_id}, "
        f"timestamp={datetime.now().isoformat()}"
    )


def get_cache_key(ticker: str, data_type: str = "price_data") -> str:
    """Generate cache key for per-user cache."""
    return f"{data_type}_{ticker}"


def is_cache_valid(cache_key: str, max_age_hours: float = 24) -> bool:
    """Check if cached data is still fresh."""
    if cache_key not in st.session_state.cache:
        return False
    if cache_key not in st.session_state.cache_timestamps:
        return False
    
    cached_time = st.session_state.cache_timestamps[cache_key]
    age_seconds = (datetime.now() - cached_time).total_seconds()
    age_hours = age_seconds / 3600
    
    return age_hours < max_age_hours


def log_request(
    ticker: str,
    action: str,
    success: bool,
    error_msg: str = None,
    request_id: str = None
) -> str:
    """Log request with session context."""
    if request_id is None:
        request_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{ticker}".encode()
        ).hexdigest()[:8]
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'request_id': request_id,
        'ticker': ticker,
        'action': action,
        'success': success,
        'error': error_msg
    }
    
    st.session_state.request_history.append(log_entry)
    
    log_msg = (
        f"[SESSION: {st.session_state.session_id}] "
        f"[REQ: {request_id}] {ticker} {action}"
    )
    if error_msg:
        log_msg += f" - ERROR: {error_msg}"
    
    logger.info(log_msg)
    return request_id


# ===== CALL THIS AT APP START (REQUIRED) =====
init_session_state()
```

**Where to insert:** Right after this line (around 668):
```python
st.set_page_config(
    page_title="Quantitative Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)
# ← INSERT PATCH HERE
```

---

## Patch 2: Update Rate Limiter in get_price_data()
**File:** `quant_engine.py`  
**Location:** Line ~1370 in `get_price_data()` function  
**Changes:** 1 line modified

### Find this:
```python
def get_price_data(ticker: str) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
    """
    Fetch and align price data with robust error handling and quality validation.
    Includes rate limiting and audit logging for access control.
    """
    # ===== INPUT VALIDATION =====
    try:
        InputValidator.validate_ticker(ticker)
    except AssertionError as e:
        logger.error(f"VALIDATION: Invalid ticker - {e}")
        return None, None
    
    # ===== SECURITY: Rate limiting to prevent abuse =====
    if not RATE_LIMITER.is_allowed(ticker):  # ← CHANGE THIS LINE
        logger.error(f"SECURITY: Rate limit exceeded for {ticker}. Request rejected.")
        return None, None
```

### Replace with:
```python
def get_price_data(ticker: str) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
    """
    Fetch and align price data with robust error handling and quality validation.
    Includes rate limiting and audit logging for access control.
    """
    # ===== INPUT VALIDATION =====
    try:
        InputValidator.validate_ticker(ticker)
    except AssertionError as e:
        logger.error(f"VALIDATION: Invalid ticker - {e}")
        return None, None
    
    # ===== SECURITY: Rate limiting to prevent abuse (PER-USER) =====
    if not st.session_state.rate_limiter.is_allowed(ticker):  # ← UPDATED
        log_request(ticker, 'fetch', False, 'rate_limit_exceeded')
        logger.error(f"SECURITY: Rate limit exceeded for {ticker}. Request rejected.")
        return None, None
```

---

## Patch 3: Add Per-User Caching Wrapper
**File:** `quant_engine.py`  
**Location:** Right after `get_price_data()` function (around line 1450)  
**Lines:** ~25 new lines

```python
def get_price_data_with_user_cache(
    ticker: str
) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
    """
    Fetch price data with per-user session caching.
    
    For each user session:
    1. Check if data in user's cache and fresh
    2. If yes: return cached data
    3. If no: fetch fresh data via get_price_data()
    4. Store result in user's cache
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        (hist_df, market_df) or (None, None) if error
    """
    cache_key = get_cache_key(ticker, "price_data")
    
    # Check user's cache
    if is_cache_valid(cache_key, max_age_hours=24):
        hist, market = st.session_state.cache[cache_key]
        logger.info(
            f"[SESSION: {st.session_state.session_id}] "
            f"Cache HIT for {ticker}"
        )
        return hist, market
    
    # Not cached or stale - fetch fresh
    logger.info(
        f"[SESSION: {st.session_state.session_id}] "
        f"Cache MISS for {ticker} - fetching fresh data"
    )
    hist, market = get_price_data(ticker)
    
    # Store in user's cache
    if hist is not None:
        st.session_state.cache[cache_key] = (hist, market)
        st.session_state.cache_timestamps[cache_key] = datetime.now()
        log_request(ticker, 'data_fetch', True)
    else:
        log_request(ticker, 'data_fetch', False, 'fetch_returned_none')
    
    return hist, market
```

**Where to insert:** After the `get_price_data()` function ends

---

## Patch 4: Update Main Analysis Call
**File:** `quant_engine.py`  
**Location:** Line ~3607 in main execution  
**Changes:** 1 line modified + 1 line added

### Find this:
```python
if analyze_btn:
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # DATA COLLECTION
            hist, market = get_price_data(ticker)  # ← CHANGE THIS
            if hist is None or hist.empty:
```

### Replace with:
```python
if analyze_btn:
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # Store in session for reference
            st.session_state.current_ticker = ticker
            
            # DATA COLLECTION (with per-user caching)
            hist, market = get_price_data_with_user_cache(ticker)  # ← UPDATED
            if hist is None or hist.empty:
```

---

## Patch 5: Update Audit Logs with Session Context
**File:** `quant_engine.py`  
**Location:** Line ~1375 (first audit log in get_price_data)  
**Changes:** 2 log statements updated

### Find this:
```python
    # ===== SECURITY: Audit logging - track data access =====
    access_log_data = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'action': 'fetch_price_data',
        'source': 'yfinance',
        'request_id': hashlib.md5(f"{datetime.now().isoformat()}{ticker}".encode()).hexdigest()[:8]
    }
    logger.info(
        f"AUDIT: Data access - ticker={ticker}, "
        f"request_id={access_log_data['request_id']}"
    )
```

### Replace with:
```python
    # ===== SECURITY: Audit logging - track data access =====
    access_log_data = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'action': 'fetch_price_data',
        'source': 'yfinance',
        'request_id': hashlib.md5(f"{datetime.now().isoformat()}{ticker}".encode()).hexdigest()[:8]
    }
    logger.info(
        f"AUDIT: Data access - "
        f"session_id={st.session_state.session_id}, "
        f"ticker={ticker}, "
        f"request_id={access_log_data['request_id']}"
    )
```

### Also update success log around line 1413:
```python
    # OLD:
    audit_msg = (
        f"AUDIT: Data fetch successful - ticker={ticker}, "
        f"rows={len(hist)}, completeness={completeness:.1%}, "
        f"request_id={access_log_data['request_id']}"
    )
    
    # NEW:
    audit_msg = (
        f"AUDIT: Data fetch successful - "
        f"session_id={st.session_state.session_id}, "
        f"ticker={ticker}, "
        f"rows={len(hist)}, completeness={completeness:.1%}, "
        f"request_id={access_log_data['request_id']}"
    )
```

---

## Patch 6: Remove Global Rate Limiter
**File:** `quant_engine.py`  
**Location:** Line ~831  
**Changes:** Comment out this line

### Find this:
```python
# Initialize rate limiter
RATE_LIMITER = RequestRateLimiter(max_requests=50, window_seconds=60)
```

### Replace with:
```python
# Note: Rate limiter moved to per-user session state
# Each user now has independent rate limiter via st.session_state
# RATE_LIMITER = RequestRateLimiter(max_requests=50, window_seconds=60)  # DEPRECATED
```

---

## Patch 7: Add Session Info Display (Optional)
**File:** `quant_engine.py`  
**Location:** After line 3552 (after title)  
**Lines:** ~30 new lines (optional)

```python
def show_session_info():
    """Display session info in sidebar for debugging."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Session Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Session ID", st.session_state.session_id[:8])
        with col2:
            cache_size = len(st.session_state.cache)
            st.metric("Cached Items", cache_size)
        
        request_count = len(st.session_state.request_history)
        st.metric("Requests", request_count)
        
        if 'session_start_time' in st.session_state:
            duration = datetime.now() - st.session_state.session_start_time
            duration_str = f"{duration.total_seconds()/60:.1f}m"
            st.metric("Session Duration", duration_str)
        
        if st.sidebar.button("Clear Cache"):
            st.session_state.cache = {}
            st.session_state.cache_timestamps = {}
            st.success("Cache cleared!")

# Call after title
st.title("Quantitative Analytics Dashboard")
st.caption("Investment research platform")

# Display session info (helpful for debugging)
show_session_info()
```

---

## Patch 8: Update Analysis Completion Log
**File:** `quant_engine.py`  
**Location:** End of main analysis block (line ~5921)  
**Changes:** 1 log statement updated

### Find this:
```python
        finally:
            logger.info(f"Analysis session completed for {ticker}")
            logger.info("="*80)
```

### Replace with:
```python
        finally:
            logger.info(
                f"[SESSION: {st.session_state.session_id}] "
                f"Analysis completed for {ticker}"
            )
            logger.info("="*80)
```

---

## Complete Patch Application Checklist

- [ ] **Patch 1:** Session initialization functions added
- [ ] **Patch 1:** `init_session_state()` called after `st.set_page_config()`
- [ ] **Patch 2:** Rate limiter updated in `get_price_data()`
- [ ] **Patch 3:** `get_price_data_with_user_cache()` function added
- [ ] **Patch 4:** Main analysis updated to use new cache function
- [ ] **Patch 5:** Audit logs updated with session_id
- [ ] **Patch 6:** Global RATE_LIMITER commented out
- [ ] **Patch 7:** Session info functions added (optional)
- [ ] **Patch 8:** Final logs updated with session_id

---

## Testing After Patches

### Quick Smoke Test (5 minutes)
```bash
streamlit run quant_engine.py
# Should start without errors
# Should show session ID in sidebar
# Should allow one analysis to complete
```

### Two-Browser Concurrency Test (15 minutes)
```
1. Open Browser 1: http://localhost:8501
2. Open Browser 2: http://localhost:8501
3. Browser 1: Enter "AAPL" → "Run Analysis"
4. While running, Browser 2: Enter "MSFT" → "Run Analysis"
5. Verify:
   ✅ Both complete successfully
   ✅ Different session IDs in sidebars
   ✅ AAPL shown in Browser 1, MSFT in Browser 2
   ✅ No errors or interference
```

### Verification Checklist
- [ ] App starts without errors
- [ ] Session ID displayed (looks like: a1b2c3d4e5f6)
- [ ] Cache counter works (shows number of cached tickers)
- [ ] Single user analysis works
- [ ] Two concurrent analyses don't interfere
- [ ] Each browser sees different data
- [ ] Logs show session_id with each request

---

## Rollback (If Needed)

If you need to revert the patches:

1. Restore `RATE_LIMITER` initialization (uncomment line 831)
2. Remove all `st.session_state.` references
3. Remove `init_session_state()` calls
4. Revert `get_price_data_with_user_cache()` back to direct `get_price_data()` calls
5. Remove session ID from logs

Alternatively, just restore from Git if you have version control.

---

## Next Steps

After these patches are applied and tested:

1. Run the test suite to ensure no regressions
2. Document any issues encountered
3. Plan Phase 2 (authentication) based on these results
4. Monitor concurrent user scenarios

---

## Support

If you encounter issues:

1. Check the error message carefully
2. Verify you applied all 8 patches correctly
3. Look for any line number shifts (if code changed)
4. Run a single-user test first before multi-user
5. Check logs for detailed error information

**Time estimate to apply all patches:** 30-45 minutes  
**Time estimate to test:** 20-30 minutes  
**Total:** ~1 hour
