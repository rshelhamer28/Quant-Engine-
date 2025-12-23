# Production Resilience Risk Analysis

## CRITICAL BREAKING POINTS

### 1. **yfinance Data Structure Changes** 🔴 CRITICAL
**Risk Level**: HIGH  
**Location**: Lines 500-501, 507-515, 532, 665-683

**Problem**: Code assumes specific column names and field names in yfinance responses:
- Expects columns: `'Adj Close'`, `'Close'`, `'Adj_Close'`, or `'CLOSE'`
- Expects info dict fields: `currentPrice`, `regularMarketPrice`, `targetMeanPrice`, `shortPercentOfFloat`, `trailingPE`, `forwardPE`, etc.

**Breaking Scenarios**:
- yfinance updates and changes column names (e.g., removes `Adj Close`)
- yfinance restructures ticker.info response format
- yfinance changes default behavior of `auto_adjust=True`

**Current Mitigation**: 
- Multiple column name fallbacks (lines 509-515)
- Price field fallbacks (lines 643-660)
- **INSUFFICIENT**: If ALL fallback fields are missing, function returns None

**Recommended Fix**:
```python
# Add comprehensive logging of actual structure received
def get_price_data(ticker):
    hist = yf.download(...)
    if hist.empty:
        logger.critical(f"yfinance returned empty data. Columns: {hist.columns.tolist() if not hist.empty else 'N/A'}")
        # Alert monitoring system
    
    # Log actual structure for debugging
    logger.info(f"yfinance structure for {ticker}: {hist.columns.tolist()}")
```

---

### 2. **Finnhub API Response Schema Changes** 🔴 CRITICAL
**Risk Level**: HIGH  
**Location**: Lines 815-840, 850-885

**Problem**: Code parses Finnhub news API expecting:
```python
{
    'headline': str,
    'source': str,
    'datetime': int,
    'summary': str,
    'url': str
}
```

**Breaking Scenarios**:
- Finnhub changes field names (e.g., `headline` → `title`)
- Finnhub removes fields without warning
- Finnhub changes datetime format from Unix timestamp to ISO string
- Finnhub changes response structure from list to dict

**Current Mitigation**: 
- Uses `.get()` with fallbacks
- Type validation (lines 833-835)
- **INSUFFICIENT**: No versioning check; no API version pinning

**Recommended Fix**:
```python
# Add response schema validation
def validate_finnhub_response(response_data):
    required_fields = ['headline', 'source', 'datetime']
    if not isinstance(response_data, list):
        raise ValueError(f"Expected list, got {type(response_data)}")
    
    for item in response_data:
        if not all(field in item for field in required_fields):
            raise ValueError(f"Missing required fields: {item.keys()}")
    
    return response_data
```

---

### 3. **Hardcoded Sector P/E Ratios** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Lines 335-343

**Problem**: Sector average P/E ratios are hardcoded:
```python
sector_avg_pe = {
    'Technology': 25, 'Healthcare': 20, 'Financial': 15,
    'Consumer Cyclical': 18, 'Industrials': 19, 'Energy': 10
}
```

**Breaking Scenarios**:
- Market conditions change radically (recession → all sectors devalue)
- New sectors added that aren't in this list (defaults to 20)
- Sector names from yfinance change (e.g., "Tech" vs "Technology")

**Current Mitigation**: 
- Default value of 20 (line 340)
- **INSUFFICIENT**: Stale data; no dynamic updates

**Recommended Fix**:
```python
# Load from external source (database/API)
# Or calculate from historical sector data
def get_sector_pe_ratio(sector, fallback=20):
    # Query database or cache
    # Update annually or on-demand
    return sector_pe_db.get(sector, fallback)
```

---

### 4. **Price Column Detection Order Dependency** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Lines 509-515

**Problem**: Code iterates through column candidates in fixed order:
```python
for candidate in ['Adj Close', 'Close', 'Adj_Close', 'CLOSE']:
    if candidate in hist.columns:
        price_col = candidate
        break
```

**Breaking Scenarios**:
- yfinance adds new column type but code selects wrong one first
- Column order in `hist.columns` matters but code ignores it
- yfinance deprecates `Adj Close` without removing it (zombie column)

**Current Mitigation**: 
- Fallback chain
- **INSUFFICIENT**: No validation that selected column makes sense

**Recommended Fix**:
```python
# Prefer adjusted close, validate logic
price_col = hist.columns[[col for col in ['Adj Close', 'Close'] if col in hist.columns][0]]
# Validate: Adj Close should be <= Close (or warn if inverted)
if hist['Adj Close'].max() > hist['Close'].max():
    logger.warning("Adjusted close > regular close. Data may be corrupted.")
```

---

### 5. **Benchmark Hardcoding (S&P 500 = ^GSPC)** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Line 501

**Problem**: Code uses `^GSPC` as global benchmark without configuration:
```python
market = yf.download("^GSPC", period="5y", ...)
```

**Breaking Scenarios**:
- Yahoo Finance changes S&P 500 ticker symbol
- User wants different benchmark (Russell 2000, etc.)
- International users need local indices (DAX, NIKKEI, etc.)

**Current Mitigation**: 
- **NONE**: Hardcoded value
- If fails, function returns None gracefully

**Recommended Fix**:
```python
BENCHMARK_TICKER = os.getenv('BENCHMARK_TICKER', '^GSPC')
# Make configurable per deployment
```

---

### 6. **Timeout Values Too Aggressive** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Lines 500-501, 774

**Problem**: 15-second timeout may be too short in high-latency scenarios:
```python
hist = yf.download(ticker, timeout=15)  # 15 seconds for 5 years of daily data
```

**Breaking Scenarios**:
- Deploy to slow network (AWS Lambda with poor internet)
- yfinance infrastructure slows down
- User's ISP has high latency

**Current Mitigation**: 
- Returns None gracefully on timeout
- Falls back to sample data
- **INSUFFICIENT**: No exponential backoff before giving up

**Recommended Fix**:
```python
# Implement adaptive timeout
def get_price_data_with_retry(ticker, initial_timeout=15, max_timeout=45):
    for attempt in range(3):
        try:
            timeout = initial_timeout * (attempt + 1)
            return yf.download(ticker, timeout=timeout)
        except requests.Timeout:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
```

---

### 7. **Date String Format Assumptions** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Line 807, 870

**Problem**: Code assumes specific date format from Finnhub:
```python
date_str = datetime.fromtimestamp(date).strftime('%b %d, %Y') if isinstance(date, (int, float))
```

**Breaking Scenarios**:
- Finnhub changes timestamp format to ISO string without notice
- Finnhub returns millisecond vs second timestamps inconsistently
- User in non-English locale expects localized date

**Current Mitigation**: 
- Try/except with "Recent" fallback (line 872)
- **INSUFFICIENT**: No logging of actual format received

**Recommended Fix**:
```python
def safe_parse_timestamp(date_value, fallback="Recent"):
    if isinstance(date_value, (int, float)):
        try:
            return datetime.fromtimestamp(date_value).strftime('%b %d, %Y')
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to parse timestamp {date_value}: {e}")
            return fallback
    elif isinstance(date_value, str):
        return date_value  # Already a string
    return fallback
```

---

### 8. **Missing yfinance.Ticker.info Fields** 🟠 HIGH
**Risk Level**: HIGH  
**Location**: Lines 643-695

**Problem**: Code accesses many optional fields from ticker.info:
```python
# Any of these could be missing or None
info.get('targetMeanPrice')
info.get('trailingPE')
info.get('shortPercentOfFloat')
info.get('marketCap')
```

**Breaking Scenarios**:
- yfinance removes support for certain fields
- OTC stocks or penny stocks lack certain data
- yfinance API endpoint changes

**Current Mitigation**: 
- Uses `.get()` with None defaults
- Type coercion with `_coerce_to_float()` (handles missing)
- **PARTIALLY SUFFICIENT**: But `partial_data` flag is set generically

**Recommended Fix**:
```python
# Track which fields are actually missing
result['missing_fields'] = []
if info.get('marketCap') is None:
    result['missing_fields'].append('marketCap')

# Alert if critical fields missing
critical_fields = ['currentPrice', 'marketCap', 'sector']
if len([f for f in critical_fields if info.get(f) is None]) > 1:
    logger.error(f"Missing multiple critical fields for {ticker}")
```

---

### 9. **Sentiment Analysis Dictionary Frozen** 🟠 HIGH
**Risk Level**: MEDIUM  
**Location**: Lines 730-785

**Problem**: Sentiment keyword dictionary is hardcoded:
```python
positive_words = {
    'surged': 0.8, 'surge': 0.8, 'soaring': 0.8, ...
}
negative_words = {...}
```

**Breaking Scenarios**:
- Market evolves; old keywords become irrelevant
- New financial terminology emerges
- Dictionary becomes outdated without manual updates

**Current Mitigation**: 
- Uses Loughran-McDonald dictionary (research-backed)
- Combines with TextBlob (NLP-based)
- **INSUFFICIENT**: No mechanism for updates

**Recommended Fix**:
```python
# Load from external source
def load_sentiment_dictionary():
    # Load from database or S3
    # Update quarterly with latest financial news
    with open('sentiment_dict_v1.2.json') as f:
        return json.load(f)
    
SENTIMENT_DICT = load_sentiment_dictionary()
```

---

### 10. **News API Endpoint URL Hardcoded** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Line 101

**Problem**: API endpoint is hardcoded without versioning:
```python
NEWS_API_BASE_URL = "https://finnhub.io/api/v1/company-news"
```

**Breaking Scenarios**:
- Finnhub deprecates v1 API (v2 comes out)
- API endpoint path changes
- API requires new headers/authentication

**Current Mitigation**: 
- **NONE**: Hardcoded string
- If endpoint breaks, entire news feature fails

**Recommended Fix**:
```python
NEWS_API_BASE_URL = os.getenv('FINNHUB_API_URL', 'https://finnhub.io/api/v1/company-news')
NEWS_API_VERSION = os.getenv('FINNHUB_API_VERSION', 'v1')
# Make configurable for API migrations
```

---

### 11. **Cache TTL Fixed** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: Line 451

**Problem**: Cache time-to-live is hardcoded to 300 seconds:
```python
@st.cache_data(ttl=300, show_spinner=False)
```

**Breaking Scenarios**:
- Market opens/closes at different times in different zones
- Holiday schedule changes
- Regulatory changes require more/less fresh data

**Current Mitigation**: 
- **NONE**: Hardcoded value
- Would require code change to update

**Recommended Fix**:
```python
# Configure dynamically
CACHE_TTL = int(os.getenv('CACHE_TTL_SECONDS', '300'))

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_price_data(ticker):
    ...
```

---

### 12. **10,000 Monte Carlo Simulations Hardcoded** 🟡 MEDIUM
**Risk Level**: MEDIUM  
**Location**: ~Line 1200+ (not shown but exists)

**Problem**: Number of simulation paths is hardcoded:
```python
# professional_monte_carlo() likely has: for i in range(10000):
```

**Breaking Scenarios**:
- Algorithm becomes slow on commodity hardware
- Regulatory requirement changes (needs more/fewer simulations)
- Server resources constrained

**Current Mitigation**: 
- **NONE**: Hardcoded number
- Performance impact if changed

**Recommended Fix**:
```python
NUM_SIMULATIONS = int(os.getenv('MONTE_CARLO_PATHS', '10000'))
```

---

## SUMMARY TABLE

| Issue | Severity | Impact | Current Mitigation | Fix Priority |
|-------|----------|--------|-------------------|--------------|
| yfinance column changes | 🔴 CRITICAL | Data parsing fails | Fallback chain | P0 |
| Finnhub schema changes | 🔴 CRITICAL | News feature breaks | Type validation | P0 |
| Sector P/E hardcoded | 🟡 MEDIUM | Inaccurate scoring | Default to 20 | P2 |
| Benchmark ticker | 🟡 MEDIUM | Can't change S&P 500 | Hardcoded | P2 |
| Timeout values | 🟡 MEDIUM | Fails on slow networks | Graceful fallback | P1 |
| Date format changes | 🟡 MEDIUM | Date parsing breaks | Try/except | P1 |
| Missing ticker.info fields | 🟠 HIGH | Incomplete fundamentals | .get() + coercion | P1 |
| Sentiment dict frozen | 🟠 HIGH | Outdated scores | Hardcoded dict | P2 |
| API endpoint URL | 🟡 MEDIUM | API migration pain | Hardcoded URL | P2 |
| Cache TTL fixed | 🟡 MEDIUM | Can't adjust freshness | Hardcoded value | P2 |
| Monte Carlo paths | 🟡 MEDIUM | Performance issues | Hardcoded number | P2 |
| Price column selection | 🟡 MEDIUM | Wrong column picked | Fixed order | P1 |

---

## IMMEDIATE ACTIONS BEFORE GOING LIVE

**P0 (Do Before Launch)**:
1. Add comprehensive logging of data structures received from yfinance and Finnhub
2. Create monitoring alerts for schema changes in API responses
3. Document exact field expectations for both yfinance and Finnhub
4. Add integration tests that verify response structure hasn't changed

**P1 (Do Within 1 Week)**:
1. Convert hardcoded values to environment variables (timeout, cache TTL, benchmark)
2. Add structured error messages when data source structure changes
3. Create fallback data source (cached JSON) if APIs fail repeatedly
4. Add metrics/logging for which fallback paths are used

**P2 (Do Within 1 Month)**:
1. Externalize sentiment dictionary (load from database)
2. Externalize sector P/E ratios (fetch from updated source)
3. Add API version detection and migration logic
4. Create admin dashboard to monitor data quality issues

