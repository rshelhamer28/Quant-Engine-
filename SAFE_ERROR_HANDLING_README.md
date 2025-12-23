# Safe Error Handling & Logging System

## Quick Start

The Quantitative Analytics Engine now includes a comprehensive safe error handling system that:

1. **Prevents Information Disclosure** - No sensitive data in user messages or logs
2. **Maintains Full Debugging Capability** - Complete error details securely logged
3. **Improves User Experience** - Clear, actionable error messages
4. **Ensures Compliance** - OWASP-aligned security practices

## Key Features

### ✅ Automatic Sensitive Data Masking
```
Input:  "Error fetching data: API key sk_live_abc123xyz"
Output: "Error fetching data: API key ***API_KEY***"
```

### ✅ Safe User Messages
```python
# User sees this
"❌ Data retrieval failed. The stock data may be unavailable or incomplete."

# Logs contain this
"ERROR Exception [ConnectionError] in get_price_data (ticker=AAPL): 
HTTPSConnectionPool(...): Max retries exceeded with url: /v10/quote..."
```

### ✅ Structured Exception Classification
- Timeout errors → Network availability advice
- Data errors → Ticker validation suggestions
- Calculation errors → Data quality explanation
- API errors → Service availability notice

### ✅ Audit Logging for Security
```
AUDIT: Data access - ticker=AAPL, request_id=a1b2c3d4
SECURITY: Rate limit exceeded for ticker GOOGL
```

## Usage

### Basic Exception Handling

```python
try:
    # Do something that might fail
    data = yf.download(ticker, period='1y')
except Exception as e:
    # Log securely with context
    SafeErrorHandler.log_exception(e, "fetch_price_data", ticker)
    # Continue without exposing error details
    return None
```

### User-Facing Error Messages

```python
try:
    hist = get_price_data(ticker)
    if hist is None:
        safe_msg = SafeErrorHandler.safe_error_message('data_error')
        st.error(f"❌ {safe_msg}")
except Exception as e:
    error_type = SafeErrorHandler.log_exception(e, "main_analysis", ticker)
    safe_msg = SafeErrorHandler.safe_error_message(error_type)
    st.error(f"❌ {safe_msg}")
```

## Components

### SafeErrorHandler Class

**`safe_error_message(error_type: str) -> str`**
- Maps exception types to user-friendly messages
- Types: `timeout`, `network`, `data_error`, `api_error`, `invalid_input`, `calculation_error`, `file_error`, `unknown`

**`log_exception(exception: Exception, context: str, ticker: str = "") -> str`**
- Logs full exception details with context
- Automatically masks sensitive data
- Returns error type for message mapping

**`handle_and_report(exception, context, ticker, default_return) -> tuple`**
- Complete exception handler in one call
- Returns: (default_value, safe_message, error_type)

### Utility Functions

**`mask_sensitive_data(text: str) -> str`**
- Masks API keys, tokens, passwords
- Masks email addresses
- Masks credential patterns

**`_safe_divide(numerator, denominator, default=None) -> Optional[float]`**
- Safe division preventing ZeroDivisionError
- Returns default on error

**`_validate_numeric_value(value, name, min_val, max_val) -> Optional[float]`**
- Validates numeric values (no NaN, Inf)
- Enforces min/max bounds
- Safe for mathematical operations

## Where It's Used

### Data Layer
- ✅ `get_price_data()` - Handles network/data errors
- ✅ `get_fundamental_data()` - Handles API/validation errors
- ✅ `load_json_secure()` - Handles file access errors

### Calculation Layer
- ✅ `calculate_advanced_metrics()` - Handles math errors
- ✅ `calculate_enhanced_var()` - Handles statistical errors
- ✅ `professional_monte_carlo()` - Handles simulation errors

### Presentation Layer
- ✅ Main analysis exception handler - Catches all unhandled errors
- ✅ Data retrieval error display - Shows safe messages
- ✅ Calculation error display - Shows safe messages
- ✅ Monte Carlo error display - Shows safe messages

## Security Benefits

| Risk | Mitigation |
|------|-----------|
| Information Disclosure | Safe generic messages, detailed logs only |
| Sensitive Data Exposure | Automatic masking of credentials, keys, emails |
| Error-Based Attacks | No stack traces or system internals exposed |
| Audit Trail | Complete logging for security investigations |
| Rate Limit Abuse | Requests rejected with security logging |
| Directory Traversal | Path validation prevents unauthorized access |

## Log Format

### File Logs (logs/quant_engine.log)
```
2024-01-15 14:23:45 | ERROR    | __main__    | Exception [Timeout] in get_price_data (ticker=AAPL): HTTPSConnectionPool(host='query2.finance.yahoo.com', port=443): Read timed out. 
2024-01-15 14:23:46 | AUDIT    | __main__    | Data access - ticker=AAPL, request_id=a1b2c3d4
2024-01-15 14:24:12 | SECURITY | __main__    | Rate limit exceeded for ticker GOOGL
```

### User Messages (Streamlit UI)
```
❌ Data retrieval failed. The stock data may be unavailable or incomplete.

💡 Try: AAPL, MSFT, TSLA, or other major US-listed stocks with 5+ years of data
```

## Examples

### Example 1: Timeout Error

**What Happens Internally:**
```
HTTPSConnectionPool(...): Read timed out. (Connection timeout)
    ↓
SafeErrorHandler.log_exception() → classified as 'timeout'
    ↓
Log entry: "Exception [Timeout] in get_price_data (ticker=AAPL): Read timed out..."
    ↓
safe_error_message('timeout') → "Request timed out. The data source may be slow..."
    ↓
User sees: "❌ Request timed out. The data source may be slow or unavailable. Please try again in a moment."
```

### Example 2: API Key Leak Prevention

**Scenario:** yfinance raises error mentioning API key

**Without Safe Handling:**
```
❌ Error: API request failed with key=sk_live_abc123xyz_secret
```

**With Safe Handling:**
```
File logs: "Exception [API Error] in get_fundamental_data (ticker=TSLA): API request failed with key=***API_KEY***"
User sees: "❌ External API error. The data source encountered an issue. Please try again later."
```

### Example 3: Calculation Error

**Data Issues:**
```
Insufficient data points, NaN values, mathematical undefined operation
    ↓
SafeErrorHandler.log_exception() → classified as 'calculation_error'
    ↓
Detailed traceback logged for debugging
    ↓
User sees: "❌ Calculation error. The data may be insufficient for this operation."
    ↓
Helpful tips: Verify ticker, try major stocks, check data completeness
```

## Best Practices

### ✅ DO

```python
# Log with context
SafeErrorHandler.log_exception(e, "operation_name", ticker)

# Show safe messages to users
safe_msg = SafeErrorHandler.safe_error_message(error_type)
st.error(f"❌ {safe_msg}")

# Add helpful context in UI
st.info("💡 Suggestions for troubleshooting...")

# Use safe division/validation
value = _safe_divide(numerator, denominator, default=0)
validated = _validate_numeric_value(value, "name", min_val, max_val)
```

### ❌ DON'T

```python
# Don't expose raw exceptions
st.error(f"Error: {str(e)}")

# Don't log without context
logger.error(f"Failed: {e}")

# Don't divide without checking
ratio = numerator / denominator  # May raise ZeroDivisionError

# Don't trust unvalidated data
result = 100 / float(value)  # May be NaN or Inf
```

## Monitoring

### Check for Security Events
```bash
grep "SECURITY:" logs/quant_engine.log
# Shows rate limiting, path traversal attempts, suspicious access patterns
```

### Check for Errors
```bash
grep "ERROR" logs/quant_engine.log | tail -20
# Latest error details for debugging
```

### Monitor Specific Ticker
```bash
grep "ticker=AAPL" logs/quant_engine.log
# All operations for that ticker
```

### Verify No Sensitive Data
```bash
grep -i "password\|secret\|token\|api" logs/quant_engine.log | grep -v "MASKED\|API_KEY"
# Should be empty - all sensitive data masked
```

## Configuration

### Error Message Types

Located in `SafeErrorHandler.safe_error_message()`:

```python
error_messages = {
    'timeout': '...',           # Network timeouts
    'network': '...',           # Connection errors
    'data_error': '...',        # Data retrieval failures
    'api_error': '...',         # External API failures
    'invalid_input': '...',     # Input validation
    'calculation_error': '...', # Math/calculation failures
    'file_error': '...',        # File system errors
    'unknown': '...',           # Unclassified
}
```

Customize messages as needed for your deployment.

### Sensitive Data Patterns

Located in `mask_sensitive_data()`:

```python
def mask_sensitive_data(text: str) -> str:
    # API keys
    text = text.replace(os.getenv('FINNHUB_API_KEY', ''), '***API_KEY***')
    # Email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '***EMAIL***', text)
    # Credentials
    text = re.sub(r'(password|token|secret|key)\s*[=:]\s*[^\s,}]+', 
                  r'\1=***MASKED***', text, flags=re.IGNORECASE)
    return text
```

Add patterns as needed for your environment.

## Testing

### Manual Test Checklist

- [ ] Invalid ticker returns safe error message
- [ ] Network timeout handled gracefully  
- [ ] Logs contain full error details
- [ ] No sensitive data in user messages
- [ ] No sensitive data in logs
- [ ] Helpful suggestions provided in UI
- [ ] Application continues running after error

### Automated Verification

```python
def test_error_handling():
    # Test safe message generation
    msg = SafeErrorHandler.safe_error_message('timeout')
    assert 'timed out' in msg.lower()
    assert 'yfinance' not in msg  # No internal details
    
    # Test data masking
    masked = mask_sensitive_data("password=secret123")
    assert '***MASKED***' in masked
    assert 'secret' not in masked
    
    # Test exception logging
    try:
        raise ValueError("test error")
    except Exception as e:
        error_type = SafeErrorHandler.log_exception(e, "test_context", "AAPL")
        assert error_type == 'invalid_input'  # ValueError → invalid_input
```

## Troubleshooting

### Problem: Users see technical error messages
**Solution:** Check that all exception handlers use SafeErrorHandler
```bash
grep "st.error" quant_engine.py | grep -v "safe_msg"
# These should use SafeErrorHandler
```

### Problem: Can't debug production errors
**Solution:** Check the full logs file
```bash
# Full error details in logs/quant_engine.log
tail -100 logs/quant_engine.log | grep ERROR
```

### Problem: Sensitive data appearing in logs
**Solution:** Update mask_sensitive_data() with new patterns
```python
# Add new patterns to mask_sensitive_data()
text = re.sub(r'new_pattern', '***MASKED***', text)
```

## Documentation

For comprehensive documentation, see:
- **ERROR_HANDLING_GUIDE.md** - Complete API reference and examples
- **ERROR_HANDLING_SUMMARY.md** - Implementation details and best practices

## Support

If you encounter errors or have questions:

1. Check the error message for suggestions
2. Review the logs: `logs/quant_engine.log`
3. Try the suggested troubleshooting steps
4. For persistent issues, collect logs and contact support

---

**Version:** 1.0  
**Status:** Production Ready  
**Last Updated:** 2024  
**Compatibility:** All Python 3.12+ environments
