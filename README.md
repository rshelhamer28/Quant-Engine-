# Quantitative Analytics Engine

A professional-grade **Streamlit dashboard** for quantitative financial analysis, featuring Monte Carlo simulations, multi-window performance metrics, sentiment analysis, and institutional-grade risk decomposition.

## 🎯 Features

- **Executive Summary**: Real-time conviction scoring, trading signals (BUY/HOLD/SELL/REDUCE), and risk classification
- **Risk & Returns Analysis**: Valuation snapshot, historical performance across 1Y/3Y/5Y windows, advanced metrics (Sharpe, Sortino, Calmar)
- **Advanced Metrics**: Risk decomposition (systematic vs idiosyncratic), rolling beta, tracking error, information ratio
- **Monte Carlo Forecasting**: 10,000 stochastic price paths with confidence intervals, Value-at-Risk (VaR), and terminal price distributions
- **Sentiment & News**: Real-time headline sentiment analysis (Loughran-McDonald + TextBlob), news impact metrics, 30-day trend analysis
- **Comprehensive Dashboard**: 7 interactive tabs with color-coded metrics, institutional-grade visualizations, and rate-sensitive Sharpe interpretation

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/quant_engine.git
   cd quant_engine
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\Activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Finnhub API key
   ```

5. **Run the app**:
   ```bash
   streamlit run quant_engine.py
   ```

Visit `http://localhost:8501` in your browser.

### Environment Variables

```
FINNHUB_API_KEY=your_finnhub_api_key_here
FINNHUB_API_URL=https://finnhub.io/api/v1/company-news
```

Get your free API key at [Finnhub](https://finnhub.io).

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Executive Summary** | Conviction score, trading signal, risk level, key metrics snapshot |
| **Risk & Returns** | Valuation metrics, return vs benchmark, historical performance table |
| **Advanced Metrics** | Sharpe/Sortino/Calmar ratios, alpha, beta, tracking error, risk decomposition |
| **Monte Carlo** | 10,000 stochastic simulations, terminal price distribution, confidence intervals |
| **Sentiment & News** | Headline sentiment analysis, 30-day trend, bullish/neutral/bearish breakdown |
| **Technical Analysis** | Rolling beta, volatility regimes, correlation analysis |
| **Model Diagnostics** | Data quality, execution logs, model assumptions, audit trail |

## 🔍 Key Metrics Explained

### Conviction Score (0-100)
Institutional-grade scoring combining:
- Risk-adjusted returns (Sharpe, Sortino, Information ratio)
- Performance metrics (Alpha, consistency, drawdown protection)
- Valuation (P/E, PEG, sector comparison)
- Technicals (Beta stability, volatility regimes)

**Trading Signals**:
- **STRONG BUY** (score ≥75 + Sharpe >1.0 + Alpha >0.03)
- **BUY** (score ≥60 + Sharpe >0.5 + Alpha >0)
- **HOLD** (mixed or neutral signals)
- **REDUCE** (score ≤40 or Sharpe <0)
- **SELL** (score ≤30)

### Risk Level
- **Low**: vol <20%, max_dd >-20%
- **Medium**: vol <25%, max_dd >-25%, Sharpe >0.5
- **Moderate-High**: vol <35%, max_dd >-35%, Sharpe >-0.2
- **High**: vol ≥35% or max_dd ≤-35%

*Note: Sharpe is rate-sensitive. Defensive assets show "Low (rate-sensitive)" at high risk-free rates.*

### Profit Probability
Monte Carlo metric: probability that forecast terminal price > current price at horizon end.

## 💡 Design Insights

- **Multi-window CAGR**: 1Y/3Y/5Y windows calculated independently. Different entry/exit points explain variance.
- **P/E Clarity**: Uses TTM GAAP earnings (may include one-time items). Compare to analyst forward estimates for context.
- **Sentiment Context**: News sentiment is a positioning indicator, not a return predictor. Strong bullish sentiment does not override weak risk-adjusted returns.
- **Data Freshness**: Live market data with 15-minute yfinance delay.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data**: yfinance (OHLCV), Finnhub (news)
- **Analytics**: NumPy, Pandas, SciPy
- **Visualization**: Plotly
- **NLP**: TextBlob, Loughran-McDonald sentiment lexicon

## 📈 Python Requirements

- Python 3.9+
- See `requirements.txt` for full dependencies

## 🔐 Security

- API keys stored in `.env` (excluded from git via `.gitignore`)
- No authentication required for local use
- For production: use Streamlit Secrets management

## 🚀 Deploy to Streamlit Cloud

1. Push code to GitHub (public or private repo)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" → Select your repo
4. Set environment variables in "Secrets"
5. Deploy!

## 📝 License

MIT License

## 📧 Contact

For questions or collaboration: [your-email]

---

**Disclaimer**: This dashboard is for educational and informational purposes only. Not financial advice. Past performance does not guarantee future results. Always consult a financial advisor before making investment decisions.
