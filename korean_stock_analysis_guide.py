"""
Korean Stock Market Data Analysis with pykrx and Financial Libraries
================================================================

This guide provides comprehensive information about pykrx library functions
and related financial analysis tools for Korean stock market data.

Dependencies:
    pip install pykrx pandas numpy ta quantstats empyrical yfinance

"""

# ============================================================================
# 1. PYKRX LIBRARY - Korean Stock Market Data
# ============================================================================

from pykrx import stock
from pykrx import bond
from pykrx import derivative
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# STOCK PRICE DATA
# -----------------------------------------------------------------------------

def get_stock_price_data():
    """
    Get historical stock price data from KRX
    """
    # Get stock price data for a specific ticker
    # stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
    df = stock.get_market_ohlcv_by_date("20220101", "20231231", "005930")  # Samsung
    
    # Get all tickers in KOSPI
    tickers = stock.get_market_ticker_list(market="KOSPI")
    
    # Get company names
    ticker_name = stock.get_market_ticker_name("005930")  # Returns company name
    
    # Get multiple stocks data
    # stock.get_market_ohlcv_by_ticker(date, market="KOSPI")
    df_all = stock.get_market_ohlcv_by_ticker("20231201", market="KOSPI")
    
    return df, tickers, ticker_name, df_all

# -----------------------------------------------------------------------------
# FUNDAMENTAL DATA
# -----------------------------------------------------------------------------

def get_fundamental_data():
    """
    Get fundamental financial data including EPS, revenue, ROE
    """
    # Get fundamental data by date
    # stock.get_market_fundamental_by_date(fromdate, todate, ticker)
    fundamental_df = stock.get_market_fundamental_by_date("20220101", "20231231", "005930")
    # Returns: BPS, PER, PBR, EPS, DIV, DPS
    
    # Get fundamental data for all stocks on specific date
    # stock.get_market_fundamental_by_ticker(date, market="KOSPI")
    all_fundamental = stock.get_market_fundamental_by_ticker("20231201", market="KOSPI")
    
    # Get business performance (revenue, operating profit, net income)
    # Note: This requires specific date ranges and may need additional processing
    
    return fundamental_df, all_fundamental

# -----------------------------------------------------------------------------
# INSTITUTIONAL & FOREIGN INVESTOR DATA
# -----------------------------------------------------------------------------

def get_investor_data():
    """
    Get institutional and foreign investor trading data
    """
    # Get trading data by investor type
    # stock.get_market_trading_value_by_date(fromdate, todate, ticker, detail=True)
    trading_df = stock.get_market_trading_value_by_date("20230101", "20231231", "005930", detail=True)
    # Returns trading value by: 개인, 기관합계, 외국인, 기타법인 etc.
    
    # Get net buying by investor type
    # stock.get_market_net_purchases_of_equities_by_ticker(date, market, investor)
    # investor options: "기관합계", "외국인", "개인"
    institutional = stock.get_market_net_purchases_of_equities_by_ticker("20231201", "KOSPI", "기관합계")
    foreign = stock.get_market_net_purchases_of_equities_by_ticker("20231201", "KOSPI", "외국인")
    
    return trading_df, institutional, foreign

# -----------------------------------------------------------------------------
# MARKET INDICES DATA
# -----------------------------------------------------------------------------

def get_market_indices():
    """
    Get market indices data (KOSPI, KOSDAQ, etc.)
    """
    # Get KOSPI index data
    kospi = stock.get_index_ohlcv_by_date("20220101", "20231231", "1001")  # KOSPI
    kosdaq = stock.get_index_ohlcv_by_date("20220101", "20231231", "2001")  # KOSDAQ
    
    # Get index portfolio (constituent stocks)
    kospi_stocks = stock.get_index_portfolio_deposit_file("1001")  # KOSPI constituents
    
    return kospi, kosdaq, kospi_stocks

# -----------------------------------------------------------------------------
# BOND AND DERIVATIVE DATA
# -----------------------------------------------------------------------------

def get_bond_derivative_data():
    """
    Get bond and derivative market data
    """
    # Bond data
    bond_yields = bond.get_otc_treasury_yields("20230101", "20231231")
    
    # Derivative data (futures, options)
    # derivative functions available for futures and options data
    
    return bond_yields

# ============================================================================
# 2. FINANCIAL CALCULATIONS AND ANALYSIS
# ============================================================================

def calculate_growth_rates(df):
    """
    Calculate various growth rates and CAGR
    """
    # Revenue/earnings growth rate (year-over-year)
    df['YoY_Growth'] = df['Close'].pct_change(periods=252)  # Assuming 252 trading days per year
    
    # Calculate CAGR (Compound Annual Growth Rate)
    def calculate_cagr(start_value, end_value, num_years):
        return (end_value / start_value) ** (1 / num_years) - 1
    
    # Example CAGR calculation
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    years = len(df) / 252  # Approximate years
    cagr = calculate_cagr(start_price, end_price, years)
    
    return df, cagr

def calculate_financial_ratios(fundamental_df):
    """
    Calculate financial ratios from fundamental data
    """
    # ROE calculation (if net income and equity data available)
    # ROE = Net Income / Shareholders' Equity
    
    # ROA calculation
    # ROA = Net Income / Total Assets
    
    # Debt-to-Equity ratio
    # D/E = Total Debt / Total Equity
    
    # Price-to-Book ratio (already in pykrx as PBR)
    pbr = fundamental_df['PBR']
    
    # Price-to-Earnings ratio (already in pykrx as PER)
    per = fundamental_df['PER']
    
    return fundamental_df

# ============================================================================
# 3. TECHNICAL ANALYSIS LIBRARIES
# ============================================================================

import ta
# Alternative: import talib (requires separate installation)

def calculate_technical_indicators(df):
    """
    Calculate technical indicators using ta library
    """
    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # Bollinger Bands
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_Mid'] = ta.volatility.bollinger_mavg(df['Close'])
    
    # Stochastic Oscillator
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    
    # Volume indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'])
    
    # Volatility indicators
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    return df

# ============================================================================
# 4. PORTFOLIO ANALYSIS WITH QUANTSTATS
# ============================================================================

import quantstats as qs

def portfolio_analysis(returns):
    """
    Comprehensive portfolio analysis using quantstats
    """
    # Basic statistics
    sharpe_ratio = qs.stats.sharpe(returns)
    max_drawdown = qs.stats.max_drawdown(returns)
    volatility = qs.stats.volatility(returns)
    
    # Risk metrics
    var_95 = qs.stats.var(returns, confidence=0.95)
    cvar_95 = qs.stats.cvar(returns, confidence=0.95)
    
    # Performance metrics
    total_return = qs.stats.comp(returns)
    annual_return = qs.stats.cagr(returns)
    
    # Generate full report
    # qs.reports.html(returns, output='report.html')  # Creates HTML report
    
    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Total Return': total_return,
        'Annual Return': annual_return
    }
    
    return metrics

# ============================================================================
# 5. EMPYRICAL - FINANCIAL STATISTICS
# ============================================================================

import empyrical as ep

def calculate_empyrical_metrics(returns, benchmark_returns=None):
    """
    Calculate financial metrics using empyrical library
    """
    # Basic metrics
    annual_return = ep.annual_return(returns)
    annual_vol = ep.annual_volatility(returns)
    sharpe = ep.sharpe_ratio(returns)
    max_dd = ep.max_drawdown(returns)
    
    # Advanced metrics
    calmar = ep.calmar_ratio(returns)
    omega = ep.omega_ratio(returns)
    sortino = ep.sortino_ratio(returns)
    
    # Benchmark comparison (if benchmark provided)
    if benchmark_returns is not None:
        alpha = ep.alpha(returns, benchmark_returns)
        beta = ep.beta(returns, benchmark_returns)
        
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'Omega Ratio': omega,
            'Sortino Ratio': sortino,
            'Alpha': alpha,
            'Beta': beta
        }
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Omega Ratio': omega,
        'Sortino Ratio': sortino
    }

# ============================================================================
# 6. MARKET DIRECTION ANALYSIS
# ============================================================================

def analyze_market_direction(price_df, volume_df, institutional_df, foreign_df):
    """
    Comprehensive market direction analysis
    """
    # Price momentum analysis
    price_df['Price_MA_5'] = price_df['Close'].rolling(window=5).mean()
    price_df['Price_MA_20'] = price_df['Close'].rolling(window=20).mean()
    price_df['Price_Trend'] = np.where(price_df['Price_MA_5'] > price_df['Price_MA_20'], 1, -1)
    
    # Volume analysis
    volume_df['Volume_MA'] = volume_df['Volume'].rolling(window=20).mean()
    volume_df['Volume_Strength'] = volume_df['Volume'] / volume_df['Volume_MA']
    
    # Institutional sentiment
    institutional_df['Institutional_Net'] = institutional_df['매수거래량'] - institutional_df['매도거래량']
    institutional_sentiment = institutional_df['Institutional_Net'].rolling(window=5).sum()
    
    # Foreign sentiment
    foreign_df['Foreign_Net'] = foreign_df['매수거래량'] - foreign_df['매도거래량']
    foreign_sentiment = foreign_df['Foreign_Net'].rolling(window=5).sum()
    
    # Combined market direction score
    direction_score = (
        price_df['Price_Trend'].iloc[-1] * 0.4 +
        np.sign(institutional_sentiment.iloc[-1]) * 0.3 +
        np.sign(foreign_sentiment.iloc[-1]) * 0.3
    )
    
    return {
        'Price Trend': price_df['Price_Trend'].iloc[-1],
        'Volume Strength': volume_df['Volume_Strength'].iloc[-1],
        'Institutional Sentiment': institutional_sentiment.iloc[-1],
        'Foreign Sentiment': foreign_sentiment.iloc[-1],
        'Overall Direction Score': direction_score
    }

# ============================================================================
# 7. USAGE EXAMPLES
# ============================================================================

def main_analysis_example():
    """
    Complete analysis example using all functions
    """
    # 1. Get stock data
    ticker = "005930"  # Samsung Electronics
    start_date = "20220101"
    end_date = "20231231"
    
    # Price data
    price_df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    
    # Fundamental data
    fundamental_df = stock.get_market_fundamental_by_date(start_date, end_date, ticker)
    
    # Investor data
    trading_df = stock.get_market_trading_value_by_date(start_date, end_date, ticker, detail=True)
    
    # 2. Calculate technical indicators
    price_df = calculate_technical_indicators(price_df)
    
    # 3. Calculate growth rates
    price_df, cagr = calculate_growth_rates(price_df)
    
    # 4. Calculate returns for portfolio analysis
    returns = price_df['Close'].pct_change().dropna()
    
    # 5. Portfolio analysis
    portfolio_metrics = portfolio_analysis(returns)
    empyrical_metrics = calculate_empyrical_metrics(returns)
    
    # 6. Market direction analysis
    market_direction = analyze_market_direction(
        price_df, price_df, trading_df, trading_df
    )
    
    print("=== Analysis Complete ===")
    print(f"CAGR: {cagr:.2%}")
    print(f"Portfolio Metrics: {portfolio_metrics}")
    print(f"Market Direction: {market_direction}")
    
    return price_df, fundamental_df, portfolio_metrics, market_direction

# ============================================================================
# 8. KEY PYKRX FUNCTIONS SUMMARY
# ============================================================================

"""
ESSENTIAL PYKRX FUNCTIONS FOR KOREAN STOCK ANALYSIS:

STOCK PRICE DATA:
- stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
- stock.get_market_ohlcv_by_ticker(date, market)
- stock.get_market_ticker_list(market="KOSPI"/"KOSDAQ")
- stock.get_market_ticker_name(ticker)

FUNDAMENTAL DATA:
- stock.get_market_fundamental_by_date(fromdate, todate, ticker)
- stock.get_market_fundamental_by_ticker(date, market)
  Returns: BPS, PER, PBR, EPS, DIV, DPS

INVESTOR DATA:
- stock.get_market_trading_value_by_date(fromdate, todate, ticker, detail=True)
- stock.get_market_net_purchases_of_equities_by_ticker(date, market, investor)
  investor options: "기관합계", "외국인", "개인"

MARKET INDICES:
- stock.get_index_ohlcv_by_date(fromdate, todate, index_code)
- stock.get_index_portfolio_deposit_file(index_code)
  Index codes: "1001"=KOSPI, "2001"=KOSDAQ

BONDS:
- bond.get_otc_treasury_yields(fromdate, todate)

TECHNICAL ANALYSIS (TA LIBRARY):
- ta.trend.sma_indicator(close, window)
- ta.trend.ema_indicator(close, window)
- ta.trend.macd_diff(close)
- ta.momentum.rsi(close, window)
- ta.volatility.bollinger_hband(close)
- ta.volume.on_balance_volume(close, volume)

PORTFOLIO ANALYSIS (QUANTSTATS):
- qs.stats.sharpe(returns)
- qs.stats.max_drawdown(returns)
- qs.stats.cagr(returns)
- qs.stats.var(returns, confidence)

EMPYRICAL METRICS:
- ep.annual_return(returns)
- ep.sharpe_ratio(returns)
- ep.alpha(returns, benchmark)
- ep.beta(returns, benchmark)
"""

if __name__ == "__main__":
    # Run example analysis
    main_analysis_example()