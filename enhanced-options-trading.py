#!/usr/bin/env python3
"""
Enhanced NSE Options Trading System
Real-time data, Pattern Recognition, Multi-timeframe Analysis
Author: AI Trading Assistant
Version: 2.0
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta, date
import ta
import warnings
from scipy.signal import argrelextrema
from scipy.stats import norm
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import calendar
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚀 Enhanced Options Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .metric-card { 
        background: rgba(255,255,255,0.1); 
        padding: 10px; 
        border-radius: 10px; 
        margin: 5px;
        backdrop-filter: blur(10px);
    }
    .signal-strong-buy { 
        background: linear-gradient(45deg, #4CAF50, #8BC34A); 
        color: white; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    .signal-buy { 
        background: linear-gradient(45deg, #2196F3, #03DAC6); 
        color: white; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center;
        font-weight: bold;
    }
    .signal-sell { 
        background: linear-gradient(45deg, #f44336, #FF5722); 
        color: white; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center;
        font-weight: bold;
    }
    .signal-hold { 
        background: linear-gradient(45deg, #FF9800, #FFC107); 
        color: white; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center;
        font-weight: bold;
    }
    .pattern-detected {
        background: linear-gradient(45deg, #9C27B0, #E91E63);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedNSEDataProvider:
    """Enhanced NSE Data Provider with multiple fallbacks"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.cookies = {}
        self.base_urls = {
            'nse': 'https://www.nseindia.com',
            'option_chain': 'https://www.nseindia.com/api/option-chain-indices',
            'indices': 'https://www.nseindia.com/api/allIndices'
        }
    
    def set_cookies(self):
        """Set cookies for NSE access"""
        try:
            response = self.session.get(self.base_urls['nse'], headers=self.headers, timeout=10)
            self.cookies = dict(response.cookies)
            return True
        except Exception as e:
            st.warning(f"Cookie setting failed: {e}")
            return False
    
    def get_live_option_chain(self, symbol: str = "NIFTY") -> Optional[pd.DataFrame]:
        """Fetch live option chain data from NSE"""
        try:
            if not self.set_cookies():
                return self._get_fallback_option_chain(symbol)
            
            url = f"{self.base_urls['option_chain']}?symbol={symbol}"
            response = self.session.get(url, headers=self.headers, cookies=self.cookies, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_option_chain_data(data)
            else:
                return self._get_fallback_option_chain(symbol)
                
        except Exception as e:
            st.error(f"NSE API Error: {e}")
            return self._get_fallback_option_chain(symbol)
    
    def _process_option_chain_data(self, data: dict) -> pd.DataFrame:
        """Process NSE option chain JSON response"""
        try:
            records = data.get('records', {}).get('data', [])
            processed_data = []
            
            for record in records:
                strike = record.get('strikePrice', 0)
                
                # Call data
                ce_data = record.get('CE', {})
                pe_data = record.get('PE', {})
                
                processed_data.append({
                    'strike': strike,
                    'ce_ltp': ce_data.get('lastPrice', 0),
                    'ce_oi': ce_data.get('openInterest', 0),
                    'ce_oi_change': ce_data.get('changeinOpenInterest', 0),
                    'ce_volume': ce_data.get('totalTradedVolume', 0),
                    'ce_iv': ce_data.get('impliedVolatility', 0),
                    'ce_delta': ce_data.get('delta', 0),
                    'ce_gamma': ce_data.get('gamma', 0),
                    'ce_theta': ce_data.get('theta', 0),
                    'ce_vega': ce_data.get('vega', 0),
                    'pe_ltp': pe_data.get('lastPrice', 0),
                    'pe_oi': pe_data.get('openInterest', 0),
                    'pe_oi_change': pe_data.get('changeinOpenInterest', 0),
                    'pe_volume': pe_data.get('totalTradedVolume', 0),
                    'pe_iv': pe_data.get('impliedVolatility', 0),
                    'pe_delta': pe_data.get('delta', 0),
                    'pe_gamma': pe_data.get('gamma', 0),
                    'pe_theta': pe_data.get('theta', 0),
                    'pe_vega': pe_data.get('vega', 0),
                })
            
            return pd.DataFrame(processed_data)
            
        except Exception as e:
            st.error(f"Data processing error: {e}")
            return pd.DataFrame()
    
    def _get_fallback_option_chain(self, symbol: str) -> pd.DataFrame:
        """Fallback option chain data when NSE API fails"""
        st.info("🔄 Using fallback data (NSE API unavailable)")
        
        # Get current price from yfinance as fallback
        ticker_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
        yf_symbol = ticker_map.get(symbol, "^NSEI")
        
        try:
            ticker = yf.Ticker(yf_symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
        except:
            current_price = 24000 if symbol == "NIFTY" else 52000
        
        # Generate realistic option chain
        return self._generate_realistic_option_chain(current_price, symbol)
    
    def _generate_realistic_option_chain(self, spot: float, symbol: str) -> pd.DataFrame:
        """Generate realistic option chain data"""
        strikes = self._get_strikes(spot, symbol)
        data = []
        
        for strike in strikes:
            # Calculate realistic premiums and Greeks
            ce_premium, pe_premium = self._calculate_realistic_premiums(spot, strike, 21)
            ce_greeks = self._calculate_greeks(spot, strike, 21, 18, "CE")
            pe_greeks = self._calculate_greeks(spot, strike, 21, 18, "PE")
            
            # Generate realistic OI and volume
            moneyness = abs(spot - strike) / spot
            base_oi = max(1000, int(25000 * np.exp(-moneyness * 10)))
            
            data.append({
                'strike': strike,
                'ce_ltp': ce_premium,
                'ce_oi': base_oi * (0.8 if strike > spot else 1.2),
                'ce_oi_change': np.random.randint(-30, 50),
                'ce_volume': int(base_oi * np.random.uniform(0.1, 0.8)),
                'ce_iv': np.random.uniform(15, 25),
                'ce_delta': ce_greeks.get('delta', 0),
                'ce_gamma': ce_greeks.get('gamma', 0),
                'ce_theta': ce_greeks.get('theta', 0),
                'ce_vega': ce_greeks.get('vega', 0),
                'pe_ltp': pe_premium,
                'pe_oi': base_oi * (1.2 if strike > spot else 0.8),
                'pe_oi_change': np.random.randint(-40, 30),
                'pe_volume': int(base_oi * np.random.uniform(0.1, 0.8)),
                'pe_iv': np.random.uniform(14, 24),
                'pe_delta': pe_greeks.get('delta', 0),
                'pe_gamma': pe_greeks.get('gamma', 0),
                'pe_theta': pe_greeks.get('theta', 0),
                'pe_vega': pe_greeks.get('vega', 0),
            })
        
        return pd.DataFrame(data)
    
    def _get_strikes(self, spot: float, symbol: str) -> List[float]:
        """Generate option strikes"""
        if symbol == "NIFTY":
            base = round(spot / 50) * 50
            return [base + i * 50 for i in range(-15, 16)]
        else:  # BANKNIFTY
            base = round(spot / 100) * 100
            return [base + i * 100 for i in range(-15, 16)]
    
    def _calculate_realistic_premiums(self, spot: float, strike: float, dte: int) -> Tuple[float, float]:
        """Calculate realistic option premiums"""
        # Simplified Black-Scholes for realistic premiums
        r = 0.065
        sigma = 0.18
        T = dte / 365.0
        
        if T <= 0:
            ce_premium = max(0, spot - strike)
            pe_premium = max(0, strike - spot)
        else:
            d1 = (np.log(spot/strike) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            ce_premium = spot * norm.cdf(d1) - strike * np.exp(-r*T) * norm.cdf(d2)
            pe_premium = strike * np.exp(-r*T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(ce_premium, 5), max(pe_premium, 5)
    
    def _calculate_greeks(self, spot: float, strike: float, dte: int, vol: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        r = 0.065
        sigma = vol / 100
        T = max(dte / 365.0, 1/365)
        
        try:
            d1 = (np.log(spot/strike) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == "CE":
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            gamma = norm.pdf(d1) / (spot * sigma * np.sqrt(T))
            theta = (-spot * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * strike * np.exp(-r*T) * (norm.cdf(d2) if option_type == "CE" else norm.cdf(-d2))) / 365
            vega = spot * norm.pdf(d1) * np.sqrt(T) / 100
            
            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'theta': round(theta, 2),
                'vega': round(vega, 3)
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

class ChartPatternDetector:
    """Advanced Chart Pattern Recognition"""
    
    def __init__(self):
        self.patterns = {}
    
    def detect_cup_and_handle(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Detect Cup and Handle pattern"""
        if len(data) < 50:
            return {"detected": False}
        
        try:
            # Find local minima and maxima
            highs = argrelextrema(data['High'].values, np.greater, order=5)[0]
            lows = argrelextrema(data['Low'].values, np.less, order=5)[0]
            
            if len(highs) < 3 or len(lows) < 2:
                return {"detected": False}
            
            # Cup and Handle logic
            recent_data = data.tail(window)
            if len(recent_data) < window:
                return {"detected": False}
                
            # Check for cup formation (U-shape)
            cup_start = len(data) - window
            cup_end = len(data) - window//3
            handle_start = cup_end
            handle_end = len(data)
            
            if handle_start >= handle_end or cup_start >= cup_end:
                return {"detected": False}
                
            cup_data = data.iloc[cup_start:cup_end]
            handle_data = data.iloc[handle_start:handle_end]
            
            # Cup criteria
            cup_high_start = cup_data['High'].iloc[0]
            cup_high_end = cup_data['High'].iloc[-1]
            cup_low = cup_data['Low'].min()
            
            # Handle criteria  
            handle_high = handle_data['High'].max()
            handle_low = handle_data['Low'].min()
            
            # Pattern validation
            cup_depth = (cup_high_start - cup_low) / cup_high_start
            handle_decline = (handle_high - handle_low) / handle_high
            
            is_cup = (0.1 <= cup_depth <= 0.5 and 
                     abs(cup_high_start - cup_high_end) / cup_high_start < 0.05)
            is_handle = (handle_decline <= cup_depth / 3 and 
                        handle_high < cup_high_end * 1.05)
            
            if is_cup and is_handle:
                return {
                    "detected": True,
                    "pattern": "Cup and Handle",
                    "confidence": 0.8,
                    "target": cup_high_start * 1.1,
                    "stop_loss": handle_low * 0.98,
                    "entry_zone": handle_high * 1.01
                }
            
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    def detect_head_and_shoulders(self, data: pd.DataFrame) -> Dict:
        """Detect Head and Shoulders pattern"""
        if len(data) < 50:
            return {"detected": False}
            
        try:
            # Find peaks
            peaks = argrelextrema(data['High'].values, np.greater, order=5)[0]
            
            if len(peaks) < 5:
                return {"detected": False}
            
            # Get recent peaks for H&S pattern
            recent_peaks = peaks[-5:]
            peak_prices = data['High'].iloc[recent_peaks].values
            
            # H&S criteria: middle peak (head) higher than shoulders
            if len(peak_prices) >= 3:
                left_shoulder = peak_prices[0]
                head = peak_prices[1] 
                right_shoulder = peak_prices[2]
                
                # Pattern validation
                head_higher = head > left_shoulder and head > right_shoulder
                shoulders_similar = abs(left_shoulder - right_shoulder) / left_shoulder < 0.1
                
                if head_higher and shoulders_similar:
                    # Find neckline (lows between peaks)
                    lows_between = data.iloc[recent_peaks[0]:recent_peaks[2]]['Low']
                    neckline = lows_between.min()
                    
                    return {
                        "detected": True,
                        "pattern": "Head and Shoulders",
                        "confidence": 0.75,
                        "target": neckline - (head - neckline),
                        "stop_loss": head * 1.02,
                        "neckline": neckline
                    }
            
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    def detect_double_top(self, data: pd.DataFrame) -> Dict:
        """Detect Double Top pattern"""
        if len(data) < 30:
            return {"detected": False}
            
        try:
            peaks = argrelextrema(data['High'].values, np.greater, order=3)[0]
            
            if len(peaks) < 2:
                return {"detected": False}
            
            # Check recent peaks
            recent_peaks = peaks[-2:]
            peak_prices = data['High'].iloc[recent_peaks].values
            
            if len(peak_prices) == 2:
                peak1, peak2 = peak_prices
                
                # Double top criteria
                similar_heights = abs(peak1 - peak2) / peak1 < 0.03
                
                if similar_heights:
                    # Find valley between peaks
                    valley_data = data.iloc[recent_peaks[0]:recent_peaks[1]]
                    valley_low = valley_data['Low'].min()
                    
                    return {
                        "detected": True,
                        "pattern": "Double Top",
                        "confidence": 0.7,
                        "target": valley_low - (peak1 - valley_low) * 0.5,
                        "stop_loss": max(peak1, peak2) * 1.02,
                        "resistance": max(peak1, peak2)
                    }
            
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    def detect_all_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect all chart patterns"""
        patterns = []
        
        # Cup and Handle
        cup_handle = self.detect_cup_and_handle(data)
        if cup_handle["detected"]:
            patterns.append(cup_handle)
        
        # Head and Shoulders
        h_and_s = self.detect_head_and_shoulders(data)
        if h_and_s["detected"]:
            patterns.append(h_and_s)
        
        # Double Top
        double_top = self.detect_double_top(data)
        if double_top["detected"]:
            patterns.append(double_top)
        
        return patterns

class EnhancedTradingSystem:
    """Enhanced Trading System with Multi-timeframe Analysis"""
    
    def __init__(self):
        self.data_provider = EnhancedNSEDataProvider()
        self.pattern_detector = ChartPatternDetector()
        self.lot_sizes = {"NIFTY": 75, "BANKNIFTY": 35}
        self.signal_history = []
        self.last_signal_time = None
        
    def fetch_multi_timeframe_data(self, symbol: str, timeframes: List[str] = ["1m", "5m", "15m", "1h"]) -> Dict:
        """Fetch data for multiple timeframes"""
        ticker_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
        yf_symbol = ticker_map.get(symbol, "^NSEI")
        
        data = {}
        for tf in timeframes:
            try:
                if tf == "1m":
                    period, interval = "1d", "1m"
                elif tf == "5m":
                    period, interval = "1d", "5m"
                elif tf == "15m":
                    period, interval = "2d", "15m"
                elif tf == "1h":
                    period, interval = "5d", "1h"
                else:
                    period, interval = "1d", "1m"
                
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    data[tf] = self.add_technical_indicators(df)
                else:
                    st.warning(f"No data for {tf} timeframe")
                    
            except Exception as e:
                st.error(f"Error fetching {tf} data: {e}")
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        if data.empty:
            return data
            
        try:
            # Moving Averages
            data['EMA_9'] = ta.trend.EMAIndicator(data['Close'], 9).ema_indicator()
            data['EMA_21'] = ta.trend.EMAIndicator(data['Close'], 21).ema_indicator()
            data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], 50).sma_indicator()
            data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], 200).sma_indicator()
            
            # Momentum Indicators
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], 14).rsi()
            data['RSI_5'] = ta.momentum.RSIIndicator(data['Close'], 5).rsi()
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bb.bollinger_hband()
            data['BB_middle'] = bb.bollinger_mavg()
            data['BB_lower'] = bb.bollinger_lband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
            data['Stoch_K'] = stoch.stoch()
            data['Stoch_D'] = stoch.stoch_signal()
            
            # ATR
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            
            # VWAP
            if 'Volume' in data.columns:
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            
            # Volume indicators
            if 'Volume' in data.columns:
                data['Volume_SMA'] = data['Volume'].rolling(20).mean()
            
            return data
            
        except Exception as e:
            st.error(f"Indicator calculation error: {e}")
            return data
    
    def generate_scalping_signals(self, data_1m: pd.DataFrame, data_5m: pd.DataFrame) -> Dict:
        """Generate scalping signals for 1-5 minute timeframes"""
        if data_1m.empty or data_5m.empty:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        try:
            # Current values from 1m timeframe
            latest_1m = data_1m.iloc[-1]
            latest_5m = data_5m.iloc[-1]
            
            signals = []
            score = 0
            
            # 1. RSI Multi-timeframe Analysis
            rsi_1m = latest_1m['RSI']
            rsi_5m = latest_5m['RSI']
            
            # Scalping RSI levels (more sensitive)
            if rsi_1m < 25 and rsi_5m < 35:
                signals.append("RSI oversold on multiple timeframes")
                score += 2
            elif rsi_1m > 75 and rsi_5m > 65:
                signals.append("RSI overbought on multiple timeframes") 
                score -= 2
            elif rsi_1m > 50 and rsi_5m > 50:
                signals.append("RSI bullish bias")
                score += 1
            elif rsi_1m < 50 and rsi_5m < 50:
                signals.append("RSI bearish bias")
                score -= 1
            
            # 2. MACD Quick Signals
            if latest_1m['MACD'] > latest_1m['MACD_signal'] and data_1m['MACD'].iloc[-2] <= data_1m['MACD_signal'].iloc[-2]:
                signals.append("MACD bullish crossover (1m)")
                score += 2
            elif latest_1m['MACD'] < latest_1m['MACD_signal'] and data_1m['MACD'].iloc[-2] >= data_1m['MACD_signal'].iloc[-2]:
                signals.append("MACD bearish crossover (1m)")
                score -= 2
            
            # 3. Bollinger Band Scalping
            close_1m = latest_1m['Close']
            bb_position = (close_1m - latest_1m['BB_lower']) / (latest_1m['BB_upper'] - latest_1m['BB_lower'])
            
            if bb_position < 0.1:
                signals.append("Price near lower Bollinger Band")
                score += 1.5
            elif bb_position > 0.9:
                signals.append("Price near upper Bollinger Band")
                score -= 1.5
            
            # 4. Stochastic Quick Momentum
            stoch_k = latest_1m['Stoch_K']
            stoch_d = latest_1m['Stoch_D']
            
            if stoch_k < 20 and stoch_k > stoch_d:
                signals.append("Stochastic oversold reversal")
                score += 1.5
            elif stoch_k > 80 and stoch_k < stoch_d:
                signals.append("Stochastic overbought reversal")
                score -= 1.5
            
            # 5. Volume Confirmation
            if 'Volume' in data_1m.columns:
                volume_ratio = latest_1m['Volume'] / latest_1m['Volume_SMA'] if latest_1m['Volume_SMA'] > 0 else 1
                if volume_ratio > 1.5:
                    if score > 0:
                        signals.append("High volume supports bullish move")
                        score += 0.5
                    else:
                        signals.append("High volume supports bearish move")
                        score -= 0.5
            
            # 6. EMA Trend Alignment
            if latest_1m['Close'] > latest_1m['EMA_9'] > latest_1m['EMA_21']:
                signals.append("Bullish EMA alignment")
                score += 1
            elif latest_1m['Close'] < latest_1m['EMA_9'] < latest_1m['EMA_21']:
                signals.append("Bearish EMA alignment")
                score -= 1
            
            # Generate final signal
            confidence = min(abs(score) * 15, 95)
            
            if score >= 3:
                signal = "STRONG BUY CE"
            elif score >= 2:
                signal = "BUY CE"
            elif score <= -3:
                signal = "STRONG BUY PE"
            elif score <= -2:
                signal = "BUY PE"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "confidence": confidence,
                "score": score,
                "signals": signals[:6],
                "timeframe": "Scalping (1m-5m)",
                "current_price": close_1m,
                "rsi_1m": rsi_1m,
                "rsi_5m": rsi_5m
            }
            
        except Exception as e:
            return {"signal": "HOLD", "confidence": 0, "reason": f"Error: {str(e)}"}
    
    def generate_intraday_signals(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame, data_1h: pd.DataFrame) -> Dict:
        """Generate intraday signals for 5m-1h timeframes"""
        if data_5m.empty or data_15m.empty:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        try:
            latest_5m = data_5m.iloc[-1]
            latest_15m = data_15m.iloc[-1]
            latest_1h = data_1h.iloc[-1] if not data_1h.empty else latest_15m
            
            signals = []
            score = 0
            
            # 1. Multi-timeframe Trend Analysis
            trend_5m = "UP" if latest_5m['EMA_9'] > latest_5m['EMA_21'] else "DOWN"
            trend_15m = "UP" if latest_15m['EMA_9'] > latest_15m['EMA_21'] else "DOWN"
            trend_1h = "UP" if latest_1h['EMA_9'] > latest_1h['EMA_21'] else "DOWN"
            
            aligned_trends = sum([1 for trend in [trend_5m, trend_15m, trend_1h] if trend == "UP"])
            
            if aligned_trends >= 2:
                signals.append(f"Multi-timeframe bullish alignment ({aligned_trends}/3)")
                score += aligned_trends
            elif aligned_trends <= 1:
                signals.append(f"Multi-timeframe bearish alignment ({3-aligned_trends}/3)")
                score -= (3 - aligned_trends)
            
            # 2. RSI Confluence
            rsi_5m = latest_5m['RSI']
            rsi_15m = latest_15m['RSI']
            rsi_1h = latest_1h['RSI']
            
            oversold_count = sum([1 for rsi in [rsi_5m, rsi_15m, rsi_1h] if rsi < 35])
            overbought_count = sum([1 for rsi in [rsi_5m, rsi_15m, rsi_1h] if rsi > 65])
            
            if oversold_count >= 2:
                signals.append("Multi-timeframe RSI oversold")
                score += 2
            elif overbought_count >= 2:
                signals.append("Multi-timeframe RSI overbought")
                score -= 2
            
            # 3. MACD Strength Analysis
            macd_bull_15m = latest_15m['MACD'] > latest_15m['MACD_signal']
            macd_bull_1h = latest_1h['MACD'] > latest_1h['MACD_signal']
            
            if macd_bull_15m and macd_bull_1h:
                signals.append("MACD bullish on higher timeframes")
                score += 1.5
            elif not macd_bull_15m and not macd_bull_1h:
                signals.append("MACD bearish on higher timeframes")
                score -= 1.5
            
            # 4. Support/Resistance Analysis
            close = latest_5m['Close']
            sma_50 = latest_15m['SMA_50']
            sma_200 = latest_1h['SMA_200']
            
            if close > sma_50 > sma_200:
                signals.append("Price above major moving averages")
                score += 1
            elif close < sma_50 < sma_200:
                signals.append("Price below major moving averages")
                score -= 1
            
            # 5. Bollinger Band Analysis
            bb_squeeze = (latest_15m['BB_upper'] - latest_15m['BB_lower']) / latest_15m['BB_middle'] < 0.1
            if bb_squeeze:
                signals.append("Bollinger Band squeeze - breakout expected")
                # Don't add to score, just flag for attention
            
            # Generate final signal
            confidence = min(abs(score) * 12, 90)
            
            if score >= 4:
                signal = "STRONG BUY CE"
            elif score >= 2.5:
                signal = "BUY CE"
            elif score <= -4:
                signal = "STRONG BUY PE"
            elif score <= -2.5:
                signal = "BUY PE"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "confidence": confidence,
                "score": score,
                "signals": signals[:6],
                "timeframe": "Intraday (5m-1h)",
                "current_price": close,
                "trend_alignment": f"{aligned_trends}/3",
                "rsi_levels": f"5m:{rsi_5m:.1f} 15m:{rsi_15m:.1f} 1h:{rsi_1h:.1f}"
            }
            
        except Exception as e:
            return {"signal": "HOLD", "confidence": 0, "reason": f"Error: {str(e)}"}
    
    def validate_signal_consistency(self, new_signal: Dict) -> Dict:
        """Validate signal consistency to prevent rapid changes"""
        current_time = datetime.now()
        
        # If it's the first signal or enough time has passed
        if (not self.signal_history or 
            not self.last_signal_time or 
            (current_time - self.last_signal_time).seconds > 60):
            
            self.signal_history.append(new_signal)
            self.last_signal_time = current_time
            
            # Keep only last 5 signals
            if len(self.signal_history) > 5:
                self.signal_history.pop(0)
                
            return new_signal
        
        # Check consistency with recent signals
        if len(self.signal_history) >= 2:
            recent_signals = [s['signal'] for s in self.signal_history[-2:]]
            
            # If new signal is very different, reduce confidence
            if new_signal['signal'] != recent_signals[-1]:
                signal_types = set(recent_signals + [new_signal['signal']])
                if len(signal_types) > 2:  # Too much variation
                    new_signal['confidence'] *= 0.7
                    new_signal['signals'].insert(0, "⚠️ Signal consistency warning")
        
        self.signal_history.append(new_signal)
        self.last_signal_time = current_time
        
        return new_signal
    
    def calculate_position_sizing(self, signal: Dict, account_size: float, risk_percent: float = 2) -> Dict:
        """Calculate position sizing based on risk management"""
        try:
            risk_amount = account_size * (risk_percent / 100)
            current_price = signal.get('current_price', 25000)
            
            # Estimate option premium (simplified)
            option_premium = current_price * 0.02  # Rough estimate
            
            # Calculate quantity
            max_quantity = int(risk_amount / option_premium)
            
            # Consider lot sizes
            symbol_type = "NIFTY"  # Default
            lot_size = self.lot_sizes[symbol_type]
            
            lots = max(1, max_quantity // lot_size)
            quantity = lots * lot_size
            
            return {
                "lots": lots,
                "quantity": quantity,
                "risk_amount": risk_amount,
                "estimated_premium": option_premium,
                "position_value": quantity * option_premium
            }
            
        except Exception as e:
            return {"error": f"Position sizing error: {str(e)}"}

# Streamlit App
def main():
    st.title("🚀 Enhanced NSE Options Trading System")
    st.markdown("*Real-time Analysis • Pattern Recognition • Multi-timeframe Signals*")
    
    # Initialize system
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = EnhancedTradingSystem()
    
    trading_system = st.session_state.trading_system
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY"],
        index=0
    )
    
    # Trading style
    trading_style = st.sidebar.selectbox(
        "Trading Style",
        ["Scalping (1m-5m)", "Intraday (5m-1h)", "Both"],
        index=0
    )
    
    # Risk management
    account_size = st.sidebar.number_input("Account Size (₹)", value=100000, step=10000)
    risk_percent = st.sidebar.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    if auto_refresh:
        st.rerun()
        time.sleep(30)
    
    # Main Analysis
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header(f"📊 {symbol} Enhanced Analysis")
            
            # Fetch multi-timeframe data
            with st.spinner("Fetching multi-timeframe data..."):
                timeframes = ["1m", "5m", "15m", "1h"]
                mtf_data = trading_system.fetch_multi_timeframe_data(symbol, timeframes)
            
            if not mtf_data:
                st.error("❌ Unable to fetch market data. Please try again.")
                return
            
            # Current price display
            if "1m" in mtf_data and not mtf_data["1m"].empty:
                current_data = mtf_data["1m"].iloc[-1]
                current_price = current_data['Close']
                
                # Price change calculation
                if len(mtf_data["1m"]) > 1:
                    prev_price = mtf_data["1m"].iloc[-2]['Close']
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100
                else:
                    price_change = 0
                    price_change_pct = 0
                
                # Display current price
                col1_1, col1_2, col1_3, col1_4 = st.columns(4)
                
                with col1_1:
                    st.metric(
                        "Current Price",
                        f"₹{current_price:.2f}",
                        f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )
                
                with col1_2:
                    rsi_1m = current_data.get('RSI', 50)
                    st.metric("RSI (1m)", f"{rsi_1m:.1f}")
                
                with col1_3:
                    if '5m' in mtf_data:
                        rsi_5m = mtf_data['5m'].iloc[-1].get('RSI', 50)
                        st.metric("RSI (5m)", f"{rsi_5m:.1f}")
                
                with col1_4:
                    if 'Volume' in current_data:
                        vol_ratio = current_data['Volume'] / current_data.get('Volume_SMA', current_data['Volume'])
                        st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
            
            # Generate signals based on trading style
            if trading_style == "Scalping (1m-5m)" or trading_style == "Both":
                st.subheader("🔥 Scalping Signals")
                
                if "1m" in mtf_data and "5m" in mtf_data:
                    scalping_signal = trading_system.generate_scalping_signals(
                        mtf_data["1m"], mtf_data["5m"]
                    )
                    scalping_signal = trading_system.validate_signal_consistency(scalping_signal)
                    
                    # Display scalping signal
                    signal_class = f"signal-{scalping_signal['signal'].lower().replace(' ', '-')}"
                    st.markdown(f"""
                    <div class="{signal_class}">
                        🎯 {scalping_signal['signal']} - {scalping_signal['confidence']:.0f}% Confidence
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Signal details
                    st.write("**Signal Analysis:**")
                    for signal in scalping_signal.get('signals', []):
                        st.write(f"• {signal}")
            
            if trading_style == "Intraday (5m-1h)" or trading_style == "Both":
                st.subheader("📈 Intraday Signals")
                
                if "5m" in mtf_data and "15m" in mtf_data:
                    intraday_signal = trading_system.generate_intraday_signals(
                        mtf_data["5m"], mtf_data["15m"], mtf_data.get("1h", pd.DataFrame())
                    )
                    
                    # Display intraday signal
                    signal_class = f"signal-{intraday_signal['signal'].lower().replace(' ', '-')}"
                    st.markdown(f"""
                    <div class="{signal_class}">
                        📊 {intraday_signal['signal']} - {intraday_signal['confidence']:.0f}% Confidence
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Signal details
                    st.write("**Signal Analysis:**")
                    for signal in intraday_signal.get('signals', []):
                        st.write(f"• {signal}")
            
            # Chart Pattern Detection
            st.subheader("🔍 Chart Pattern Analysis")
            
            if "15m" in mtf_data:
                patterns = trading_system.pattern_detector.detect_all_patterns(mtf_data["15m"])
                
                if patterns:
                    for pattern in patterns:
                        st.markdown(f"""
                        <div class="pattern-detected">
                            🎭 {pattern['pattern']} Detected! 
                            Confidence: {pattern['confidence']*100:.0f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Pattern details
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            if 'target' in pattern:
                                st.metric("Target", f"₹{pattern['target']:.0f}")
                        with col_p2:
                            if 'stop_loss' in pattern:
                                st.metric("Stop Loss", f"₹{pattern['stop_loss']:.0f}")
                        with col_p3:
                            if 'entry_zone' in pattern:
                                st.metric("Entry Zone", f"₹{pattern['entry_zone']:.0f}")
                else:
                    st.info("No significant chart patterns detected at this time.")
        
        with col2:
            st.header("📋 Trading Panel")
            
            # Live option chain
            st.subheader("📊 Live Option Chain")
            with st.spinner("Fetching option chain..."):
                option_chain = trading_system.data_provider.get_live_option_chain(symbol)
            
            if option_chain is not None and not option_chain.empty:
                # Display key strikes
                current_price = mtf_data.get("1m", pd.DataFrame()).iloc[-1]['Close'] if "1m" in mtf_data else 25000
                
                # Find ATM strikes
                atm_strikes = option_chain[
                    (option_chain['strike'] >= current_price - 200) & 
                    (option_chain['strike'] <= current_price + 200)
                ].head(5)
                
                st.write("**Near ATM Strikes:**")
                for _, row in atm_strikes.iterrows():
                    strike = row['strike']
                    ce_ltp = row['ce_ltp']
                    pe_ltp = row['pe_ltp']
                    ce_oi = row['ce_oi']
                    pe_oi = row['pe_oi']
                    
                    st.write(f"**{strike}:** CE ₹{ce_ltp:.1f} (OI: {ce_oi:,}) | PE ₹{pe_ltp:.1f} (OI: {pe_oi:,})")
                
                # PCR Analysis
                total_ce_oi = option_chain['ce_oi'].sum()
                total_pe_oi = option_chain['pe_oi'].sum()
                pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1
                
                st.metric("Put-Call Ratio", f"{pcr:.2f}")
                
                # PCR interpretation
                if pcr > 1.3:
                    st.warning("🐻 High PCR - Bearish sentiment")
                elif pcr < 0.7:
                    st.warning("🐂 Low PCR - Bullish sentiment")
                else:
                    st.info("😐 Neutral PCR")
            
            # Position sizing
            st.subheader("💰 Position Sizing")
            
            if 'scalping_signal' in locals():
                position_info = trading_system.calculate_position_sizing(
                    scalping_signal, account_size, risk_percent
                )
                
                if 'error' not in position_info:
                    st.write(f"**Suggested Lots:** {position_info['lots']}")
                    st.write(f"**Quantity:** {position_info['quantity']}")
                    st.write(f"**Risk Amount:** ₹{position_info['risk_amount']:,.0f}")
                    st.write(f"**Est. Position Value:** ₹{position_info['position_value']:,.0f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** This tool is for educational purposes only. 
    Always conduct your own research and consult with financial advisors before trading.
    """)

if __name__ == "__main__":
    main()