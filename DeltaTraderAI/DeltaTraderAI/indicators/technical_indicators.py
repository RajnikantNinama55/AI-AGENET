import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical indicators calculator for trading strategies
    
    Provides methods to calculate various technical indicators like SMA, RSI, ATR, etc.
    """
    
    def add_sma(self, data, period, col_name=None):
        """
        Add Simple Moving Average to the dataframe
        
        Args:
            data: OHLCV DataFrame
            period: SMA period
            col_name: Column name for the indicator
            
        Returns:
            DataFrame with SMA indicator
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_name = col_name or f'sma_{period}'
        
        try:
            data[col_name] = data['close'].rolling(window=period).mean()
            return data
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return data
    
    def add_ema(self, data, period, col_name=None):
        """
        Add Exponential Moving Average to the dataframe
        
        Args:
            data: OHLCV DataFrame
            period: EMA period
            col_name: Column name for the indicator
            
        Returns:
            DataFrame with EMA indicator
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_name = col_name or f'ema_{period}'
        
        try:
            data[col_name] = data['close'].ewm(span=period, adjust=False).mean()
            return data
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return data
    
    def add_rsi(self, data, period=14, col_name=None):
        """
        Add Relative Strength Index to the dataframe
        
        Args:
            data: OHLCV DataFrame
            period: RSI period
            col_name: Column name for the indicator
            
        Returns:
            DataFrame with RSI indicator
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_name = col_name or f'rsi_{period}'
        
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            data[col_name] = 100 - (100 / (1 + rs))
            return data
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return data
    
    def add_atr(self, data, period=14, col_name=None):
        """
        Add Average True Range to the dataframe
        
        Args:
            data: OHLCV DataFrame
            period: ATR period
            col_name: Column name for the indicator
            
        Returns:
            DataFrame with ATR indicator
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_name = col_name or f'atr_{period}'
        
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            data[col_name] = true_range.rolling(window=period).mean()
            return data
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return data
    
    def add_bollinger_bands(self, data, period=20, std_dev=2, col_prefix=None):
        """
        Add Bollinger Bands to the dataframe
        
        Args:
            data: OHLCV DataFrame
            period: Bollinger Band period
            std_dev: Standard deviation multiplier
            col_prefix: Column prefix for the indicators
            
        Returns:
            DataFrame with Bollinger Band indicators
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_prefix = col_prefix or 'bb'
        
        try:
            # Calculate middle band (SMA)
            middle_band_col = f'{col_prefix}_middle'
            data[middle_band_col] = data['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = data['close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band_col = f'{col_prefix}_upper'
            lower_band_col = f'{col_prefix}_lower'
            
            data[upper_band_col] = data[middle_band_col] + (std_dev * std)
            data[lower_band_col] = data[middle_band_col] - (std_dev * std)
            
            return data
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return data
    
    def add_macd(self, data, fast_period=12, slow_period=26, signal_period=9, col_prefix=None):
        """
        Add MACD to the dataframe
        
        Args:
            data: OHLCV DataFrame
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            col_prefix: Column prefix for the indicators
            
        Returns:
            DataFrame with MACD indicators
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_prefix = col_prefix or 'macd'
        
        try:
            # Calculate fast and slow EMAs
            fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_col = f'{col_prefix}_line'
            data[macd_col] = fast_ema - slow_ema
            
            # Calculate signal line
            signal_col = f'{col_prefix}_signal'
            data[signal_col] = data[macd_col].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            hist_col = f'{col_prefix}_hist'
            data[hist_col] = data[macd_col] - data[signal_col]
            
            return data
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return data
    
    def add_roc(self, data, period=14, col_name=None):
        """
        Add Rate of Change to the dataframe
        
        Args:
            data: OHLCV DataFrame
            period: ROC period
            col_name: Column name for the indicator
            
        Returns:
            DataFrame with ROC indicator
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_name = col_name or f'roc_{period}'
        
        try:
            data[col_name] = data['close'].pct_change(periods=period) * 100
            return data
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
            return data
    
    def add_stochastic(self, data, k_period=14, d_period=3, col_prefix=None):
        """
        Add Stochastic Oscillator to the dataframe
        
        Args:
            data: OHLCV DataFrame
            k_period: %K period
            d_period: %D period
            col_prefix: Column prefix for the indicators
            
        Returns:
            DataFrame with Stochastic indicators
        """
        if data is None or len(data) == 0:
            logger.error("Empty dataframe provided")
            return data
            
        col_prefix = col_prefix or 'stoch'
        
        try:
            # Calculate %K
            low_min = data['low'].rolling(window=k_period).min()
            high_max = data['high'].rolling(window=k_period).max()
            
            k_col = f'{col_prefix}_k'
            data[k_col] = 100 * ((data['close'] - low_min) / (high_max - low_min))
            
            # Calculate %D (SMA of %K)
            d_col = f'{col_prefix}_d'
            data[d_col] = data[k_col].rolling(window=d_period).mean()
            
            return data
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {e}")
            return data
