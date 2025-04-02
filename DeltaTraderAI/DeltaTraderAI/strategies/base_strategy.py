from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from api.delta_exchange import DeltaExchangeAPI
from indicators.technical_indicators import TechnicalIndicators
from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    
    Provides common functionality and defines the interface that
    all strategy implementations must follow.
    """
    
    def __init__(self, symbol, timeframe, parameters=None, api_client=None, account_balance=10000.0):
        """
        Initialize the strategy
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USDT')
            timeframe (str): Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            parameters (dict): Strategy-specific parameters
            api_client (DeltaExchangeAPI): API client instance
            account_balance (float): Initial account balance for risk management
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.parameters = parameters or {}
        self.api_client = api_client or DeltaExchangeAPI()
        self.indicators = TechnicalIndicators()
        self.historical_data = None
        
        # Initialize risk manager
        self.risk_manager = RiskManager(account_balance=account_balance)
        
        # Performance tracking
        self.trades = []
        self.performance_metrics = {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
        }
        
        # Position state
        self.position = {
            'is_open': False,
            'direction': None,  # 'long' or 'short'
            'entry_price': 0,
            'size': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'entry_time': None
        }
        
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} on {timeframe}")
    
    def fetch_historical_data(self, limit=100):
        """
        Fetch historical candle data from the exchange
        
        Args:
            limit (int): Number of candles to fetch
            
        Returns:
            DataFrame: OHLCV data with additional indicators
        """
        try:
            candles_data = self.api_client.get_candles(
                symbol=self.symbol,
                resolution=self.timeframe,
                limit=limit
            )
            
            if not candles_data.get('result'):
                logger.error(f"Failed to fetch historical data for {self.symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(candles_data['result'])
            
            # Rename columns to standard OHLCV names
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            self.historical_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def update_historical_data(self):
        """Update historical data with the latest candles"""
        if self.historical_data is None:
            return self.fetch_historical_data()
            
        # Get the latest candle
        last_timestamp = self.historical_data.index[-1]
        
        try:
            # Fetch new candles since the last timestamp
            new_candles = self.api_client.get_candles(
                symbol=self.symbol,
                resolution=self.timeframe,
                start_time=int(last_timestamp.timestamp() * 1000) + 1
            )
            
            if not new_candles.get('result') or len(new_candles['result']) == 0:
                logger.debug("No new candles available")
                return self.historical_data
                
            # Convert new candles to DataFrame
            new_df = pd.DataFrame(new_candles['result'])
            
            # Rename columns
            new_df = new_df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert timestamp to datetime
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            
            # Sort by timestamp
            new_df = new_df.sort_values('timestamp')
            
            # Set timestamp as index
            new_df.set_index('timestamp', inplace=True)
            
            # Append new candles
            self.historical_data = pd.concat([self.historical_data, new_df])
            
            return self.historical_data
            
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            return self.historical_data
    
    def calculate_indicators(self, data=None):
        """
        Calculate technical indicators for the strategy
        
        Args:
            data (DataFrame, optional): OHLCV data to use, uses historical_data if None
            
        Returns:
            DataFrame: Data with indicators added
        """
        if data is None:
            if self.historical_data is None:
                self.fetch_historical_data()
            data = self.historical_data
            
        # Child classes will implement specific indicators
        return data
    
    @abstractmethod
    def generate_signal(self, data=None):
        """
        Generate trading signal based on strategy logic
        
        Args:
            data (DataFrame, optional): Data with indicators
            
        Returns:
            str: Signal ('buy', 'sell', or 'hold')
        """
        pass
    
    def execute_signal(self, signal, price=None):
        """
        Execute a trading signal with risk management
        
        Args:
            signal (str): Trading signal ('buy', 'sell', or 'hold')
            price (float, optional): Current price, if None will fetch from API
            
        Returns:
            dict: Trade details if executed, None otherwise
        """
        # Don't trade if we have a hold signal
        if signal == 'hold':
            return None
            
        # Check risk rules before proceeding
        if not self.is_trading_allowed():
            logger.warning(f"Trading not allowed due to risk limits: {self.risk_manager.pause_reason}")
            return None
            
        # Fetch current price if not provided
        if price is None:
            # Get current price from ticker
            ticker_data = self.api_client.get_tickers(self.symbol)
            if not ticker_data.get('result'):
                logger.error("Failed to fetch current price")
                return None
                
            price = float(ticker_data['result'][0]['mark_price'])
        
        # Check for stop loss/take profit if we have an open position
        if self.position['is_open']:
            # Update trailing stop if in profit
            self.adjust_trailing_stop(price)
            
            # Close position if stop loss or take profit hit
            trade_result = self.check_stop_loss_take_profit(price, adjust_trailing_stop=False)
            if trade_result is not None:
                # Position was closed due to stop loss or take profit
                logger.info("Position closed due to stop loss or take profit")
                
                # Reset and prepare for new position if signal allows
                if (signal == 'buy' and trade_result['direction'] == 'short') or \
                   (signal == 'sell' and trade_result['direction'] == 'long'):
                    pass  # Position was closed, we can open a new one below
                else:
                    return trade_result  # Return the closed trade info
            
            # If we still have an open position, check if signal is opposite direction
            if self.position['is_open']:
                if (signal == 'buy' and self.position['direction'] == 'short') or \
                   (signal == 'sell' and self.position['direction'] == 'long'):
                    # Close the current position due to signal
                    logger.info(f"Closing position due to opposite signal: {signal}")
                    self._close_position(price)
                else:
                    # Same direction signal, do nothing but check trailing stop
                    self.adjust_trailing_stop(price)
                    return None
        
        # If we reached here, we can open a new position if risk rules allow
        if signal == 'buy':
            return self._open_position('long', price)
        elif signal == 'sell':
            return self._open_position('short', price)
            
        return None
    
    def _open_position(self, direction, price):
        """
        Open a new trading position with risk management
        
        Args:
            direction (str): 'long' or 'short'
            price (float): Entry price
            
        Returns:
            dict: Position details or None if position can't be opened
        """
        # Check if we can open a position based on risk rules
        if not self.risk_manager.can_open_position(self.symbol, price, direction):
            logger.warning(f"Risk manager prevented opening {direction} position at {price}")
            return None
        
        # Get risk parameters
        stop_loss_pct = self.parameters.get('stop_loss', 2.0) / 100
        take_profit_pct = self.parameters.get('take_profit', 3.0) / 100
        
        # Calculate stop loss and take profit levels
        if direction == 'long':
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else:  # short
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)
        
        # Calculate volatility if we have historical data
        volatility = None
        if self.historical_data is not None and len(self.historical_data) > 20:
            volatility = self.risk_manager.calculate_market_volatility(self.historical_data['close'])
            logger.info(f"Market volatility for {self.symbol}: {volatility:.2f}%")
            
        # Calculate position size based on risk parameters
        size = self.risk_manager.calculate_position_size(
            symbol=self.symbol,
            entry_price=price,
            stop_loss_price=stop_loss,
            volatility=volatility
        )
        
        # Don't open position if size is zero (risk limits reached)
        if size <= 0:
            logger.warning("Position size is zero, skipping trade")
            return None
        
        # Update position state
        self.position = {
            'is_open': True,
            'direction': direction,
            'entry_price': price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': pd.Timestamp.now()
        }
        
        logger.info(f"Opened {direction} position: {size:.4f} units @ {price} with stop loss {stop_loss:.2f} and take profit {take_profit:.2f}")
        
        # In a real trading scenario, this would place an order via the API
        # self.api_client.place_order(...)
        
        return self.position
    
    def _close_position(self, price):
        """
        Close an existing trading position with risk management
        
        Args:
            price (float): Exit price
            
        Returns:
            dict: Trade details
        """
        if not self.position['is_open']:
            return None
            
        # Calculate profit/loss
        if self.position['direction'] == 'long':
            pnl = (price - self.position['entry_price']) * self.position['size']
            pnl_percentage = (price - self.position['entry_price']) / self.position['entry_price'] * 100
        else:  # short
            pnl = (self.position['entry_price'] - price) * self.position['size']
            pnl_percentage = (self.position['entry_price'] - price) / self.position['entry_price'] * 100
        
        # Create trade record
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': pd.Timestamp.now(),
            'direction': self.position['direction'],
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'size': self.position['size'],
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'symbol': self.symbol,
            'strategy': self.__class__.__name__
        }
        
        # Add to trade history
        self.trades.append(trade)
        
        # Record trade with risk manager for daily PnL tracking
        self.risk_manager.record_trade(trade)
        
        # Update account balance in risk manager (in a real system, this would come from the exchange)
        self.risk_manager.update_account_balance(self.risk_manager.account_balance + pnl)
        
        # Reset position
        self.position = {
            'is_open': False,
            'direction': None,
            'entry_price': 0,
            'size': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'entry_time': None
        }
        
        logger.info(f"Closed position at {price} with PnL: {pnl:.2f} ({pnl_percentage:.2f}%)")
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Log risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        logger.info(
            f"Risk metrics after trade - Daily PnL: ${risk_metrics['daily_pnl']:.2f}, " 
            f"Drawdown: {risk_metrics['drawdown']:.2f}%, "
            f"Trades today: {risk_metrics['trade_count_today']}"
        )
        
        return trade
    
    def check_stop_loss_take_profit(self, current_price, adjust_trailing_stop=True):
        """
        Check if stop loss or take profit levels are hit and adjust trailing stop if profitable
        
        Args:
            current_price (float): Current market price
            adjust_trailing_stop (bool): Whether to adjust trailing stop loss
            
        Returns:
            dict: Trade details if position is closed, None otherwise
        """
        if not self.position['is_open']:
            return None
            
        # First, check if we need to adjust trailing stop
        if adjust_trailing_stop:
            self.adjust_trailing_stop(current_price)
            
        # Then check if stop loss or take profit is hit
        if self.position['direction'] == 'long':
            if current_price <= self.position['stop_loss']:
                logger.info(f"Stop loss triggered at {current_price}")
                return self._close_position(current_price)
                
            if current_price >= self.position['take_profit']:
                logger.info(f"Take profit triggered at {current_price}")
                return self._close_position(current_price)
                
        else:  # short
            if current_price >= self.position['stop_loss']:
                logger.info(f"Stop loss triggered at {current_price}")
                return self._close_position(current_price)
                
            if current_price <= self.position['take_profit']:
                logger.info(f"Take profit triggered at {current_price}")
                return self._close_position(current_price)
                
        return None
        
    def adjust_trailing_stop(self, current_price, trail_percent=0.5):
        """
        Adjust stop loss using trailing stop technique
        
        Args:
            current_price (float): Current market price
            trail_percent (float): Percentage of profit to protect (0.5 = 50%)
            
        Returns:
            bool: True if stop loss was adjusted, False otherwise
        """
        if not self.position['is_open']:
            return False
            
        # Use the risk manager's trailing stop logic
        new_stop = self.risk_manager.adjust_stop_loss(
            position=self.position,
            current_price=current_price,
            adjust_pct=trail_percent
        )
        
        if new_stop is not None:
            # Update the stop loss in our position
            self.position['stop_loss'] = new_stop
            logger.info(f"Adjusted trailing stop to {new_stop:.2f}")
            return True
            
        return False
    
    def _update_performance_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trades:
            return
            
        # Calculate win rate
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Calculate profit factor
        total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average profit and loss
        avg_profit = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / (len(self.trades) - len(winning_trades)) if len(self.trades) > len(winning_trades) else 0
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum([t['pnl'] for t in self.trades])
        max_drawdown = 0
        peak = 0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['pnl_percentage'] for t in self.trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Update metrics
        self.performance_metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_performance_metrics(self):
        """Get strategy performance metrics"""
        return self.performance_metrics
    
    def get_trade_history(self):
        """Get trade history"""
        return self.trades
        
    def get_risk_metrics(self):
        """
        Get current risk management metrics
        
        Returns:
            dict: Risk metrics from the risk manager
        """
        return self.risk_manager.get_risk_metrics()
        
    def is_trading_allowed(self):
        """
        Check if trading is currently allowed based on risk rules
        
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        # Check if trading is paused
        if self.risk_manager.trading_paused:
            logger.warning(f"Trading is paused: {self.risk_manager.pause_reason}")
            return False
            
        # Check daily loss limit
        if not self.risk_manager.check_daily_loss_limit():
            return False
            
        # Check trade frequency limit
        if not self.risk_manager.check_trade_frequency():
            return False
            
        return True
