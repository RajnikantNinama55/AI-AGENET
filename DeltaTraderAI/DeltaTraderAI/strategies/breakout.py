import logging
import pandas as pd
from strategies.base_strategy import BaseStrategy
from config import Config

logger = logging.getLogger(__name__)

class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy using period high/lows and ATR
    
    This strategy goes long when price breaks above the highest high of the last N periods,
    and goes short when price breaks below the lowest low of the last N periods.
    """
    
    def __init__(self, symbol, timeframe, parameters=None, api_client=None, account_balance=10000.0):
        """
        Initialize the breakout strategy
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candle timeframe
            parameters (dict, optional): Strategy parameters
            api_client: API client instance
            account_balance (float): Initial account balance for risk management
        """
        # Set default parameters if none provided
        if parameters is None:
            parameters = Config.BREAKOUT_DEFAULTS.copy()
            
        super().__init__(symbol, timeframe, parameters, api_client, account_balance)
        
        logger.info(f"Initialized Breakout Strategy with parameters: {parameters}")
    
    def calculate_indicators(self, data=None):
        """
        Calculate indicators for breakout strategy
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        data = super().calculate_indicators(data)
        
        if data is None or len(data) == 0:
            logger.error("No data available for calculating indicators")
            return None
        
        # Extract parameters
        period = self.parameters.get('period', 20)
        atr_period = self.parameters.get('atr_period', 14)
        
        # Calculate ATR
        data = self.indicators.add_atr(data, atr_period, 'atr')
        
        # Calculate rolling highest high and lowest low
        data['highest_high'] = data['high'].rolling(window=period).max()
        data['lowest_low'] = data['low'].rolling(window=period).min()
        
        return data
    
    def generate_signal(self, data=None):
        """
        Generate trading signal based on breakout levels
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            str: Signal ('buy', 'sell', or 'hold')
        """
        if data is None:
            data = self.calculate_indicators()
            
        if data is None or len(data) < 2:
            logger.warning("Insufficient data for signal generation")
            return 'hold'
        
        # Get the last two rows for breakout detection
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        # Check for NaN values in the indicators
        if (last_row['highest_high'] is None or last_row['lowest_low'] is None or 
            prev_row['highest_high'] is None or prev_row['lowest_low'] is None or
            last_row['atr'] is None):
            logger.warning("Missing indicator values, unable to generate signal")
            return 'hold'
        
        # Extract parameters
        atr_multiplier = self.parameters.get('atr_multiplier', 2.0)
        
        # Calculate breakout levels
        high_breakout = prev_row['highest_high']
        low_breakout = prev_row['lowest_low']
        
        # Add ATR buffer to breakout levels to reduce false signals
        high_breakout += last_row['atr'] * atr_multiplier
        low_breakout -= last_row['atr'] * atr_multiplier
        
        # Bullish breakout
        if last_row['close'] > high_breakout:
            return 'buy'
        
        # Bearish breakout
        elif last_row['close'] < low_breakout:
            return 'sell'
        
        # No breakout
        return 'hold'
    
    def backtest(self, data=None, initial_capital=10000):
        """
        Backtest the strategy on historical data
        
        Args:
            data: Historical OHLCV data
            initial_capital: Initial capital for the backtest
            
        Returns:
            dict: Backtest results
        """
        if data is None:
            data = self.fetch_historical_data(limit=500)  # Get more data for backtest
            
        if data is None or len(data) == 0:
            logger.error("No data available for backtesting")
            return None
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Initialize backtest variables
        capital = initial_capital
        position = None
        position_size = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Extract parameters
        atr_multiplier = self.parameters.get('atr_multiplier', 2.0)
        
        # Generate signals for each candle
        for i in range(max(21, self.parameters.get('period', 20) + 1), len(data)):
            prev_row = data.iloc[i-1]
            current_row = data.iloc[i]
            
            # Skip rows with missing indicator values
            if (current_row['highest_high'] is None or current_row['lowest_low'] is None or 
                prev_row['highest_high'] is None or prev_row['lowest_low'] is None or
                current_row['atr'] is None):
                continue
            
            # Get close price
            close_price = current_row['close']
            
            # Calculate breakout levels based on previous candle
            high_breakout = prev_row['highest_high'] + prev_row['atr'] * atr_multiplier
            low_breakout = prev_row['lowest_low'] - prev_row['atr'] * atr_multiplier
            
            # Check for signal
            if position is None:  # No position
                # Bullish breakout
                if close_price > high_breakout:
                    # Open long position
                    position = 'long'
                    position_size = capital * 0.95 / close_price  # Use 95% of capital
                    entry_price = close_price
                    logger.debug(f"Backtest: Opened long position at {close_price}")
                
                # Bearish breakout
                elif close_price < low_breakout:
                    # Open short position
                    position = 'short'
                    position_size = capital * 0.95 / close_price  # Use 95% of capital
                    entry_price = close_price
                    logger.debug(f"Backtest: Opened short position at {close_price}")
            
            else:  # Have position
                # Calculate profit/loss
                if position == 'long':
                    pnl = (close_price - entry_price) * position_size
                    # Close long on bearish breakout
                    if close_price < low_breakout:
                        trades.append({
                            'type': 'long',
                            'entry': entry_price,
                            'exit': close_price,
                            'pnl': pnl,
                            'pnl_pct': (close_price - entry_price) / entry_price * 100
                        })
                        capital += pnl
                        position = None
                        logger.debug(f"Backtest: Closed long position at {close_price}, PnL: {pnl:.2f}")
                
                else:  # short position
                    pnl = (entry_price - close_price) * position_size
                    # Close short on bullish breakout
                    if close_price > high_breakout:
                        trades.append({
                            'type': 'short',
                            'entry': entry_price,
                            'exit': close_price,
                            'pnl': pnl,
                            'pnl_pct': (entry_price - close_price) / entry_price * 100
                        })
                        capital += pnl
                        position = None
                        logger.debug(f"Backtest: Closed short position at {close_price}, PnL: {pnl:.2f}")
            
            # Track equity
            equity = capital
            if position == 'long':
                equity += (close_price - entry_price) * position_size
            elif position == 'short':
                equity += (entry_price - close_price) * position_size
                
            equity_curve.append(equity)
        
        # Close any open position at the end
        if position is not None:
            close_price = data.iloc[-1]['close']
            
            if position == 'long':
                pnl = (close_price - entry_price) * position_size
            else:  # short
                pnl = (entry_price - close_price) * position_size
                
            trades.append({
                'type': position,
                'entry': entry_price,
                'exit': close_price,
                'pnl': pnl,
                'pnl_pct': (close_price - entry_price) / entry_price * 100 if position == 'long' else (entry_price - close_price) / entry_price * 100
            })
            
            capital += pnl
            logger.debug(f"Backtest: Closed final {position} position at {close_price}, PnL: {pnl:.2f}")
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate drawdown
        max_equity = 0
        max_drawdown = 0
        
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            drawdown = (max_equity - equity) / max_equity * 100 if max_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Return backtest results
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        return results
