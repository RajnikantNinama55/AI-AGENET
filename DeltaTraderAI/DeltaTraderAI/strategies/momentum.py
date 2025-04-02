import logging
from strategies.base_strategy import BaseStrategy
from config import Config

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using price changes
    
    This strategy goes long when price momentum is strong to the upside,
    and goes short when price momentum is strong to the downside.
    """
    
    def __init__(self, symbol, timeframe, parameters=None, api_client=None, account_balance=10000.0):
        """
        Initialize the momentum strategy
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candle timeframe
            parameters (dict, optional): Strategy parameters
            api_client: API client instance
            account_balance (float): Initial account balance for risk management
        """
        # Set default parameters if none provided
        if parameters is None:
            parameters = Config.MOMENTUM_DEFAULTS.copy()
            
        super().__init__(symbol, timeframe, parameters, api_client, account_balance)
        
        logger.info(f"Initialized Momentum Strategy with parameters: {parameters}")
    
    def calculate_indicators(self, data=None):
        """
        Calculate momentum indicators
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        data = super().calculate_indicators(data)
        
        if data is None or len(data) == 0:
            logger.error("No data available for calculating indicators")
            return None
        
        # Extract parameters
        period = self.parameters.get('period', 14)
        
        # Calculate rate of change (ROC)
        data = self.indicators.add_roc(data, period, 'momentum')
        
        return data
    
    def generate_signal(self, data=None):
        """
        Generate trading signal based on momentum
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            str: Signal ('buy', 'sell', or 'hold')
        """
        if data is None:
            data = self.calculate_indicators()
            
        if data is None or len(data) < 1:
            logger.warning("Insufficient data for signal generation")
            return 'hold'
        
        # Get the last row
        last_row = data.iloc[-1]
        
        # Check for NaN values in the indicators
        if last_row['momentum'] is None:
            logger.warning("Missing momentum value, unable to generate signal")
            return 'hold'
        
        # Extract parameters
        threshold = self.parameters.get('threshold', 0.5)
        
        # Positive momentum above threshold -> buy signal
        if last_row['momentum'] > threshold:
            return 'buy'
        
        # Negative momentum below negative threshold -> sell signal
        elif last_row['momentum'] < -threshold:
            return 'sell'
        
        # Momentum in neutral range -> hold
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
        threshold = self.parameters.get('threshold', 0.5)
        
        # Generate signals for each candle
        for i in range(self.parameters.get('period', 14) + 1, len(data)):
            current_row = data.iloc[i]
            
            # Skip rows with missing indicator values
            if current_row['momentum'] is None:
                continue
            
            # Get close price and momentum
            close_price = current_row['close']
            momentum = current_row['momentum']
            
            # Check for signal
            if position is None:  # No position
                # Strong positive momentum -> buy signal
                if momentum > threshold:
                    # Open long position
                    position = 'long'
                    position_size = capital * 0.95 / close_price  # Use 95% of capital
                    entry_price = close_price
                    logger.debug(f"Backtest: Opened long position at {close_price}, Momentum: {momentum:.2f}")
                
                # Strong negative momentum -> sell signal
                elif momentum < -threshold:
                    # Open short position
                    position = 'short'
                    position_size = capital * 0.95 / close_price  # Use 95% of capital
                    entry_price = close_price
                    logger.debug(f"Backtest: Opened short position at {close_price}, Momentum: {momentum:.2f}")
            
            else:  # Have position
                # Calculate profit/loss
                if position == 'long':
                    pnl = (close_price - entry_price) * position_size
                    # Close long when momentum turns negative
                    if momentum < -threshold:
                        trades.append({
                            'type': 'long',
                            'entry': entry_price,
                            'exit': close_price,
                            'pnl': pnl,
                            'pnl_pct': (close_price - entry_price) / entry_price * 100
                        })
                        capital += pnl
                        position = None
                        logger.debug(f"Backtest: Closed long position at {close_price}, PnL: {pnl:.2f}, Momentum: {momentum:.2f}")
                
                else:  # short position
                    pnl = (entry_price - close_price) * position_size
                    # Close short when momentum turns positive
                    if momentum > threshold:
                        trades.append({
                            'type': 'short',
                            'entry': entry_price,
                            'exit': close_price,
                            'pnl': pnl,
                            'pnl_pct': (entry_price - close_price) / entry_price * 100
                        })
                        capital += pnl
                        position = None
                        logger.debug(f"Backtest: Closed short position at {close_price}, PnL: {pnl:.2f}, Momentum: {momentum:.2f}")
            
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
