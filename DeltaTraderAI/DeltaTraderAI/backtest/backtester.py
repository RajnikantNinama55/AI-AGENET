import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from api.delta_exchange import DeltaExchangeAPI
from config import Config

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtesting engine for trading strategies
    
    This class handles backtesting of individual strategies or combinations
    of strategies against historical market data.
    """
    
    def __init__(self, api_client=None):
        """
        Initialize the backtester
        
        Args:
            api_client: API client for fetching historical data
        """
        self.api_client = api_client or DeltaExchangeAPI()
        self.results = {}
    
    def fetch_historical_data(self, symbol, timeframe, start_time=None, end_time=None, limit=5000):
        """
        Fetch historical data for backtesting
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start timestamp
            end_time: End timestamp
            limit: Number of candles
            
        Returns:
            DataFrame: Historical OHLCV data
        """
        # Get current date for validation
        now = datetime.now()
        
        # Convert start and end times to timestamps if provided as strings
        if isinstance(start_time, str):
            start_date = datetime.strptime(start_time, "%Y-%m-%d")
            # Ensure start date is not in the future
            if start_date > now:
                start_date = now - timedelta(days=30)  # Use 30 days ago as fallback
            start_time = int(start_date.timestamp() * 1000)
            
        if isinstance(end_time, str):
            end_date = datetime.strptime(end_time, "%Y-%m-%d")
            # Ensure end date is not in the future
            if end_date > now:
                end_date = now
            end_time = int(end_date.timestamp() * 1000)
            
        # If end_time is None, use current time
        if end_time is None:
            end_time = int(now.timestamp() * 1000)
            
        # Ensure start_time is not in the future and default to 30 days ago if not specified
        if start_time is None:
            start_time = int((now - timedelta(days=30)).timestamp() * 1000)
        elif start_time > end_time:
            start_time = int((now - timedelta(days=30)).timestamp() * 1000)
        
        try:
            # Fetch data in chunks if needed (API may have limitations)
            all_candles = []
            remaining = limit
            current_end = end_time
            
            while remaining > 0:
                chunk_size = min(1000, remaining)  # Most APIs limit to 1000 candles per request
                
                candles_data = self.api_client.get_candles(
                    symbol=symbol,
                    resolution=timeframe,
                    start_time=start_time,
                    end_time=current_end,
                    limit=chunk_size
                )
                
                if not candles_data.get('result') or len(candles_data['result']) == 0:
                    break
                    
                chunk = candles_data['result']
                all_candles.extend(chunk)
                
                # Update parameters for next request
                remaining -= len(chunk)
                if remaining > 0 and len(chunk) > 0:
                    # Use earliest timestamp as new end_time
                    earliest = min(c['time'] for c in chunk)
                    current_end = earliest - 1
                else:
                    break
            
            if not all_candles:
                logger.error(f"Failed to fetch historical data for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(all_candles)
            
            # Rename columns to standard OHLCV names
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert columns to proper numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    # Handle potential string values by converting to float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def backtest_strategy(self, strategy, historical_data=None, initial_capital=10000):
        """
        Backtest a single strategy
        
        Args:
            strategy: Strategy instance
            historical_data: Historical OHLCV data (fetched if None)
            initial_capital: Initial capital for the backtest
            
        Returns:
            dict: Backtest results
        """
        if historical_data is None:
            # Fetch data if not provided
            historical_data = self.fetch_historical_data(
                symbol=strategy.symbol,
                timeframe=strategy.timeframe,
                start_time=Config.BACKTEST_START_DATE,
                end_time=Config.BACKTEST_END_DATE
            )
            
            if historical_data is None:
                logger.error("Failed to fetch historical data for backtesting")
                return None
        
        # Run strategy backtest
        strategy_name = strategy.__class__.__name__
        logger.info(f"Backtesting {strategy_name} on {strategy.symbol} {strategy.timeframe}")
        
        results = strategy.backtest(historical_data, initial_capital)
        
        if results:
            # Store results
            self.results[strategy_name] = results
            logger.info(f"Backtest complete for {strategy_name}: Return: {results['total_return']:.2f}%, "
                        f"Win Rate: {results['win_rate']:.2f}%, Max Drawdown: {results['max_drawdown']:.2f}%")
        else:
            logger.error(f"Backtest failed for {strategy_name}")
        
        return results
    
    def backtest_multi_strategy(self, strategies, historical_data=None, initial_capital=10000, weights=None):
        """
        Backtest multiple strategies with customizable weights
        
        Args:
            strategies: List of strategy instances
            historical_data: Historical OHLCV data (fetched if None)
            initial_capital: Initial capital for the backtest
            weights: Dictionary of strategy weights (equal if None)
            
        Returns:
            dict: Combined backtest results
        """
        if not strategies:
            logger.error("No strategies provided for multi-strategy backtest")
            return None
        
        # Ensure all strategies use the same symbol and timeframe
        symbol = strategies[0].symbol
        timeframe = strategies[0].timeframe
        
        for s in strategies:
            if s.symbol != symbol or s.timeframe != timeframe:
                logger.error("All strategies must use the same symbol and timeframe for multi-strategy backtest")
                return None
        
        if historical_data is None:
            # Fetch data if not provided
            historical_data = self.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=Config.BACKTEST_START_DATE,
                end_time=Config.BACKTEST_END_DATE
            )
            
            if historical_data is None:
                logger.error("Failed to fetch historical data for backtesting")
                return None
        
        # Initialize with equal weights if not provided
        if weights is None:
            weights = {s.__class__.__name__: 1.0 / len(strategies) for s in strategies}
        
        # Backtest each strategy individually first
        strategy_results = {}
        for s in strategies:
            strategy_name = s.__class__.__name__
            results = self.backtest_strategy(s, historical_data.copy(), initial_capital)
            if results:
                strategy_results[strategy_name] = results
        
        if not strategy_results:
            logger.error("All individual strategy backtests failed")
            return None
        
        # Combine the signals using the provided weights
        combined_signals = pd.DataFrame(index=historical_data.index)
        combined_signals['close'] = historical_data['close']
        
        # Get signals from each strategy
        for s in strategies:
            strategy_name = s.__class__.__name__
            
            # Calculate indicators
            data = s.calculate_indicators(historical_data.copy())
            
            # Generate signal for each data point
            signals = pd.Series(index=data.index)
            for i in range(len(data)):
                # Create a slice of data up to current point
                slice_data = data.iloc[:i+1]
                
                # Skip if not enough data
                if len(slice_data) < 2:
                    signals.iloc[i] = 0  # Hold
                    continue
                
                # Generate signal
                signal = s.generate_signal(slice_data)
                
                # Convert signal to numeric
                if signal == 'buy':
                    signals.iloc[i] = 1
                elif signal == 'sell':
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0
            
            # Add to combined signals
            combined_signals[strategy_name] = signals
        
        # Apply weights to combined signals
        weighted_signal = pd.Series(0, index=combined_signals.index)
        
        for s in strategies:
            strategy_name = s.__class__.__name__
            if strategy_name in combined_signals.columns and strategy_name in weights:
                weighted_signal += combined_signals[strategy_name] * weights[strategy_name]
        
        # Threshold the weighted signal
        combined_signals['combined'] = np.where(weighted_signal > 0.2, 1, np.where(weighted_signal < -0.2, -1, 0))
        
        # Run backtest on combined signal
        capital = initial_capital
        position = None
        position_size = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(combined_signals)):
            current_row = combined_signals.iloc[i]
            prev_row = combined_signals.iloc[i-1]
            
            signal = current_row['combined']
            close_price = current_row['close']
            
            # Check for signal changes
            if position is None:  # No position
                if signal == 1:  # Buy signal
                    # Open long position
                    position = 'long'
                    position_size = capital * 0.95 / close_price  # Use 95% of capital
                    entry_price = close_price
                    logger.debug(f"Backtest: Multi-strategy opened long position at {close_price}")
                
                elif signal == -1:  # Sell signal
                    # Open short position
                    position = 'short'
                    position_size = capital * 0.95 / close_price  # Use 95% of capital
                    entry_price = close_price
                    logger.debug(f"Backtest: Multi-strategy opened short position at {close_price}")
            
            else:  # Have position
                # Calculate profit/loss
                if position == 'long':
                    pnl = (close_price - entry_price) * position_size
                    # Close long when signal turns sell
                    if signal == -1:
                        trades.append({
                            'type': 'long',
                            'entry': entry_price,
                            'exit': close_price,
                            'pnl': pnl,
                            'pnl_pct': (close_price - entry_price) / entry_price * 100
                        })
                        capital += pnl
                        position = None
                        logger.debug(f"Backtest: Multi-strategy closed long position at {close_price}, PnL: {pnl:.2f}")
                
                else:  # short position
                    pnl = (entry_price - close_price) * position_size
                    # Close short when signal turns buy
                    if signal == 1:
                        trades.append({
                            'type': 'short',
                            'entry': entry_price,
                            'exit': close_price,
                            'pnl': pnl,
                            'pnl_pct': (entry_price - close_price) / entry_price * 100
                        })
                        capital += pnl
                        position = None
                        logger.debug(f"Backtest: Multi-strategy closed short position at {close_price}, PnL: {pnl:.2f}")
            
            # Track equity
            equity = capital
            if position == 'long':
                equity += (close_price - entry_price) * position_size
            elif position == 'short':
                equity += (entry_price - close_price) * position_size
                
            equity_curve.append(equity)
        
        # Close any open position at the end
        if position is not None:
            close_price = combined_signals.iloc[-1]['close']
            
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
            logger.debug(f"Backtest: Multi-strategy closed final {position} position at {close_price}, PnL: {pnl:.2f}")
        
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
        
        # Combined results
        combined_results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'weights': weights,
            'individual_results': strategy_results
        }
        
        # Store results
        self.results['combined'] = combined_results
        
        logger.info(f"Multi-strategy backtest complete: Return: {combined_results['total_return']:.2f}%, "
                   f"Win Rate: {combined_results['win_rate']:.2f}%, Max Drawdown: {combined_results['max_drawdown']:.2f}%")
        
        return combined_results
    
    def plot_results(self, strategy_name=None, show_trades=True):
        """
        Plot backtest results
        
        Args:
            strategy_name: Name of the strategy to plot (plots all if None)
            show_trades: Whether to mark trades on the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if not self.results:
            logger.error("No backtest results to plot")
            return None
        
        if strategy_name and strategy_name not in self.results:
            logger.error(f"Strategy {strategy_name} not found in results")
            return None
        
        plt.figure(figsize=(12, 8))
        
        if strategy_name:
            # Plot single strategy
            results = self.results[strategy_name]
            self._plot_equity_curve(results, strategy_name, show_trades)
        else:
            # Plot all strategies
            for name, results in self.results.items():
                self._plot_equity_curve(results, name, show_trades=False)
        
        plt.title('Backtest Results')
        plt.xlabel('Trade Number')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def _plot_equity_curve(self, results, name, show_trades=True):
        """
        Plot equity curve for a strategy
        
        Args:
            results: Backtest results
            name: Strategy name
            show_trades: Whether to mark trades
        """
        equity_curve = results['equity_curve']
        plt.plot(range(len(equity_curve)), equity_curve, label=f"{name} (Return: {results['total_return']:.2f}%)")
        
        if show_trades and 'trades' in results:
            trade_points = []
            trade_values = []
            trade_colors = []
            
            # Calculate portfolio value at each trade
            equity_index = 0
            
            for trade in results['trades']:
                # Find approximate position in equity curve
                equity_index += 1  # Simplification, should match actual trade timing
                if equity_index < len(equity_curve):
                    trade_points.append(equity_index)
                    trade_values.append(equity_curve[equity_index])
                    trade_colors.append('g' if trade['pnl'] > 0 else 'r')
            
            plt.scatter(trade_points, trade_values, c=trade_colors, s=50)
    
    def get_results(self, strategy_name=None):
        """
        Get backtest results
        
        Args:
            strategy_name: Name of the strategy (all results if None)
            
        Returns:
            dict: Backtest results
        """
        if not self.results:
            logger.warning("No backtest results available")
            return {}
            
        if strategy_name:
            return self.results.get(strategy_name, {})
        else:
            return self.results
    
    def get_combined_metrics(self):
        """
        Get combined performance metrics for all strategies
        
        Returns:
            dict: Combined metrics
        """
        if not self.results:
            logger.warning("No backtest results available")
            return {}
        
        metrics = {}
        
        for name, results in self.results.items():
            metrics[name] = {
                'return': results.get('total_return', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'trades': results.get('total_trades', 0)
            }
        
        return metrics
    
    def optimize_parameters(self, strategy_class, symbol, timeframe, param_grid, historical_data=None):
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_class: Strategy class
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            param_grid: Dictionary of parameter grids to search
            historical_data: Historical OHLCV data (fetched if None)
            
        Returns:
            dict: Best parameters and results
        """
        if historical_data is None:
            # Fetch data if not provided
            historical_data = self.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=Config.BACKTEST_START_DATE,
                end_time=Config.BACKTEST_END_DATE
            )
            
            if historical_data is None:
                logger.error("Failed to fetch historical data for optimization")
                return None
        
        logger.info(f"Optimizing parameters for {strategy_class.__name__} on {symbol} {timeframe}")
        
        # Generate parameter combinations
        from itertools import product
        
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_return = -float('inf')
        best_params = None
        best_results = None
        
        # For tracking progress
        total_combinations = np.prod([len(v) for v in param_values])
        current = 0
        
        for params in product(*param_values):
            current += 1
            
            # Create parameter dictionary
            param_dict = dict(zip(param_keys, params))
            
            # Create strategy with these parameters
            strategy = strategy_class(symbol, timeframe, param_dict, self.api_client)
            
            # Run backtest
            results = self.backtest_strategy(strategy, historical_data.copy())
            
            if results and results['total_return'] > best_return:
                best_return = results['total_return']
                best_params = param_dict
                best_results = results
            
            # Log progress
            if current % 10 == 0 or current == total_combinations:
                logger.info(f"Optimization progress: {current}/{total_combinations} combinations tested")
        
        if best_params:
            logger.info(f"Best parameters found: {best_params} with return: {best_return:.2f}%")
        else:
            logger.error("Optimization failed to find valid parameters")
        
        return {
            'parameters': best_params,
            'results': best_results
        }
