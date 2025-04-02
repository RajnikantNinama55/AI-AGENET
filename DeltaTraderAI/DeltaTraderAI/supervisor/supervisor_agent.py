"""
Supervisor Agent for Trading Bot

This agent analyzes historical data and optimizes the trading strategies
by suggesting improvements such as:
- Better parameters for existing strategies
- New technical indicators to incorporate
- Adjustments to RL model hyperparameters
- Optimal strategy combinations

The supervisor runs in the background, continuously analyzing performance
and making recommendations for improvement. It also integrates with the
evolution system for continuous learning and adaptation.
"""

import logging
import time
import os
import numpy as np
import pandas as pd
from threading import Thread, Event
from datetime import datetime, timedelta
import json

from backtest.backtester import Backtester
from api.delta_exchange import DeltaExchangeAPI
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.momentum import MomentumStrategy
from rl.ppo_model import PPOModel
from config import Config
from models import StrategyConfiguration, db
from evolution.evolution_manager import EvolutionManager
from evolution.continuous_learning import ContinuousLearningScheduler
from evolution.adaptive_risk import AdaptiveRiskManager

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """
    Supervisor Agent that monitors and optimizes trading strategies.
    
    This agent runs in the background as a separate thread, analyzing strategy
    performance and making recommendations for improvements. It can optimize
    strategy parameters, suggest new indicators, and combine strategies for
    optimal performance.
    """
    
    def __init__(self, optimization_interval=3600, active_strategies=None):
        """
        Initialize the supervisor agent
        
        Args:
            optimization_interval (int): How often to run optimization in seconds
            active_strategies (dict): Reference to active strategy instances
        """
        self.optimization_interval = optimization_interval
        self.active_strategies = active_strategies or {}
        # Get the current user's API settings
        from models import User
        from app import db
        
        with db.session.no_autoflush:
            user = User.query.first()
            
        if user:
            self.api_client = DeltaExchangeAPI(
                api_key=user.api_key or Config.DELTA_EXCHANGE_API_KEY,
                api_secret=user.api_secret or Config.DELTA_EXCHANGE_API_SECRET
            )
        else:
            self.api_client = DeltaExchangeAPI(
                api_key=Config.DELTA_EXCHANGE_API_KEY,
                api_secret=Config.DELTA_EXCHANGE_API_SECRET
            )
        self.backtester = Backtester(api_client=self.api_client)
        self.ppo_model = PPOModel()
        self.running = False
        self.stop_event = Event()
        self.supervisor_thread = None
        self.recommendations = []
        self.last_optimization = None
        
        # Default hyperparameters for RL model
        self.rl_hyperparams = {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'batch_size': 64,
            'n_epochs': 10
        }
        
        # Performance metrics tracked for each strategy
        self.strategy_metrics = {}
        
        # Initialize the evolution system components
        self.evolution_manager = EvolutionManager(
            supervisor=self,
            api_client=self.api_client,
            backtester=self.backtester
        )
        
        self.adaptive_risk_manager = AdaptiveRiskManager(
            evolution_manager=self.evolution_manager,
            api_client=self.api_client
        )
        
        self.continuous_learning = None  # Will be initialized when starting
        
        # Directory for storing strategy optimization data
        os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
        
        logger.info("Supervisor Agent initialized with Evolution System")
    
    def start(self):
        """Start the supervisor agent in a background thread"""
        if self.running:
            logger.warning("Supervisor Agent is already running")
            return False
        
        # Initialize the continuous learning scheduler if not already
        if self.continuous_learning is None:
            self.continuous_learning = ContinuousLearningScheduler(
                supervisor=self,
                api_client=self.api_client,
                backtester=self.backtester,
                ppo_model=self.ppo_model
            )
        
        # Start the evolution system
        self.continuous_learning.start()
        
        # Start the supervisor agent
        self.running = True
        self.stop_event.clear()
        self.supervisor_thread = Thread(target=self._supervisor_loop)
        self.supervisor_thread.daemon = True
        self.supervisor_thread.start()
        
        logger.info("Supervisor Agent started with Continuous Learning")
        return True
    
    def stop(self):
        """Stop the supervisor agent"""
        if not self.running:
            logger.warning("Supervisor Agent is not running")
            return False
        
        # Stop the evolution system
        if self.continuous_learning:
            self.continuous_learning.stop()
        
        # Stop the supervisor agent
        self.running = False
        self.stop_event.set()
        if self.supervisor_thread:
            self.supervisor_thread.join(timeout=5.0)
        
        logger.info("Supervisor Agent stopped")
        return True
    
    def _supervisor_loop(self):
        """Main loop for the supervisor agent"""
        logger.info("Supervisor Agent loop started")
        
        # Import Flask app to get application context
        from app import app
        
        while not self.stop_event.is_set():
            try:
                # Check if it's time to run optimization
                current_time = datetime.now()
                
                if (self.last_optimization is None or 
                    (current_time - self.last_optimization).total_seconds() >= self.optimization_interval):
                    
                    logger.info("Running strategy optimization")
                    
                    # Run all database operations in an application context
                    with app.app_context():
                        # Collect data about current strategies
                        self._collect_strategy_data()
                        
                        # Optimize strategy parameters
                        self._optimize_strategy_parameters()
                        
                        # Analyze strategy combination
                        self._analyze_strategy_combination()
                        
                        # Check for new indicators to add
                        self._suggest_new_indicators()
                        
                        # Optimize RL hyperparameters
                        self._optimize_rl_hyperparameters()
                        
                        # Apply high-confidence changes automatically
                        self._apply_safe_optimizations()
                    
                    # Update last optimization time
                    self.last_optimization = current_time
                    
                    logger.info("Strategy optimization completed")
                
                # Sleep for a while before checking again
                for _ in range(60):  # Check every minute if we should stop
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in supervisor loop: {e}")
                time.sleep(60)  # Sleep for a minute and try again
    
    def _collect_strategy_data(self):
        """Collect performance data for each active strategy"""
        try:
            # Get all active strategy configurations from database
            strategy_configs = StrategyConfiguration.query.filter_by(is_active=True).all()
            
            if not strategy_configs:
                logger.warning("No active strategy configurations found")
                return
            
            for config in strategy_configs:
                # Initialize strategy for testing
                strategy_type = config.strategy_type
                symbol = config.symbol
                timeframe = config.timeframe
                parameters = config.parameters
                
                # Create key for tracking this strategy
                strategy_key = f"{strategy_type}_{symbol}_{timeframe}"
                
                # Initialize metrics if not exist
                if strategy_key not in self.strategy_metrics:
                    self.strategy_metrics[strategy_key] = {
                        'win_rate': [],
                        'profit_factor': [],
                        'avg_profit': [],
                        'max_drawdown': [],
                        'sharpe_ratio': [],
                        'parameters': [],
                        'returns': []
                    }
                
                # Fetch historical data
                historical_data = self.backtester.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_time=datetime.now().strftime('%Y-%m-%d')
                )
                
                if historical_data is None or len(historical_data) == 0:
                    logger.warning(f"No historical data available for {symbol} {timeframe}")
                    continue
                
                # Initialize strategy with current parameters
                strategy_instance = self._create_strategy_instance(
                    strategy_type, symbol, timeframe, parameters
                )
                
                if strategy_instance is None:
                    continue
                
                # Run backtest
                backtest_results = self.backtester.backtest_strategy(
                    strategy=strategy_instance,
                    historical_data=historical_data.copy()
                )
                
                if backtest_results:
                    # Store metrics
                    metrics = self.strategy_metrics[strategy_key]
                    metrics['win_rate'].append(backtest_results.get('win_rate', 0))
                    metrics['profit_factor'].append(backtest_results.get('profit_factor', 0))
                    metrics['max_drawdown'].append(backtest_results.get('max_drawdown', 0))
                    metrics['returns'].append(backtest_results.get('total_return', 0))
                    metrics['parameters'].append(parameters)
                    
                    # Calculate Sharpe ratio (simplified)
                    if 'equity_curve' in backtest_results and len(backtest_results['equity_curve']) > 1:
                        equity = backtest_results['equity_curve']
                        returns = [((equity[i] / equity[i-1]) - 1) for i in range(1, len(equity))]
                        avg_return = np.mean(returns) if returns else 0
                        std_return = np.std(returns) if returns else 1
                        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
                        metrics['sharpe_ratio'].append(sharpe)
                    else:
                        metrics['sharpe_ratio'].append(0)
                    
                    # Calculate average profit
                    if 'trades' in backtest_results and backtest_results['trades']:
                        profits = [t.get('pnl', 0) for t in backtest_results['trades']]
                        avg_profit = np.mean(profits) if profits else 0
                        metrics['avg_profit'].append(avg_profit)
                    else:
                        metrics['avg_profit'].append(0)
                    
                    logger.info(f"Collected metrics for {strategy_key}: "
                                f"Win Rate: {backtest_results.get('win_rate', 0):.2f}%, "
                                f"Return: {backtest_results.get('total_return', 0):.2f}%")
        
        except Exception as e:
            logger.error(f"Error collecting strategy data: {e}")
    
    def _optimize_strategy_parameters(self):
        """
        Optimize parameters for each strategy by testing different combinations
        and measuring performance.
        """
        try:
            optimized_parameters = {}
            
            # Get all active strategy configurations from database
            strategy_configs = StrategyConfiguration.query.filter_by(is_active=True).all()
            
            for config in strategy_configs:
                strategy_type = config.strategy_type
                symbol = config.symbol
                timeframe = config.timeframe
                current_params = config.parameters
                
                strategy_key = f"{strategy_type}_{symbol}_{timeframe}"
                
                # Skip if we don't have historical metrics
                if strategy_key not in self.strategy_metrics:
                    continue
                
                # Fetch recent historical data for optimization
                historical_data = self.backtester.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_time=datetime.now().strftime('%Y-%m-%d')
                )
                
                if historical_data is None or len(historical_data) == 0:
                    logger.warning(f"No historical data available for parameter optimization of {strategy_key}")
                    continue
                
                # Generate parameter combinations to test
                param_combinations = self._generate_parameter_combinations(strategy_type, current_params)
                
                best_return = -float('inf')
                best_params = None
                
                # Test each parameter combination
                for params in param_combinations:
                    # Initialize strategy with test parameters
                    strategy_instance = self._create_strategy_instance(
                        strategy_type, symbol, timeframe, params
                    )
                    
                    if strategy_instance is None:
                        continue
                    
                    # Run backtest
                    results = self.backtester.backtest_strategy(
                        strategy=strategy_instance,
                        historical_data=historical_data.copy()
                    )
                    
                    if results:
                        total_return = results.get('total_return', 0)
                        win_rate = results.get('win_rate', 0)
                        max_drawdown = results.get('max_drawdown', 0)
                        
                        # Use a scoring formula that considers multiple factors
                        score = total_return * 0.6 + win_rate * 0.3 - max_drawdown * 0.1
                        
                        if score > best_return:
                            best_return = score
                            best_params = params
                            logger.debug(f"New best parameters for {strategy_key}: {best_params}, Score: {score:.2f}")
                
                # Store the optimized parameters
                if best_params and best_params != current_params:
                    improvement = best_return - max(self.strategy_metrics[strategy_key]['returns']) if self.strategy_metrics[strategy_key]['returns'] else best_return
                    
                    # Skip negligible improvements
                    if improvement > 0.5:  # Significant if >0.5% improvement
                        optimized_parameters[strategy_key] = {
                            'old_params': current_params,
                            'new_params': best_params,
                            'improvement': improvement,
                            'config_id': config.id
                        }
                        
                        logger.info(f"Found improved parameters for {strategy_key}: {best_params}")
                        logger.info(f"Estimated improvement: {improvement:.2f}%")
                        
                        # Add to recommendations
                        self.recommendations.append({
                            'type': 'parameter_optimization',
                            'strategy_key': strategy_key,
                            'old_params': current_params,
                            'new_params': best_params,
                            'improvement': improvement,
                            'confidence': 'high' if improvement > 3 else 'medium',
                            'timestamp': datetime.now().isoformat(),
                            'config_id': config.id
                        })
            
            return optimized_parameters
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return {}
    
    def _generate_parameter_combinations(self, strategy_type, current_params):
        """
        Generate parameter combinations to test for the given strategy type.
        """
        param_combinations = [current_params]  # Include current parameters
        
        try:
            # Generate variations based on strategy type
            if strategy_type == 'trend_following':
                # Test different SMA windows
                for short_window in [10, 15, 20, 25, 30]:
                    for long_window in [40, 50, 60, 80, 100]:
                        if short_window < long_window:
                            # Vary take profit and stop loss
                            for take_profit in [2.0, 3.0, 4.0, 5.0]:
                                for stop_loss in [1.0, 1.5, 2.0, 2.5]:
                                    params = {
                                        'short_window': short_window,
                                        'long_window': long_window,
                                        'take_profit': take_profit,
                                        'stop_loss': stop_loss
                                    }
                                    # Skip if too similar to current
                                    if not self._params_too_similar(params, current_params):
                                        param_combinations.append(params)
            
            elif strategy_type == 'mean_reversion':
                # Test different RSI settings
                for rsi_period in [9, 14, 21, 30]:
                    for oversold in [20, 25, 30, 35]:
                        for overbought in [65, 70, 75, 80]:
                            for take_profit in [1.5, 2.0, 2.5, 3.0]:
                                for stop_loss in [1.0, 1.5, 2.0, 2.5]:
                                    params = {
                                        'rsi_period': rsi_period,
                                        'oversold_threshold': oversold,
                                        'overbought_threshold': overbought,
                                        'take_profit': take_profit,
                                        'stop_loss': stop_loss
                                    }
                                    if not self._params_too_similar(params, current_params):
                                        param_combinations.append(params)
            
            elif strategy_type == 'breakout':
                # Test different breakout settings
                for period in [10, 15, 20, 25, 30]:
                    for atr_period in [10, 14, 21]:
                        for atr_multiplier in [1.0, 1.5, 2.0, 2.5, 3.0]:
                            for take_profit in [3.0, 4.0, 5.0]:
                                for stop_loss in [1.5, 2.0, 2.5, 3.0]:
                                    params = {
                                        'period': period,
                                        'atr_period': atr_period,
                                        'atr_multiplier': atr_multiplier,
                                        'take_profit': take_profit,
                                        'stop_loss': stop_loss
                                    }
                                    if not self._params_too_similar(params, current_params):
                                        param_combinations.append(params)
            
            elif strategy_type == 'momentum':
                # Test different momentum settings
                for period in [7, 10, 14, 21, 28]:
                    for threshold in [0.3, 0.5, 0.7, 1.0]:
                        for take_profit in [2.0, 3.0, 4.0]:
                            for stop_loss in [1.5, 2.0, 2.5]:
                                params = {
                                    'period': period,
                                    'threshold': threshold,
                                    'take_profit': take_profit,
                                    'stop_loss': stop_loss
                                }
                                if not self._params_too_similar(params, current_params):
                                    param_combinations.append(params)
            
            # Limit the number of combinations to test
            max_combinations = 20
            if len(param_combinations) > max_combinations:
                # Keep current params and a random selection of others
                samples = [current_params] + np.random.choice(
                    [p for p in param_combinations if p != current_params], 
                    size=max_combinations-1, 
                    replace=False
                ).tolist()
                return samples
            
            return param_combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {e}")
            return [current_params]
    
    def _params_too_similar(self, params1, params2):
        """Check if two parameter sets are too similar to both test"""
        diff_count = 0
        for key in params1:
            if key in params2:
                # For numeric values
                if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    # Calculate percent difference
                    if params2[key] != 0:
                        percent_diff = abs(params1[key] - params2[key]) / abs(params2[key])
                        if percent_diff > 0.1:  # 10% difference threshold
                            diff_count += 1
                elif params1[key] != params2[key]:
                    diff_count += 1
        
        # If at least 2 parameters are significantly different
        return diff_count < 2
    
    def _create_strategy_instance(self, strategy_type, symbol, timeframe, parameters):
        """Create a strategy instance of the specified type with given parameters"""
        try:
            if strategy_type == 'trend_following':
                return TrendFollowingStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=self.api_client
                )
            elif strategy_type == 'mean_reversion':
                return MeanReversionStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=self.api_client
                )
            elif strategy_type == 'breakout':
                return BreakoutStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=self.api_client
                )
            elif strategy_type == 'momentum':
                return MomentumStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=self.api_client
                )
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating strategy instance: {e}")
            return None
    
    def _analyze_strategy_combination(self):
        """
        Analyze how to best combine different strategies by optimizing weights
        """
        try:
            # Group strategies by symbol and timeframe
            strategy_groups = {}
            
            # Get all active strategy configurations from database
            strategy_configs = StrategyConfiguration.query.filter_by(is_active=True).all()
            
            if not strategy_configs:
                return
            
            for config in strategy_configs:
                key = f"{config.symbol}_{config.timeframe}"
                if key not in strategy_groups:
                    strategy_groups[key] = []
                strategy_groups[key].append(config)
            
            # For each group, optimize the combination weights
            for key, configs in strategy_groups.items():
                if len(configs) < 2:
                    # Need at least 2 strategies to combine
                    continue
                
                symbol, timeframe = key.split('_')
                
                # Fetch historical data
                historical_data = self.backtester.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_time=datetime.now().strftime('%Y-%m-%d')
                )
                
                if historical_data is None or len(historical_data) == 0:
                    logger.warning(f"No historical data available for {symbol} {timeframe}")
                    continue
                
                # Initialize strategies
                strategies = []
                for config in configs:
                    strategy_instance = self._create_strategy_instance(
                        config.strategy_type, config.symbol, config.timeframe, config.parameters
                    )
                    if strategy_instance:
                        strategies.append(strategy_instance)
                
                if len(strategies) < 2:
                    continue
                
                # Use PPO model to optimize weights
                optimized_weights = self.ppo_model.optimize_strategy_weights(
                    strategies=strategies,
                    historical_data=historical_data,
                    episodes=10  # Reduced for faster processing
                )
                
                if optimized_weights:
                    # Check if weights are significantly different from current weights
                    current_weights = {config.strategy_type: config.weight for config in configs}
                    weight_diff = False
                    
                    for strategy_type, weight in optimized_weights.items():
                        if strategy_type in current_weights:
                            if abs(weight - current_weights[strategy_type]) > 0.1:  # 10% threshold
                                weight_diff = True
                                break
                    
                    if weight_diff:
                        logger.info(f"Optimized weights for {key}: {optimized_weights}")
                        
                        # Add to recommendations
                        self.recommendations.append({
                            'type': 'weight_optimization',
                            'symbol_timeframe': key,
                            'old_weights': current_weights,
                            'new_weights': optimized_weights,
                            'confidence': 'medium',
                            'timestamp': datetime.now().isoformat(),
                            'configs': [config.id for config in configs]
                        })
        
        except Exception as e:
            logger.error(f"Error analyzing strategy combination: {e}")
    
    def _suggest_new_indicators(self):
        """
        Analyze strategy performance and suggest new indicators that might improve performance
        """
        try:
            suggested_indicators = []
            
            # Get all active strategy configurations from database
            strategy_configs = StrategyConfiguration.query.filter_by(is_active=True).all()
            
            for config in strategy_configs:
                strategy_type = config.strategy_type
                symbol = config.symbol
                timeframe = config.timeframe
                
                strategy_key = f"{strategy_type}_{symbol}_{timeframe}"
                
                # Check recent performance
                if strategy_key in self.strategy_metrics:
                    metrics = self.strategy_metrics[strategy_key]
                    
                    # Only suggest if we have enough data
                    if len(metrics['win_rate']) >= 2:
                        avg_win_rate = np.mean(metrics['win_rate'])
                        avg_profit_factor = np.mean([pf for pf in metrics['profit_factor'] if pf != float('inf')])
                        avg_return = np.mean(metrics['returns'])
                        
                        # Check conditions that indicate need for specific indicators
                        if strategy_type == 'trend_following' and avg_win_rate < 50:
                            # Low win rate in trend strategy might benefit from adding a filter
                            suggested_indicators.append({
                                'strategy_key': strategy_key,
                                'indicator': 'ADX',
                                'reason': 'Adding ADX as a trend strength filter could improve win rate',
                                'implementation': 'Use ADX > 25 to confirm trend strength before entering trades',
                                'confidence': 'medium',
                                'config_id': config.id
                            })
                        
                        elif strategy_type == 'mean_reversion' and avg_profit_factor < 1.2:
                            # Low profit factor in mean reversion might need better entry timing
                            suggested_indicators.append({
                                'strategy_key': strategy_key,
                                'indicator': 'Bollinger Bands',
                                'reason': 'Using Bollinger Bands with RSI could improve entry timing',
                                'implementation': 'Buy when price touches lower band AND RSI is oversold',
                                'confidence': 'high',
                                'config_id': config.id
                            })
                        
                        elif strategy_type == 'breakout' and avg_return < 5:
                            # Poor returns in breakout strategy might need volume confirmation
                            suggested_indicators.append({
                                'strategy_key': strategy_key,
                                'indicator': 'Volume',
                                'reason': 'Adding volume confirmation to breakouts could reduce false signals',
                                'implementation': 'Only enter breakout trades when volume is above average',
                                'confidence': 'medium',
                                'config_id': config.id
                            })
                        
                        elif strategy_type == 'momentum' and avg_win_rate < 45:
                            # Poor win rate in momentum might benefit from MACD
                            suggested_indicators.append({
                                'strategy_key': strategy_key,
                                'indicator': 'MACD',
                                'reason': 'MACD can help filter out low-quality momentum signals',
                                'implementation': 'Only enter when MACD histogram is increasing in the trade direction',
                                'confidence': 'high',
                                'config_id': config.id
                            })
            
            # Add to recommendations
            for suggestion in suggested_indicators:
                self.recommendations.append({
                    'type': 'new_indicator',
                    'strategy_key': suggestion['strategy_key'],
                    'indicator': suggestion['indicator'],
                    'reason': suggestion['reason'],
                    'implementation': suggestion['implementation'],
                    'confidence': suggestion['confidence'],
                    'timestamp': datetime.now().isoformat(),
                    'config_id': suggestion['config_id']
                })
                
                logger.info(f"Suggested new indicator for {suggestion['strategy_key']}: {suggestion['indicator']}")
            
            return suggested_indicators
            
        except Exception as e:
            logger.error(f"Error suggesting new indicators: {e}")
            return []
    
    def _optimize_rl_hyperparameters(self):
        """
        Optimize hyperparameters for the RL model based on performance metrics
        """
        try:
            # This would actually involve a grid search or Bayesian optimization
            # For now, use a simplified approach based on performance
            
            # Get combined performance across all strategies
            all_returns = []
            all_win_rates = []
            
            for key, metrics in self.strategy_metrics.items():
                if metrics['returns']:
                    all_returns.extend(metrics['returns'])
                if metrics['win_rate']:
                    all_win_rates.extend(metrics['win_rate'])
            
            if not all_returns:
                return
            
            avg_return = np.mean(all_returns)
            avg_win_rate = np.mean(all_win_rates) if all_win_rates else 0
            
            new_hyperparams = self.rl_hyperparams.copy()
            changes_made = False
            
            # Adjust learning rate based on performance trend
            if len(all_returns) > 2:
                # Check if returns are improving or declining
                returns_trend = np.polyfit(range(len(all_returns)), all_returns, 1)[0]
                
                if returns_trend < -0.5:  # Declining performance
                    # Reduce learning rate to stabilize
                    new_hyperparams['learning_rate'] = max(0.0001, self.rl_hyperparams['learning_rate'] * 0.8)
                    changes_made = True
                elif returns_trend > 0.5:  # Improving performance
                    # Slightly increase learning rate to speed up adaptation
                    new_hyperparams['learning_rate'] = min(0.001, self.rl_hyperparams['learning_rate'] * 1.2)
                    changes_made = True
            
            # Adjust gamma (discount factor) based on win rate
            if avg_win_rate < 45:
                # More emphasis on immediate rewards
                new_hyperparams['gamma'] = max(0.9, self.rl_hyperparams['gamma'] * 0.95)
                changes_made = True
            elif avg_win_rate > 60:
                # More emphasis on long-term rewards
                new_hyperparams['gamma'] = min(0.995, self.rl_hyperparams['gamma'] * 1.01)
                changes_made = True
            
            # Adjust clip range based on overall return
            if avg_return < 0:
                # More conservative updates when performing poorly
                new_hyperparams['clip_range'] = max(0.1, self.rl_hyperparams['clip_range'] * 0.9)
                changes_made = True
            elif avg_return > 10:
                # Allow larger updates when performing well
                new_hyperparams['clip_range'] = min(0.3, self.rl_hyperparams['clip_range'] * 1.1)
                changes_made = True
            
            if changes_made:
                logger.info(f"Optimized RL hyperparameters: {new_hyperparams}")
                
                # Add to recommendations
                self.recommendations.append({
                    'type': 'rl_hyperparameters',
                    'old_params': self.rl_hyperparams,
                    'new_params': new_hyperparams,
                    'confidence': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update hyperparameters
                self.rl_hyperparams = new_hyperparams
                
                # Update PPO model if it has a method for this
                if hasattr(self.ppo_model, 'update_hyperparameters'):
                    self.ppo_model.update_hyperparameters(new_hyperparams)
        
        except Exception as e:
            logger.error(f"Error optimizing RL hyperparameters: {e}")
    
    def _apply_safe_optimizations(self):
        """
        Apply high-confidence optimizations without waiting for manual approval
        """
        try:
            applied_count = 0
            
            # Find high-confidence parameter optimizations
            for recommendation in self.recommendations:
                if (recommendation['type'] == 'parameter_optimization' and 
                    recommendation['confidence'] == 'high' and
                    'applied' not in recommendation):
                    
                    config_id = recommendation['config_id']
                    new_params = recommendation['new_params']
                    
                    # Get the strategy configuration
                    config = StrategyConfiguration.query.get(config_id)
                    if not config:
                        continue
                    
                    # Update parameters
                    old_params = config.parameters
                    config.parameters = new_params
                    
                    # Save changes
                    db.session.commit()
                    
                    # Mark as applied
                    recommendation['applied'] = True
                    recommendation['applied_at'] = datetime.now().isoformat()
                    
                    logger.info(f"Applied high-confidence parameter optimization to {config.strategy_type} "
                                f"strategy for {config.symbol} {config.timeframe}")
                    logger.info(f"Old parameters: {old_params}")
                    logger.info(f"New parameters: {new_params}")
                    
                    # Update the active strategy if it exists
                    strategy_key = f"{config.strategy_type}_{config.symbol}_{config.timeframe}"
                    if self.active_strategies and strategy_key in self.active_strategies:
                        self.active_strategies[strategy_key].parameters = new_params
                        logger.info(f"Updated parameters for active strategy {strategy_key}")
                    
                    applied_count += 1
            
            # Apply adaptive risk management optimizations
            risk_updates = self._optimize_risk_parameters()
            applied_count += risk_updates
            
            if applied_count > 0:
                logger.info(f"Applied {applied_count} optimizations automatically")
            
            return applied_count
            
        except Exception as e:
            logger.error(f"Error applying safe optimizations: {e}")
            db.session.rollback()
            return 0
            
    def _optimize_risk_parameters(self):
        """
        Use the adaptive risk manager to optimize risk parameters for all active strategies.
        
        Returns:
            int: Number of strategies updated
        """
        try:
            # Update market regime first
            self.evolution_manager.detect_market_regime()
            
            # Update risk parameters for all active strategies
            updated_count = self.adaptive_risk_manager.update_strategy_risk_parameters()
            
            if updated_count > 0:
                logger.info(f"Updated risk parameters for {updated_count} strategies based on adaptive risk management")
                
                # Add as recommendations
                for strategy in StrategyConfiguration.query.filter_by(is_active=True).all():
                    strategy_key = f"{strategy.strategy_type}_{strategy.symbol}_{strategy.timeframe}"
                    
                    # If this strategy has risk metadata, add a recommendation
                    if strategy.parameters and "risk_metadata" in strategy.parameters:
                        self.recommendations.append({
                            'type': 'risk_optimization',
                            'strategy_key': strategy_key,
                            'risk_parameters': strategy.parameters.get("risk_metadata", {}),
                            'confidence': 'high',
                            'timestamp': datetime.now().isoformat(),
                            'config_id': strategy.id,
                            'applied': True,
                            'applied_at': datetime.now().isoformat()
                        })
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Error optimizing risk parameters: {e}")
            return 0
    
    def get_recommendations(self, max_count=10, min_confidence=None):
        """
        Get recent recommendations from the supervisor
        
        Args:
            max_count (int): Maximum number of recommendations to return
            min_confidence (str): Minimum confidence level ('low', 'medium', 'high')
            
        Returns:
            list: Recent recommendations
        """
        try:
            # Sort by timestamp, most recent first
            sorted_recommendations = sorted(
                self.recommendations,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            
            # Filter by confidence if specified
            if min_confidence:
                confidence_levels = {'low': 0, 'medium': 1, 'high': 2}
                min_level = confidence_levels.get(min_confidence.lower(), 0)
                
                filtered = [
                    r for r in sorted_recommendations
                    if confidence_levels.get(r.get('confidence', 'low').lower(), 0) >= min_level
                ]
                
                return filtered[:max_count]
            
            return sorted_recommendations[:max_count]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def apply_recommendation(self, recommendation_id):
        """
        Apply a specific recommendation by ID
        
        Args:
            recommendation_id (int): Index of the recommendation in the list
            
        Returns:
            bool: True if applied successfully, False otherwise
        """
        try:
            if recommendation_id < 0 or recommendation_id >= len(self.recommendations):
                logger.warning(f"Invalid recommendation ID: {recommendation_id}")
                return False
            
            recommendation = self.recommendations[recommendation_id]
            
            # Check if already applied
            if recommendation.get('applied', False):
                logger.warning(f"Recommendation {recommendation_id} already applied")
                return False
            
            if recommendation['type'] == 'parameter_optimization':
                config_id = recommendation['config_id']
                new_params = recommendation['new_params']
                
                # Get the strategy configuration
                config = StrategyConfiguration.query.get(config_id)
                if not config:
                    logger.warning(f"Strategy configuration {config_id} not found")
                    return False
                
                # Update parameters
                config.parameters = new_params
                
                # Save changes
                db.session.commit()
                
                # Mark as applied
                recommendation['applied'] = True
                recommendation['applied_at'] = datetime.now().isoformat()
                
                logger.info(f"Applied parameter optimization to {config.strategy_type} "
                            f"strategy for {config.symbol} {config.timeframe}")
                
                # Update the active strategy if it exists
                strategy_key = f"{config.strategy_type}_{config.symbol}_{config.timeframe}"
                if self.active_strategies and strategy_key in self.active_strategies:
                    self.active_strategies[strategy_key].parameters = new_params
                
                return True
                
            elif recommendation['type'] == 'weight_optimization':
                configs = recommendation['configs']
                new_weights = recommendation['new_weights']
                
                for config_id in configs:
                    config = StrategyConfiguration.query.get(config_id)
                    if not config:
                        continue
                    
                    if config.strategy_type in new_weights:
                        config.weight = new_weights[config.strategy_type]
                
                # Save changes
                db.session.commit()
                
                # Mark as applied
                recommendation['applied'] = True
                recommendation['applied_at'] = datetime.now().isoformat()
                
                logger.info(f"Applied weight optimization for {recommendation['symbol_timeframe']}")
                
                return True
            
            elif recommendation['type'] == 'risk_optimization':
                config_id = recommendation['config_id']
                
                # Get the strategy configuration
                config = StrategyConfiguration.query.get(config_id)
                if not config:
                    logger.warning(f"Strategy configuration {config_id} not found")
                    return False
                
                # Generate new risk parameters
                risk_params = self.adaptive_risk_manager.get_risk_parameters(
                    strategy_type=config.strategy_type,
                    symbol=config.symbol
                )
                
                # Update strategy parameters with risk adjustments
                params = config.parameters.copy()
                params["stop_loss"] = risk_params["stop_loss"]
                params["take_profit"] = risk_params["take_profit"]
                
                # Add risk metadata
                if "risk_metadata" not in params:
                    params["risk_metadata"] = {}
                    
                params["risk_metadata"] = {
                    "position_size": risk_params["position_size"],
                    "daily_loss_limit": risk_params["daily_loss_limit"],
                    "market_volatility": risk_params["market_volatility"],
                    "last_updated": datetime.now().isoformat()
                }
                
                # Save the updated parameters
                config.parameters = params
                db.session.commit()
                
                # Mark as applied
                recommendation['applied'] = True
                recommendation['applied_at'] = datetime.now().isoformat()
                
                logger.info(f"Applied risk optimization to {config.strategy_type} "
                            f"strategy for {config.symbol} {config.timeframe}")
                
                return True
                
            else:
                logger.warning(f"Cannot automatically apply recommendation of type {recommendation['type']}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying recommendation: {e}")
            db.session.rollback()
            return False
    
    def export_recommendations(self, filepath=None):
        """
        Export recommendations to a JSON file
        
        Args:
            filepath (str): Path to save the file
            
        Returns:
            str: Path to the saved file or JSON string if no path provided
        """
        try:
            # Convert recommendations to serializable format
            export_data = []
            
            for rec in self.recommendations:
                # Create a copy to avoid modifying the original
                export_rec = rec.copy()
                
                # Convert any non-serializable objects to strings
                for key, value in export_rec.items():
                    if isinstance(value, (datetime, np.ndarray)):
                        export_rec[key] = str(value)
                
                export_data.append(export_rec)
            
            # Export as JSON
            json_data = json.dumps(export_data, indent=2)
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(json_data)
                return filepath
            else:
                return json_data
                
        except Exception as e:
            logger.error(f"Error exporting recommendations: {e}")
            return None