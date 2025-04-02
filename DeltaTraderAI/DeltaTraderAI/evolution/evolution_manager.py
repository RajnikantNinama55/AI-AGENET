"""
Evolution Manager for Trading Bot

This module manages the evolutionary aspects of the trading bot:
1. Tracks historical performance of strategies
2. Implements genetic algorithm concepts for strategy evolution
3. Manages the continuous learning process of the RL model
4. Adapts to changing market conditions through periodic evaluation
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from copy import deepcopy

from models import StrategyConfiguration, TradingSession, Trade
from app import db
from config import Config
from rl.ppo_model import PPOModel

logger = logging.getLogger(__name__)

class EvolutionManager:
    """
    Evolution Manager that handles the continuous improvement of trading strategies.
    
    This manager implements evolutionary algorithms to evolve trading strategies
    over time, adapt to changing market conditions, and continuously optimize
    the reinforcement learning models.
    """
    
    def __init__(self, supervisor=None, api_client=None, backtester=None):
        """
        Initialize the evolution manager
        
        Args:
            supervisor: Reference to supervisor agent
            api_client: API client for fetching data
            backtester: Backtester for evaluating strategies
        """
        self.supervisor = supervisor
        self.api_client = api_client
        self.backtester = backtester
        self.evolution_history = []
        self.generation = 0
        self.population_size = 20
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        self.selection_pressure = 0.3  # Percentage of top performers to select
        self.strategy_lifetimes = {}  # Track how long strategies have been active
        self.market_regime = "unknown"  # Current market regime (trending, ranging, volatile)
        self.regime_history = []  # Track market regime changes
        
        # Load previous evolution data if available
        self._load_evolution_history()
        
        logger.info("Evolution Manager initialized")
    
    def _load_evolution_history(self):
        """Load previous evolution history if available"""
        evolution_file = os.path.join(Config.DATA_DIRECTORY, "evolution_history.json")
        try:
            if os.path.exists(evolution_file):
                with open(evolution_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_history = data.get('history', [])
                    self.generation = data.get('generation', 0)
                    self.regime_history = data.get('regime_history', [])
                logger.info(f"Loaded evolution history: Generation {self.generation}")
        except Exception as e:
            logger.error(f"Error loading evolution history: {e}")
    
    def _save_evolution_history(self):
        """Save evolution history to file"""
        evolution_file = os.path.join(Config.DATA_DIRECTORY, "evolution_history.json")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(evolution_file), exist_ok=True)
            
            with open(evolution_file, 'w') as f:
                json.dump({
                    'history': self.evolution_history,
                    'generation': self.generation,
                    'regime_history': self.regime_history
                }, f)
            logger.info(f"Saved evolution history: Generation {self.generation}")
        except Exception as e:
            logger.error(f"Error saving evolution history: {e}")
    
    def detect_market_regime(self, symbol="BTC-USDT", timeframe="1h", lookback_days=30):
        """
        Detect the current market regime based on price action
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            lookback_days: Number of days to analyze
            
        Returns:
            str: Market regime ('trending', 'ranging', 'volatile')
        """
        try:
            # Fetch historical data
            if self.backtester is None:
                logger.warning("Backtester not available for regime detection")
                return "unknown"
            
            # Simply pass None for start_time and let backtester handle the dates
            # The backtester will default to the past 30 days
            historical_data = self.backtester.fetch_historical_data(
                symbol=symbol, 
                timeframe=timeframe,
                start_time=None,
                end_time=None
            )
            
            if historical_data is None or len(historical_data) < 20:
                logger.warning(f"Insufficient data for regime detection: {len(historical_data) if historical_data is not None else 0} candles")
                return "unknown"
            
            # Calculate key metrics
            # 1. Trend strength using ADX
            high = historical_data['high'].values
            low = historical_data['low'].values
            close = historical_data['close'].values
            
            # Calculate True Range
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Calculate Directional Movement
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate ADX (simplified)
            period = 14
            smoothed_plus_dm = np.mean(plus_dm[-period:])
            smoothed_minus_dm = np.mean(minus_dm[-period:])
            smoothed_tr = np.mean(tr[-period:])
            
            plus_di = 100 * smoothed_plus_dm / smoothed_tr if smoothed_tr > 0 else 0
            minus_di = 100 * smoothed_minus_dm / smoothed_tr if smoothed_tr > 0 else 0
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            adx = np.mean(np.array([dx])) # Simplified ADX
            
            # 2. Volatility using ATR
            atr = np.mean(tr[-period:])
            atr_percent = atr / close[-1] * 100  # ATR as percentage of price
            
            # 3. Range-bound detection using Bollinger Bands
            rolling_mean = np.mean(close[-20:])
            rolling_std = np.std(close[-20:])
            
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            
            # Get closing prices within bands
            closes_within_bands = np.sum((close[-20:] >= lower_band) & (close[-20:] <= upper_band))
            band_percentage = closes_within_bands / 20
            
            # Determine regime
            if adx > 25:
                regime = "trending"
            elif band_percentage > 0.8 and atr_percent < 3:
                regime = "ranging"
            elif atr_percent > 5:
                regime = "volatile"
            else:
                regime = "mixed"
            
            # Check if regime has changed
            if regime != self.market_regime:
                self.regime_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "old_regime": self.market_regime,
                    "new_regime": regime,
                    "adx": float(adx),
                    "atr_percent": float(atr_percent),
                    "band_percentage": float(band_percentage)
                })
                logger.info(f"Market regime changed from {self.market_regime} to {regime}")
            
            self.market_regime = regime
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def get_optimal_strategy_mix(self):
        """
        Get optimal strategy mix based on current market regime
        
        Returns:
            dict: Optimal strategy mix with weights
        """
        if self.market_regime == "trending":
            return {
                "trend_following": 0.5,
                "momentum": 0.3,
                "breakout": 0.15,
                "mean_reversion": 0.05
            }
        elif self.market_regime == "ranging":
            return {
                "mean_reversion": 0.6,
                "trend_following": 0.2,
                "momentum": 0.1,
                "breakout": 0.1
            }
        elif self.market_regime == "volatile":
            return {
                "breakout": 0.4,
                "momentum": 0.3,
                "mean_reversion": 0.2,
                "trend_following": 0.1
            }
        else:  # mixed or unknown
            return {
                "trend_following": 0.25,
                "mean_reversion": 0.25,
                "breakout": 0.25,
                "momentum": 0.25
            }
    
    def evolve_strategies(self, strategies=None):
        """
        Evolve a population of strategies using genetic algorithm
        
        Args:
            strategies: Current active strategies (if None, load from database)
            
        Returns:
            list: Evolved strategy configurations
        """
        try:
            # Get current active strategies if not provided
            if strategies is None:
                strategies = StrategyConfiguration.query.filter_by(is_active=True).all()
            
            if not strategies:
                logger.warning("No active strategies found for evolution")
                return []
            
            # Increment generation counter
            self.generation += 1
            logger.info(f"Starting evolution generation {self.generation}")
            
            # Step 1: Evaluate fitness of current population
            strategy_fitness = self._evaluate_strategy_fitness(strategies)
            
            # Step 2: Selection - choose top performers
            selected_strategies = self._selection(strategies, strategy_fitness)
            
            # Step 3: Crossover - create new strategies from selected ones
            new_strategies = self._crossover(selected_strategies)
            
            # Step 4: Mutation - randomly modify some strategies
            evolved_strategies = self._mutation(new_strategies)
            
            # Step 5: Record evolution
            self._record_evolution(strategies, evolved_strategies, strategy_fitness)
            
            # Save evolution history
            self._save_evolution_history()
            
            return evolved_strategies
            
        except Exception as e:
            logger.error(f"Error during strategy evolution: {e}")
            return strategies  # Return original strategies on error
    
    def _evaluate_strategy_fitness(self, strategies):
        """
        Evaluate the fitness of each strategy
        
        Args:
            strategies: List of strategy configurations
            
        Returns:
            dict: Strategy ID to fitness value mapping
        """
        fitness = {}
        
        for strategy in strategies:
            # Get recent performance metrics
            trades = Trade.query.join(TradingSession).filter(
                TradingSession.user_id == strategy.user_id,
                Trade.strategy == strategy.strategy_type,
                Trade.entry_time >= datetime.now() - timedelta(days=30)
            ).all()
            
            # Calculate performance metrics
            if trades:
                win_count = sum(1 for t in trades if t.pnl_percentage and t.pnl_percentage > 0)
                win_rate = win_count / len(trades) if trades else 0
                avg_profit = sum(t.pnl_percentage for t in trades if t.pnl_percentage) / len(trades) if trades else 0
                max_drawdown = abs(min(t.pnl_percentage for t in trades if t.pnl_percentage) or 0)
                
                # Calculate Sharpe-like ratio
                returns = [t.pnl_percentage for t in trades if t.pnl_percentage]
                returns_std = np.std(returns) if len(returns) > 1 else 1
                sharpe = (np.mean(returns) / returns_std) if returns_std > 0 else 0
                
                # Combine metrics for fitness
                raw_fitness = (
                    win_rate * 0.3 + 
                    avg_profit * 0.4 + 
                    (sharpe * 0.2 if sharpe > 0 else 0) - 
                    (max_drawdown * 0.1 if max_drawdown > 0 else 0)
                )
                
                # Adjust based on strategy lifetime (favor strategies that have proven themselves)
                strategy_key = f"{strategy.id}"
                lifetime = self.strategy_lifetimes.get(strategy_key, 0) + 1
                self.strategy_lifetimes[strategy_key] = lifetime
                
                # Age bonus: strategies that survived multiple generations get a small boost
                age_bonus = min(0.2, lifetime / 10)  # Max 20% bonus for 10+ generation lifetime
                
                # Apply regime-specific adjustment
                regime_multiplier = self._get_regime_multiplier(strategy.strategy_type)
                
                # Final fitness
                fitness[strategy.id] = max(0, raw_fitness * (1 + age_bonus) * regime_multiplier)
            else:
                # No trade data - assign baseline fitness with small random variation
                baseline = 0.1 + (random.random() * 0.1)
                fitness[strategy.id] = baseline
        
        return fitness
    
    def _get_regime_multiplier(self, strategy_type):
        """Get fitness multiplier based on strategy type and current market regime"""
        if self.market_regime == "trending":
            multipliers = {
                "trend_following": 1.3,
                "momentum": 1.2,
                "breakout": 1.1,
                "mean_reversion": 0.8
            }
        elif self.market_regime == "ranging":
            multipliers = {
                "mean_reversion": 1.3,
                "trend_following": 0.8,
                "momentum": 0.9,
                "breakout": 1.0
            }
        elif self.market_regime == "volatile":
            multipliers = {
                "breakout": 1.3,
                "momentum": 1.2,
                "mean_reversion": 1.0,
                "trend_following": 0.7
            }
        else:  # mixed or unknown
            multipliers = {
                "trend_following": 1.0,
                "mean_reversion": 1.0,
                "breakout": 1.0,
                "momentum": 1.0
            }
        
        return multipliers.get(strategy_type, 1.0)
    
    def _selection(self, strategies, fitness):
        """
        Select top performing strategies
        
        Args:
            strategies: List of strategy configurations
            fitness: Fitness values for each strategy
            
        Returns:
            list: Selected strategies
        """
        # Sort strategies by fitness
        sorted_strategies = sorted(strategies, key=lambda s: fitness.get(s.id, 0), reverse=True)
        
        # Select top performers
        selection_count = max(2, int(len(strategies) * self.selection_pressure))
        return sorted_strategies[:selection_count]
    
    def _crossover(self, parent_strategies):
        """
        Perform crossover between selected strategies
        
        Args:
            parent_strategies: List of selected strategies
            
        Returns:
            list: New strategies after crossover
        """
        # Create offspring while maintaining original population size
        offspring = parent_strategies.copy()  # Keep parents
        needed_offspring = max(0, self.population_size - len(offspring))
        
        for _ in range(needed_offspring):
            if len(parent_strategies) < 2:
                # Not enough parents, clone with small mutations
                if parent_strategies:
                    parent = random.choice(parent_strategies)
                    new_strategy = self._clone_with_small_mutation(parent)
                    offspring.append(new_strategy)
                continue
            
            # Select two distinct parents
            parent1, parent2 = random.sample(parent_strategies, 2)
            
            # Crossover their parameters
            if random.random() < self.crossover_rate:
                # Create a new strategy by mixing parameters
                new_strategy = self._crossover_parameters(parent1, parent2)
                offspring.append(new_strategy)
        
        return offspring
    
    def _crossover_parameters(self, parent1, parent2):
        """Mix parameters from two parent strategies"""
        new_params = {}
        strategy_type = parent1.strategy_type  # Keep same type
        
        # Get parameters based on strategy type
        if strategy_type == "trend_following":
            # Mix parameters with weighted average
            weight = random.random()  # Blend factor between two parents
            p1_params = parent1.parameters
            p2_params = parent2.parameters
            
            new_params = {
                "short_window": int(p1_params["short_window"] * weight + p2_params["short_window"] * (1 - weight)),
                "long_window": int(p1_params["long_window"] * weight + p2_params["long_window"] * (1 - weight)),
                "take_profit": p1_params["take_profit"] * weight + p2_params["take_profit"] * (1 - weight),
                "stop_loss": p1_params["stop_loss"] * weight + p2_params["stop_loss"] * (1 - weight)
            }
        elif strategy_type == "mean_reversion":
            p1_params = parent1.parameters
            p2_params = parent2.parameters
            
            # Mix parameters with one-point crossover (some from parent1, some from parent2)
            new_params = {
                "rsi_period": p1_params["rsi_period"],
                "oversold_threshold": p2_params["oversold_threshold"],
                "overbought_threshold": p2_params["overbought_threshold"],
                "take_profit": p1_params["take_profit"],
                "stop_loss": p2_params["stop_loss"]
            }
        elif strategy_type == "breakout":
            p1_params = parent1.parameters
            p2_params = parent2.parameters
            
            # Mix parameters, taking best of each (based on theory)
            new_params = {
                "period": min(p1_params["period"], p2_params["period"]),  # Shorter period is more responsive
                "atr_period": max(p1_params["atr_period"], p2_params["atr_period"]),  # Longer ATR is more stable
                "atr_multiplier": (p1_params["atr_multiplier"] + p2_params["atr_multiplier"]) / 2,  # Average
                "take_profit": max(p1_params["take_profit"], p2_params["take_profit"]),  # Higher take profit
                "stop_loss": min(p1_params["stop_loss"], p2_params["stop_loss"])  # Tighter stop loss
            }
        elif strategy_type == "momentum":
            p1_params = parent1.parameters
            p2_params = parent2.parameters
            
            # Uniform crossover (random choice for each parameter)
            new_params = {
                "period": p1_params["period"] if random.random() < 0.5 else p2_params["period"],
                "threshold": p1_params["threshold"] if random.random() < 0.5 else p2_params["threshold"],
                "take_profit": p1_params["take_profit"] if random.random() < 0.5 else p2_params["take_profit"],
                "stop_loss": p1_params["stop_loss"] if random.random() < 0.5 else p2_params["stop_loss"]
            }
        
        # Create new strategy object with crossed parameters
        new_strategy = StrategyConfiguration(
            user_id=parent1.user_id,
            strategy_type=strategy_type,
            symbol=parent1.symbol,
            timeframe=parent1.timeframe,
            parameters=new_params,
            weight=parent1.weight,
            is_active=True
        )
        
        return new_strategy
    
    def _clone_with_small_mutation(self, parent):
        """Clone a strategy with a small mutation"""
        new_params = deepcopy(parent.parameters)
        
        # Apply small mutation to parameters
        if parent.strategy_type == "trend_following":
            new_params["short_window"] = max(5, int(new_params["short_window"] * (0.9 + 0.2 * random.random())))
            new_params["long_window"] = max(10, int(new_params["long_window"] * (0.9 + 0.2 * random.random())))
        elif parent.strategy_type == "mean_reversion":
            new_params["rsi_period"] = max(5, int(new_params["rsi_period"] * (0.9 + 0.2 * random.random())))
            new_params["oversold_threshold"] = max(10, min(40, new_params["oversold_threshold"] + random.randint(-5, 5)))
            new_params["overbought_threshold"] = min(90, max(60, new_params["overbought_threshold"] + random.randint(-5, 5)))
        elif parent.strategy_type == "breakout":
            new_params["period"] = max(5, int(new_params["period"] * (0.9 + 0.2 * random.random())))
            new_params["atr_multiplier"] = max(0.5, new_params["atr_multiplier"] * (0.9 + 0.2 * random.random()))
        elif parent.strategy_type == "momentum":
            new_params["period"] = max(3, int(new_params["period"] * (0.9 + 0.2 * random.random())))
            new_params["threshold"] = max(0.1, new_params["threshold"] * (0.9 + 0.2 * random.random()))
        
        # Mutate take profit and stop loss for all strategies
        new_params["take_profit"] = max(0.5, new_params["take_profit"] * (0.9 + 0.2 * random.random()))
        new_params["stop_loss"] = max(0.5, new_params["stop_loss"] * (0.9 + 0.2 * random.random()))
        
        # Create new strategy object
        new_strategy = StrategyConfiguration(
            user_id=parent.user_id,
            strategy_type=parent.strategy_type,
            symbol=parent.symbol,
            timeframe=parent.timeframe,
            parameters=new_params,
            weight=parent.weight,
            is_active=True
        )
        
        return new_strategy
    
    def _mutation(self, strategies):
        """
        Apply random mutations to strategies
        
        Args:
            strategies: List of strategies
            
        Returns:
            list: Mutated strategies
        """
        mutated_strategies = []
        
        for strategy in strategies:
            # Randomly decide whether to mutate this strategy
            if random.random() < self.mutation_rate:
                # Deep copy the strategy parameters
                params = deepcopy(strategy.parameters) if strategy.parameters else {}
                
                # Apply mutation based on strategy type
                if strategy.strategy_type == "trend_following":
                    # Mutate moving average windows
                    if "short_window" in params and random.random() < 0.5:
                        params["short_window"] = max(5, int(params["short_window"] * (0.7 + 0.6 * random.random())))
                    if "long_window" in params and random.random() < 0.5:
                        min_long = params.get("short_window", 10) + 5
                        params["long_window"] = max(min_long, int(params["long_window"] * (0.7 + 0.6 * random.random())))
                
                elif strategy.strategy_type == "mean_reversion":
                    # Mutate RSI parameters
                    if "rsi_period" in params and random.random() < 0.5:
                        params["rsi_period"] = max(2, int(params["rsi_period"] * (0.7 + 0.6 * random.random())))
                    if "oversold_threshold" in params and random.random() < 0.5:
                        params["oversold_threshold"] = max(10, min(40, params["oversold_threshold"] + random.randint(-10, 10)))
                    if "overbought_threshold" in params and random.random() < 0.5:
                        params["overbought_threshold"] = min(90, max(60, params["overbought_threshold"] + random.randint(-10, 10)))
                
                elif strategy.strategy_type == "breakout":
                    # Mutate breakout parameters
                    if "period" in params and random.random() < 0.5:
                        params["period"] = max(5, int(params["period"] * (0.7 + 0.6 * random.random())))
                    if "atr_period" in params and random.random() < 0.5:
                        params["atr_period"] = max(3, int(params["atr_period"] * (0.7 + 0.6 * random.random())))
                    if "atr_multiplier" in params and random.random() < 0.5:
                        params["atr_multiplier"] = max(0.5, min(5.0, params["atr_multiplier"] * (0.7 + 0.6 * random.random())))
                
                elif strategy.strategy_type == "momentum":
                    # Mutate momentum parameters
                    if "period" in params and random.random() < 0.5:
                        params["period"] = max(3, int(params["period"] * (0.7 + 0.6 * random.random())))
                    if "threshold" in params and random.random() < 0.5:
                        params["threshold"] = max(0.1, min(2.0, params["threshold"] * (0.7 + 0.6 * random.random())))
                
                # Mutate take profit and stop loss for all strategies
                if "take_profit" in params and random.random() < 0.5:
                    params["take_profit"] = max(0.5, min(10.0, params["take_profit"] * (0.7 + 0.6 * random.random())))
                if "stop_loss" in params and random.random() < 0.5:
                    params["stop_loss"] = max(0.5, min(10.0, params["stop_loss"] * (0.7 + 0.6 * random.random())))
                
                # Create a new mutated strategy
                mutated = StrategyConfiguration(
                    user_id=strategy.user_id,
                    strategy_type=strategy.strategy_type,
                    symbol=strategy.symbol,
                    timeframe=strategy.timeframe,
                    parameters=params,
                    weight=strategy.weight,
                    is_active=True
                )
                mutated_strategies.append(mutated)
            else:
                # Keep original strategy
                mutated_strategies.append(strategy)
        
        return mutated_strategies
    
    def _record_evolution(self, original_strategies, evolved_strategies, fitness):
        """Record evolution details for history and analysis"""
        evolution_record = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.market_regime,
            "original_count": len(original_strategies),
            "evolved_count": len(evolved_strategies),
            "top_fitness": max(fitness.values()) if fitness else 0,
            "avg_fitness": sum(fitness.values()) / len(fitness) if fitness else 0,
            "strategies": []
        }
        
        # Record details of evolved strategies
        for strategy in evolved_strategies:
            evolution_record["strategies"].append({
                "strategy_type": strategy.strategy_type,
                "parameters": strategy.parameters,
                "fitness": fitness.get(strategy.id, 0)
            })
        
        self.evolution_history.append(evolution_record)
    
    def evolve_rl_model(self, ppo_model=None):
        """
        Evolve the RL model hyperparameters
        
        Args:
            ppo_model: The PPO model to evolve (if None, create new one)
            
        Returns:
            dict: Updated hyperparameters
        """
        if ppo_model is None:
            ppo_model = PPOModel()
        
        # Get current hyperparameters
        current = {
            'learning_rate': ppo_model.learning_rate,
            'gamma': ppo_model.gamma,
            'clip_range': ppo_model.clip_range,
            'batch_size': ppo_model.batch_size,
            'n_steps': ppo_model.n_steps
        }
        
        # Adjust based on recent performance and market regime
        # For simplicity, we'll adjust based on market regime
        new_params = {}
        
        if self.market_regime == "trending":
            # In trending markets, favor longer-term rewards
            new_params['gamma'] = min(0.99, current['gamma'] * 1.01)  # Increase discount factor
            new_params['learning_rate'] = max(0.0001, current['learning_rate'] * 0.95)  # Slower learning
        
        elif self.market_regime == "ranging":
            # In ranging markets, focus on exploitation
            new_params['clip_range'] = max(0.1, current['clip_range'] * 0.9)  # Reduced clip range
            new_params['learning_rate'] = max(0.0001, current['learning_rate'] * 0.98)  # Slightly slower learning
        
        elif self.market_regime == "volatile":
            # In volatile markets, favor exploration
            new_params['clip_range'] = min(0.3, current['clip_range'] * 1.1)  # Increased clip range
            new_params['learning_rate'] = min(0.001, current['learning_rate'] * 1.05)  # Faster learning
        
        # Update model hyperparameters
        updated = ppo_model.update_hyperparameters(**new_params)
        
        logger.info(f"Evolved RL model hyperparameters: {updated}")
        return updated
    
    def create_strategy_diversity(self, user_id, base_symbols=None, timeframes=None):
        """
        Create a diverse set of strategies across different symbols and timeframes
        
        Args:
            user_id: User ID for the strategies
            base_symbols: List of symbols to use (defaults to config)
            timeframes: List of timeframes to use (defaults to config)
            
        Returns:
            list: Created strategy configurations
        """
        try:
            # Set defaults if parameters are None
            if base_symbols is None:
                base_symbols = Config.DEFAULT_SYMBOLS
                
            if timeframes is None:
                timeframes = Config.DEFAULT_TIMEFRAMES
            
            if not base_symbols or not timeframes:
                logger.warning("No symbols or timeframes provided for strategy diversity")
                base_symbols = ["BTC-USDT", "ETH-USDT"] if not base_symbols else base_symbols
                timeframes = ["1h", "4h"] if not timeframes else timeframes
            
            # Select subset of symbols and timeframes to avoid too many strategies
            symbols = random.sample(base_symbols, min(2, len(base_symbols)))
            selected_timeframes = random.sample(timeframes, min(2, len(timeframes)))
            
            strategies = []
            strategy_types = ["trend_following", "mean_reversion", "breakout", "momentum"]
            
            for symbol in symbols:
                for timeframe in selected_timeframes:
                    # Create one of each strategy type
                    for strategy_type in strategy_types:
                        # Get default parameters with appropriate defaults if not available
                        params = {}
                        
                        if strategy_type == "trend_following":
                            params = getattr(Config, "TREND_FOLLOWING_DEFAULTS", {
                                "short_window": 20,
                                "long_window": 50,
                                "take_profit": 3.0,
                                "stop_loss": 2.0
                            }).copy()
                        elif strategy_type == "mean_reversion":
                            params = getattr(Config, "MEAN_REVERSION_DEFAULTS", {
                                "rsi_period": 14,
                                "oversold_threshold": 30,
                                "overbought_threshold": 70,
                                "take_profit": 2.0,
                                "stop_loss": 2.0
                            }).copy()
                        elif strategy_type == "breakout":
                            params = getattr(Config, "BREAKOUT_DEFAULTS", {
                                "period": 20,
                                "atr_period": 14,
                                "atr_multiplier": 2.0,
                                "take_profit": 4.0,
                                "stop_loss": 2.0
                            }).copy()
                        elif strategy_type == "momentum":
                            params = getattr(Config, "MOMENTUM_DEFAULTS", {
                                "period": 14,
                                "threshold": 0.5,
                                "take_profit": 3.0,
                                "stop_loss": 2.0
                            }).copy()
                        
                        # Apply small random variations
                        for key in params:
                            # Add 20% random variation
                            params[key] = params[key] * (0.9 + 0.2 * random.random())
                            
                            # Ensure integers for window parameters
                            if key in ["short_window", "long_window", "period", "rsi_period", "atr_period"]:
                                params[key] = max(2, int(params[key]))
                        
                        # Create strategy configuration
                        strategy = StrategyConfiguration(
                            user_id=user_id,
                            strategy_type=strategy_type,
                            symbol=symbol,
                            timeframe=timeframe,
                            parameters=params,
                            weight=0.25,  # Equal weights initially
                            is_active=True
                        )
                        
                        strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error creating strategy diversity: {e}")
            return []
    
    def get_evolution_history(self, max_records=10):
        """
        Get recent evolution history
        
        Args:
            max_records: Maximum number of records to return
            
        Returns:
            list: Recent evolution history records
        """
        # Return most recent records
        return self.evolution_history[-max_records:] if self.evolution_history else []
    
    def get_market_regime_history(self, max_records=10):
        """
        Get market regime change history
        
        Args:
            max_records: Maximum number of records to return
            
        Returns:
            list: Market regime change history
        """
        return self.regime_history[-max_records:] if self.regime_history else []