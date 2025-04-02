"""
Adaptive Risk Management for Trading Bot

This module implements adaptive risk management that evolves based on:
1. Market volatility and regime
2. Recent trading performance
3. Portfolio drawdown
4. Strategy-specific risk profiles
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import math

from models import Trade, TradingSession, StrategyConfiguration
from app import db
from config import Config

logger = logging.getLogger(__name__)

class AdaptiveRiskManager:
    """
    Adaptive Risk Manager that dynamically adjusts risk parameters.
    
    This manager automatically adapts risk parameters such as position size,
    stop loss distances, and daily loss limits based on market conditions,
    recent performance, and portfolio health.
    """
    
    def __init__(self, evolution_manager=None, api_client=None):
        """
        Initialize the adaptive risk manager
        
        Args:
            evolution_manager: Reference to evolution manager
            api_client: API client for market data
        """
        self.evolution_manager = evolution_manager
        self.api_client = api_client
        
        # Default risk parameters
        self.base_position_size = 0.02  # 2% of capital per trade
        self.max_position_size = 0.05   # 5% maximum
        self.min_position_size = 0.005  # 0.5% minimum
        
        self.base_daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_daily_loss_limit = 0.10   # 10% maximum
        self.min_daily_loss_limit = 0.02   # 2% minimum
        
        # Strategy-specific risk factors (multipliers)
        self.strategy_risk_factors = {
            "trend_following": 1.0,
            "mean_reversion": 1.2,  # Higher risk due to counter-trend nature
            "breakout": 1.1,        # Slightly higher risk for breakouts
            "momentum": 0.9         # Lower risk due to trend alignment
        }
        
        # Performance tracking
        self.recent_win_rate = 0.5  # Default initial value
        self.recent_sharpe = 0.0    # Default initial value
        self.recent_max_drawdown = 0.0  # Default initial value
        
        # Market volatility tracking
        self.market_volatility = 0.0  # Current market volatility
        
        # Risk allocation by market regime
        self.regime_risk_allocation = {
            "trending": {
                "trend_following": 1.2,
                "momentum": 1.1,
                "breakout": 0.9,
                "mean_reversion": 0.7
            },
            "ranging": {
                "trend_following": 0.7,
                "momentum": 0.8,
                "breakout": 0.9,
                "mean_reversion": 1.3
            },
            "volatile": {
                "trend_following": 0.6,
                "momentum": 0.8,
                "breakout": 1.2,
                "mean_reversion": 0.8
            },
            "mixed": {
                "trend_following": 1.0,
                "momentum": 1.0,
                "breakout": 1.0,
                "mean_reversion": 1.0
            },
            "unknown": {
                "trend_following": 0.8,
                "momentum": 0.8,
                "breakout": 0.8,
                "mean_reversion": 0.8
            }
        }
        
        # Initialize with current data
        self._update_performance_metrics()
        
        logger.info("Adaptive Risk Manager initialized")
    
    def _update_performance_metrics(self):
        """Update performance metrics from recent trading activity"""
        try:
            # Get trades from the last 30 days
            recent_trades = Trade.query.filter(
                Trade.entry_time >= datetime.now() - timedelta(days=30)
            ).all()
            
            if not recent_trades:
                logger.info("No recent trades found for performance metrics")
                return
            
            # Calculate win rate
            win_count = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
            self.recent_win_rate = win_count / len(recent_trades) if recent_trades else 0.5
            
            # Calculate returns for Sharpe ratio
            returns = []
            for trade in recent_trades:
                if trade.pnl_percentage is not None:
                    returns.append(trade.pnl_percentage / 100.0)  # Convert to decimal
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0.01
                self.recent_sharpe = avg_return / std_return if std_return > 0 else 0
            
            # Calculate max drawdown
            # Get trading sessions from the last 30 days
            recent_sessions = TradingSession.query.filter(
                TradingSession.start_time >= datetime.now() - timedelta(days=30)
            ).all()
            
            if recent_sessions:
                self.recent_max_drawdown = max((s.max_drawdown or 0) for s in recent_sessions)
            
            logger.info(f"Updated performance metrics: Win rate: {self.recent_win_rate:.2f}, "
                       f"Sharpe: {self.recent_sharpe:.2f}, Max drawdown: {self.recent_max_drawdown:.2f}")
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def update_market_volatility(self, symbol="BTC-USDT", lookback_days=10):
        """
        Update market volatility metric based on recent price action
        
        Args:
            symbol: Trading pair symbol
            lookback_days: Number of days to analyze
            
        Returns:
            float: Updated volatility (as decimal)
        """
        try:
            if not self.api_client:
                logger.warning("API client not available for volatility calculation")
                return self.market_volatility
            
            # Get recent candle data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            candles = self.api_client.get_candles(
                symbol=symbol,
                resolution="1d",
                start_time=int(start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000)
            )
            
            if not candles or len(candles) < 5:
                logger.warning(f"Insufficient candle data for volatility calculation: {len(candles) if candles else 0} candles")
                return self.market_volatility
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Calculate daily returns
            df['close'] = df['close'].astype(float)
            df['returns'] = df['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = df['returns'].std()
            
            # Scale volatility by market expectations (annualized)
            annualized_vol = volatility * math.sqrt(365)
            
            # Update volatility metric
            self.market_volatility = float(annualized_vol)
            
            logger.info(f"Updated market volatility: {self.market_volatility:.4f}")
            return self.market_volatility
            
        except Exception as e:
            logger.error(f"Error updating market volatility: {e}")
            return self.market_volatility
    
    def get_position_size(self, strategy_type, symbol=None):
        """
        Calculate adaptive position size for a given strategy and symbol
        
        Args:
            strategy_type: Type of trading strategy
            symbol: Trading pair symbol (optional)
            
        Returns:
            float: Position size as percentage of capital
        """
        try:
            # Start with base position size
            position_size = self.base_position_size
            
            # Adjust based on strategy risk factor
            strategy_factor = self.strategy_risk_factors.get(strategy_type, 1.0)
            position_size *= strategy_factor
            
            # Adjust based on market regime
            if self.evolution_manager:
                regime = self.evolution_manager.market_regime
                regime_factor = self.regime_risk_allocation.get(regime, {}).get(strategy_type, 1.0)
                position_size *= regime_factor
            
            # Adjust based on recent performance
            perf_factor = 1.0
            
            # If win rate is high, we can be more aggressive
            if self.recent_win_rate > 0.6:
                perf_factor += 0.2
            elif self.recent_win_rate < 0.4:
                perf_factor -= 0.3
            
            # If Sharpe ratio is good, we can be more aggressive
            if self.recent_sharpe > 1.5:
                perf_factor += 0.15
            elif self.recent_sharpe < 0:
                perf_factor -= 0.25
            
            # If recent drawdown is high, be more conservative
            if self.recent_max_drawdown > 0.1:  # 10% drawdown
                perf_factor -= 0.3
            
            position_size *= perf_factor
            
            # Adjust based on market volatility
            if self.market_volatility > 0:
                # Higher volatility = smaller position size
                vol_factor = 1.0 - min(0.5, self.market_volatility * 3)
                position_size *= vol_factor
            
            # Ensure position size is within limits
            position_size = max(self.min_position_size, min(self.max_position_size, position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.base_position_size
    
    def get_daily_loss_limit(self):
        """
        Calculate adaptive daily loss limit
        
        Returns:
            float: Daily loss limit as percentage of capital
        """
        try:
            # Start with base daily loss limit
            loss_limit = self.base_daily_loss_limit
            
            # Adjust based on recent performance
            perf_factor = 1.0
            
            # If win rate is low, tighten the loss limit
            if self.recent_win_rate < 0.4:
                perf_factor -= 0.2
            
            # If Sharpe ratio is negative, tighten the loss limit
            if self.recent_sharpe < 0:
                perf_factor -= 0.2
            
            # If recent drawdown is high, tighten the loss limit
            if self.recent_max_drawdown > 0.1:  # 10% drawdown
                perf_factor -= 0.3
            
            loss_limit *= perf_factor
            
            # Adjust based on market volatility
            if self.market_volatility > 0:
                # Higher volatility = tighter loss limit
                vol_factor = 1.0 - min(0.5, self.market_volatility * 3)
                loss_limit *= vol_factor
            
            # Ensure loss limit is within bounds
            loss_limit = max(self.min_daily_loss_limit, min(self.max_daily_loss_limit, loss_limit))
            
            return loss_limit
            
        except Exception as e:
            logger.error(f"Error calculating daily loss limit: {e}")
            return self.base_daily_loss_limit
    
    def get_stop_loss_distance(self, strategy_type, symbol=None):
        """
        Calculate adaptive stop loss distance for a given strategy and symbol
        
        Args:
            strategy_type: Type of trading strategy
            symbol: Trading pair symbol (optional)
            
        Returns:
            float: Stop loss distance as percentage from entry
        """
        try:
            # Get strategy default parameters
            if strategy_type == "trend_following":
                base_stop = Config.TREND_FOLLOWING_DEFAULTS.get("stop_loss", 2.0)
            elif strategy_type == "mean_reversion":
                base_stop = Config.MEAN_REVERSION_DEFAULTS.get("stop_loss", 2.0)
            elif strategy_type == "breakout":
                base_stop = Config.BREAKOUT_DEFAULTS.get("stop_loss", 2.0)
            elif strategy_type == "momentum":
                base_stop = Config.MOMENTUM_DEFAULTS.get("stop_loss", 2.0)
            else:
                base_stop = 2.0  # Default
            
            # Adjust based on market regime
            regime_factor = 1.0
            if self.evolution_manager:
                regime = self.evolution_manager.market_regime
                
                if regime == "volatile":
                    regime_factor = 1.3  # Wider stops in volatile markets
                elif regime == "ranging":
                    regime_factor = 0.9  # Tighter stops in ranging markets
            
            # Adjust based on market volatility
            vol_factor = 1.0
            if self.market_volatility > 0:
                # Higher volatility = wider stops
                vol_factor = 1.0 + min(1.0, self.market_volatility * 5)
            
            # Calculate final stop loss distance
            stop_distance = base_stop * regime_factor * vol_factor
            
            return stop_distance
            
        except Exception as e:
            logger.error(f"Error calculating stop loss distance: {e}")
            return 2.0  # Default
    
    def get_take_profit_distance(self, strategy_type, symbol=None):
        """
        Calculate adaptive take profit distance for a given strategy and symbol
        
        Args:
            strategy_type: Type of trading strategy
            symbol: Trading pair symbol (optional)
            
        Returns:
            float: Take profit distance as percentage from entry
        """
        try:
            # Get strategy default parameters
            if strategy_type == "trend_following":
                base_tp = Config.TREND_FOLLOWING_DEFAULTS.get("take_profit", 3.0)
            elif strategy_type == "mean_reversion":
                base_tp = Config.MEAN_REVERSION_DEFAULTS.get("take_profit", 2.0)
            elif strategy_type == "breakout":
                base_tp = Config.BREAKOUT_DEFAULTS.get("take_profit", 4.0)
            elif strategy_type == "momentum":
                base_tp = Config.MOMENTUM_DEFAULTS.get("take_profit", 3.0)
            else:
                base_tp = 3.0  # Default
            
            # Adjust based on market regime
            regime_factor = 1.0
            if self.evolution_manager:
                regime = self.evolution_manager.market_regime
                
                if regime == "trending":
                    regime_factor = 1.3  # Wider take profits in trending markets
                elif regime == "volatile":
                    regime_factor = 0.8  # Tighter take profits in volatile markets
            
            # Adjust based on performance
            perf_factor = 1.0
            
            # If win rate is high, we can be more patient
            if self.recent_win_rate > 0.6:
                perf_factor += 0.2
            
            # Calculate final take profit distance
            take_profit_distance = base_tp * regime_factor * perf_factor
            
            return take_profit_distance
            
        except Exception as e:
            logger.error(f"Error calculating take profit distance: {e}")
            return 3.0  # Default
    
    def get_risk_parameters(self, strategy_type, symbol=None):
        """
        Get complete set of risk parameters for a given strategy and symbol
        
        Args:
            strategy_type: Type of trading strategy
            symbol: Trading pair symbol (optional)
            
        Returns:
            dict: Risk parameters
        """
        # Update performance metrics
        self._update_performance_metrics()
        
        # Update market volatility
        self.update_market_volatility(symbol=symbol if symbol else "BTC-USDT")
        
        # Calculate all risk parameters
        position_size = self.get_position_size(strategy_type, symbol)
        daily_loss_limit = self.get_daily_loss_limit()
        stop_loss = self.get_stop_loss_distance(strategy_type, symbol)
        take_profit = self.get_take_profit_distance(strategy_type, symbol)
        
        return {
            "position_size": position_size,
            "daily_loss_limit": daily_loss_limit,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "market_volatility": self.market_volatility,
            "recent_win_rate": self.recent_win_rate,
            "recent_sharpe": self.recent_sharpe,
            "recent_max_drawdown": self.recent_max_drawdown,
            "market_regime": self.evolution_manager.market_regime if self.evolution_manager else "unknown"
        }
    
    def update_strategy_risk_parameters(self, strategy_id=None):
        """
        Update risk parameters for a specific strategy or all active strategies
        
        Args:
            strategy_id: ID of strategy to update (None updates all active)
            
        Returns:
            int: Number of strategies updated
        """
        try:
            # Query strategies to update
            if strategy_id:
                strategies = StrategyConfiguration.query.filter_by(id=strategy_id, is_active=True).all()
            else:
                strategies = StrategyConfiguration.query.filter_by(is_active=True).all()
            
            if not strategies:
                logger.warning(f"No active strategies found to update risk parameters")
                return 0
            
            count = 0
            for strategy in strategies:
                # Get adaptive risk parameters
                risk_params = self.get_risk_parameters(strategy.strategy_type, strategy.symbol)
                
                # Update strategy parameters
                params = strategy.parameters.copy()
                params["stop_loss"] = risk_params["stop_loss"]
                params["take_profit"] = risk_params["take_profit"]
                
                # Store additional risk parameters in metadata
                if "risk_metadata" not in params:
                    params["risk_metadata"] = {}
                
                params["risk_metadata"] = {
                    "position_size": risk_params["position_size"],
                    "daily_loss_limit": risk_params["daily_loss_limit"],
                    "market_volatility": risk_params["market_volatility"],
                    "last_updated": datetime.now().isoformat()
                }
                
                # Update strategy
                strategy.parameters = params
                count += 1
            
            # Commit changes
            db.session.commit()
            
            logger.info(f"Updated risk parameters for {count} strategies")
            return count
            
        except Exception as e:
            logger.error(f"Error updating strategy risk parameters: {e}")
            db.session.rollback()
            return 0