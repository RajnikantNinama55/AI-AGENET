"""
Continuous Learning Scheduler for Trading Bot

This module handles scheduling the continuous learning process:
1. Regular evaluation of strategy performance
2. Periodic evolution of strategy parameters
3. Market regime detection and adaptation
4. Automated RL model training with new data
"""

import logging
import time
import threading
from datetime import datetime, timedelta
import schedule
import random

from config import Config
from evolution.evolution_manager import EvolutionManager
from models import StrategyConfiguration, Trade, TradingSession
from app import db

logger = logging.getLogger(__name__)

class ContinuousLearningScheduler:
    """
    Scheduler for continuous learning and evolution of trading strategies.
    
    This scheduler runs in a background thread and periodically executes
    evolution, adaptation, and learning operations to improve the trading
    system over time.
    """
    
    def __init__(self, supervisor=None, api_client=None, backtester=None, ppo_model=None):
        """
        Initialize the continuous learning scheduler
        
        Args:
            supervisor: Reference to supervisor agent
            api_client: API client for market data
            backtester: Backtester for strategy evaluation
            ppo_model: PPO model for reinforcement learning
        """
        self.supervisor = supervisor
        self.api_client = api_client
        self.backtester = backtester
        self.ppo_model = ppo_model
        
        self.evolution_manager = EvolutionManager(
            supervisor=supervisor,
            api_client=api_client,
            backtester=backtester
        )
        
        self.running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Last execution timestamps
        self.last_regime_detection = None
        self.last_evolution = None
        self.last_rl_optimization = None
        
        # Learning schedule
        self.regime_detection_interval = 4  # hours
        self.evolution_interval = 24  # hours
        self.rl_optimization_interval = 12  # hours
        
        # Initialize schedules
        self._initialize_schedules()
        
        logger.info("Continuous Learning Scheduler initialized")
    
    def _initialize_schedules(self):
        """Initialize the schedules for each task"""
        # Market regime detection (multiple times per day)
        for hour in [0, 4, 8, 12, 16, 20]:
            schedule.every().day.at(f"{hour:02d}:00").do(self._detect_market_regime)
        
        # Strategy evolution (once per day at a random hour to avoid predictability)
        evolution_hour = random.randint(1, 5)  # Early morning hours when markets might be less volatile
        schedule.every().day.at(f"{evolution_hour:02d}:30").do(self._evolve_strategies)
        
        # RL model optimization (twice per day)
        schedule.every().day.at("06:15").do(self._optimize_rl_model)
        schedule.every().day.at("18:15").do(self._optimize_rl_model)
        
        # Performance analysis (every 8 hours)
        for hour in [2, 10, 18]:
            schedule.every().day.at(f"{hour:02d}:45").do(self._analyze_performance)
        
        # Strategy weights adjustment (every 6 hours)
        for hour in [0, 6, 12, 18]:
            schedule.every().day.at(f"{hour:02d}:10").do(self._adjust_strategy_weights)
    
    def start(self):
        """Start the continuous learning scheduler in a background thread"""
        if self.running:
            logger.warning("Continuous Learning Scheduler is already running")
            return False
        
        self.running = True
        self.stop_event.clear()
        
        # Create and start the scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="ContinuousLearningThread"
        )
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Continuous Learning Scheduler started")
        return True
    
    def stop(self):
        """Stop the continuous learning scheduler"""
        if not self.running:
            logger.warning("Continuous Learning Scheduler is not running")
            return False
        
        self.running = False
        self.stop_event.set()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Continuous Learning Scheduler stopped")
        return True
    
    def _scheduler_loop(self):
        """Main loop for the scheduler"""
        logger.info("Continuous Learning Scheduler loop started")
        
        # Initial detection on startup
        self._detect_market_regime()
        
        while not self.stop_event.is_set():
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Sleep for a short time to avoid high CPU usage
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                # Sleep a bit longer on error
                time.sleep(60)
    
    def _detect_market_regime(self):
        """Detect the current market regime and adapt strategies"""
        try:
            logger.info("Detecting market regime")
            
            # Detect market regime
            regime = self.evolution_manager.detect_market_regime()
            
            # Log regime detection
            logger.info(f"Current market regime: {regime}")
            self.last_regime_detection = datetime.now()
            
            # Use Flask application context for database operations if needed
            from app import app
            with app.app_context():
                # Update regime in database if it changed
                if regime != "unknown" and self.evolution_manager.market_regime != regime:
                    # Record regime change if a model exists for it
                    try:
                        from models import MarketRegimeChange
                        regime_change = MarketRegimeChange(
                            regime=regime,
                            timestamp=datetime.now()
                        )
                        db.session.add(regime_change)
                        db.session.commit()
                        logger.info(f"Market regime changed from {self.evolution_manager.market_regime or 'unknown'} to {regime}")
                    except Exception as e:
                        logger.warning(f"Could not record regime change in database: {e}")
                        db.session.rollback()
                
                    # Update the evolution manager
                    self.evolution_manager.market_regime = regime
                
            # Adapt to the detected regime
            if regime != "unknown":
                self._adapt_to_regime(regime)
            
            return True
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return False
    
    def _adapt_to_regime(self, regime):
        """Adapt strategies to the current market regime"""
        try:
            # Get optimal strategy mix for the current regime
            optimal_mix = self.evolution_manager.get_optimal_strategy_mix()
            
            # Use Flask application context for database operations
            from app import app
            with app.app_context():
                # Update strategy weights in the database
                strategies_by_type = {}
                strategies = StrategyConfiguration.query.filter_by(is_active=True).all()
                
                for strategy in strategies:
                    if strategy.strategy_type not in strategies_by_type:
                        strategies_by_type[strategy.strategy_type] = []
                    strategies_by_type[strategy.strategy_type].append(strategy)
                
                for strategy_type, weight in optimal_mix.items():
                    if strategy_type in strategies_by_type:
                        # Distribute weight among strategies of this type
                        strategies_of_type = strategies_by_type[strategy_type]
                        weight_per_strategy = weight / len(strategies_of_type)
                        
                        for strategy in strategies_of_type:
                            strategy.weight = weight_per_strategy
                
                # Commit changes
                db.session.commit()
            
            logger.info(f"Adapted strategy weights to {regime} regime: {optimal_mix}")
            return True
        except Exception as e:
            logger.error(f"Error adapting to market regime: {e}")
            # Use Flask app context for rollback too
            from app import app
            with app.app_context():
                db.session.rollback()
            return False
    
    def _evolve_strategies(self):
        """Evolve trading strategies using genetic algorithm"""
        try:
            logger.info("Evolving trading strategies")
            
            # Use Flask application context for database operations
            from app import app
            with app.app_context():
                # Get current active strategies
                current_strategies = StrategyConfiguration.query.filter_by(is_active=True).all()
                
                if not current_strategies:
                    logger.warning("No active strategies found for evolution")
                    return False
                
                # Evolve strategies
                evolved_strategies = self.evolution_manager.evolve_strategies(current_strategies)
                
                if not evolved_strategies:
                    logger.warning("No evolved strategies returned")
                    return False
                
                # Deactivate old strategies
                for strategy in current_strategies:
                    strategy.is_active = False
                
                # Add evolved strategies to database
                for strategy in evolved_strategies:
                    db.session.add(strategy)
                
                # Commit changes
                db.session.commit()
            
            self.last_evolution = datetime.now()
            logger.info(f"Successfully evolved {len(evolved_strategies)} strategies")
            
            return True
        except Exception as e:
            logger.error(f"Error evolving strategies: {e}")
            # Use Flask app context for rollback too
            from app import app
            with app.app_context():
                db.session.rollback()
            return False
    
    def _optimize_rl_model(self):
        """Optimize reinforcement learning model hyperparameters"""
        try:
            logger.info("Optimizing RL model hyperparameters")
            
            if self.ppo_model is None:
                logger.warning("No PPO model available for optimization")
                return False
            
            # Evolve RL model hyperparameters
            updated_params = self.evolution_manager.evolve_rl_model(self.ppo_model)
            
            if not updated_params:
                logger.warning("No hyperparameter updates applied")
                return False
            
            self.last_rl_optimization = datetime.now()
            logger.info(f"Successfully optimized RL model: {updated_params}")
            
            return True
        except Exception as e:
            logger.error(f"Error optimizing RL model: {e}")
            return False
    
    def _analyze_performance(self):
        """Analyze trading performance and generate insights"""
        try:
            logger.info("Analyzing trading performance")
            
            # Use Flask application context for database operations
            from app import app
            with app.app_context():
                # Get recent trades
                recent_trades = Trade.query.join(TradingSession).filter(
                    Trade.entry_time >= datetime.now() - timedelta(days=7)
                ).all()
                
                if not recent_trades:
                    logger.info("No recent trades found for performance analysis")
                    return False
                
                # Calculate key metrics
                total_trades = len(recent_trades)
                profitable_trades = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                
                # Calculate PnL by strategy type
                pnl_by_strategy = {}
                for trade in recent_trades:
                    if trade.strategy not in pnl_by_strategy:
                        pnl_by_strategy[trade.strategy] = []
                    
                    if trade.pnl is not None:
                        pnl_by_strategy[trade.strategy].append(trade.pnl)
                
                # Calculate average PnL by strategy
                avg_pnl_by_strategy = {}
                for strategy, pnls in pnl_by_strategy.items():
                    if pnls:
                        avg_pnl_by_strategy[strategy] = sum(pnls) / len(pnls)
                
                # Log performance metrics
                logger.info(f"Performance metrics: Total trades: {total_trades}, Win rate: {win_rate:.2%}")
                for strategy, avg_pnl in avg_pnl_by_strategy.items():
                    logger.info(f"  {strategy}: Avg PnL: {avg_pnl:.2f}, Trades: {len(pnl_by_strategy[strategy])}")
            
            return True
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return False
    
    def _adjust_strategy_weights(self):
        """Adjust strategy weights based on recent performance"""
        try:
            logger.info("Adjusting strategy weights based on performance")
            
            # Use Flask application context for database operations
            from app import app
            with app.app_context():
                # Get strategies
                strategies = StrategyConfiguration.query.filter_by(is_active=True).all()
                
                if not strategies:
                    logger.warning("No active strategies found for weight adjustment")
                    return False
                
                # Get recent trades for each strategy type
                strategy_performance = {}
                
                for strategy in strategies:
                    # Get recent trades for this strategy
                    trades = Trade.query.join(TradingSession).filter(
                        Trade.strategy == strategy.strategy_type,
                        Trade.entry_time >= datetime.now() - timedelta(days=3)
                    ).all()
                    
                    if trades:
                        # Calculate performance score (simplified)
                        win_count = sum(1 for t in trades if t.pnl and t.pnl > 0)
                        win_rate = win_count / len(trades)
                        avg_profit = sum(t.pnl for t in trades if t.pnl) / len(trades)
                        
                        # Combine into a score
                        score = (win_rate * 0.4) + (max(0, avg_profit) * 0.6)
                        
                        if strategy.strategy_type not in strategy_performance:
                            strategy_performance[strategy.strategy_type] = []
                        
                        strategy_performance[strategy.strategy_type].append({
                            'id': strategy.id,
                            'score': score
                        })
                
                # Calculate adjustments (total weights must sum to 1.0)
                if not strategy_performance:
                    logger.info("No performance data available for weight adjustment")
                    return False
                
                # Calculate average score per strategy type
                avg_scores = {}
                for strategy_type, performances in strategy_performance.items():
                    avg_scores[strategy_type] = sum(p['score'] for p in performances) / len(performances)
                
                total_score = sum(avg_scores.values())
                if total_score <= 0:
                    logger.warning("Total performance score is zero or negative")
                    return False
                
                # Calculate new weights
                new_weights = {k: v / total_score for k, v in avg_scores.items()}
                
                # Ensure minimum weights
                min_weight = 0.05
                adjusted_weights = {}
                remaining_weight = 1.0
                
                for strategy_type, weight in new_weights.items():
                    if weight < min_weight:
                        adjusted_weights[strategy_type] = min_weight
                        remaining_weight -= min_weight
                    else:
                        adjusted_weights[strategy_type] = weight
                
                # Normalize the remaining weights
                remaining_total = sum(w for t, w in adjusted_weights.items() if w > min_weight)
                if remaining_total > 0:
                    for strategy_type in adjusted_weights:
                        if adjusted_weights[strategy_type] > min_weight:
                            adjusted_weights[strategy_type] = (
                                adjusted_weights[strategy_type] / remaining_total * remaining_weight
                            )
                
                # Apply new weights to strategies
                strategies_by_type = {}
                for strategy in strategies:
                    if strategy.strategy_type not in strategies_by_type:
                        strategies_by_type[strategy.strategy_type] = []
                    strategies_by_type[strategy.strategy_type].append(strategy)
                
                for strategy_type, weight in adjusted_weights.items():
                    if strategy_type in strategies_by_type:
                        strategies_of_type = strategies_by_type[strategy_type]
                        weight_per_strategy = weight / len(strategies_of_type)
                        
                        for strategy in strategies_of_type:
                            strategy.weight = weight_per_strategy
                
                # Commit changes
                db.session.commit()
                
                logger.info(f"Adjusted strategy weights: {adjusted_weights}")
            return True
        except Exception as e:
            logger.error(f"Error adjusting strategy weights: {e}")
            # Use Flask app context for rollback too
            from app import app
            with app.app_context():
                db.session.rollback()
            return False
    
    def get_status(self):
        """Get the current status of the continuous learning scheduler"""
        return {
            "running": self.running,
            "last_regime_detection": self.last_regime_detection.isoformat() if self.last_regime_detection else None,
            "last_evolution": self.last_evolution.isoformat() if self.last_evolution else None,
            "last_rl_optimization": self.last_rl_optimization.isoformat() if self.last_rl_optimization else None,
            "current_market_regime": self.evolution_manager.market_regime,
            "generation": self.evolution_manager.generation
        }
    
    def force_run_task(self, task_name):
        """Force a task to run immediately"""
        tasks = {
            "detect_regime": self._detect_market_regime,
            "evolve_strategies": self._evolve_strategies,
            "optimize_rl": self._optimize_rl_model,
            "analyze_performance": self._analyze_performance,
            "adjust_weights": self._adjust_strategy_weights
        }
        
        if task_name in tasks:
            try:
                result = tasks[task_name]()
                logger.info(f"Forced task {task_name} executed with result: {result}")
                return result
            except Exception as e:
                logger.error(f"Error executing forced task {task_name}: {e}")
                return False
        else:
            logger.warning(f"Unknown task name: {task_name}")
            return False