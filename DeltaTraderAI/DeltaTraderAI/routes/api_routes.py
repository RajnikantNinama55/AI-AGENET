import logging
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, session

from models import User, TradingSession, Trade, StrategyConfiguration
from app import db
from api.delta_exchange import DeltaExchangeAPI
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.momentum import MomentumStrategy
from rl.ppo_model import PPOModel
from backtest.backtester import Backtester
from config import Config

logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api_bp', __name__, url_prefix='/api')

# Global variables to track bot state and signals
BOT_STATUS = "Stopped"
LAST_SIGNAL = None
BOT_RUNNING = False
ACTIVE_TRADES = []
ACTIVE_STRATEGIES = {}

@api_bp.route('/get_tickers', methods=['GET'])
def get_tickers():
    """Get ticker data from Delta Exchange"""
    try:
        api_client = DeltaExchangeAPI()
        symbol = request.args.get('symbol')
        
        if symbol:
            response = api_client.get_tickers(symbol=symbol)
            # Return the response directly
            return jsonify({"success": True, "data": response.get('result', [])})
        else:
            # Get tickers for default symbols
            symbols = Config.DEFAULT_SYMBOLS
            all_tickers = []
            
            for symbol in symbols:
                ticker_data = api_client.get_tickers(symbol=symbol)
                if ticker_data and 'result' in ticker_data:
                    all_tickers.extend(ticker_data['result'])
            
            return jsonify({"success": True, "data": all_tickers})
    except Exception as e:
        logger.error(f"Error getting tickers: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_bot_status', methods=['GET'])
def get_bot_status():
    """Get the current status of the trading bot and risk metrics"""
    global BOT_STATUS, ACTIVE_STRATEGIES
    
    risk_metrics = None
    
    # Get risk metrics from one of the active strategies if available
    if ACTIVE_STRATEGIES and len(ACTIVE_STRATEGIES) > 0:
        # Get the first active strategy
        strategy = list(ACTIVE_STRATEGIES.values())[0]
        if hasattr(strategy, 'get_risk_metrics'):
            try:
                risk_metrics = strategy.get_risk_metrics()
            except Exception as e:
                logger.error(f"Error getting risk metrics: {e}")
    
    response = {
        "success": True, 
        "status": BOT_STATUS
    }
    
    # Add risk metrics if available
    if risk_metrics:
        response["risk_metrics"] = risk_metrics
        
        # Add trading_paused status from risk manager
        if risk_metrics.get('trading_paused'):
            response["paused_reason"] = risk_metrics.get('pause_reason')
    
    return jsonify(response)

@api_bp.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global BOT_STATUS, BOT_RUNNING, ACTIVE_STRATEGIES
    
    if BOT_RUNNING:
        return jsonify({"success": False, "message": "Bot is already running"})
    
    try:
        data = request.json or {}
        account_balance = float(data.get('initial_balance', 10000.0))
        
        # Get active strategy configurations from the database
        strategy_configs = StrategyConfiguration.query.filter_by(is_active=True).all()
        
        if not strategy_configs:
            # Create a default strategy configuration if none exists
            strategy_config = StrategyConfiguration(
                user_id=1,  # Default user ID
                strategy_type='trend_following',
                symbol='BTC-USDT',
                timeframe='1h',
                parameters=Config.TREND_FOLLOWING_DEFAULTS,
                weight=1.0,
                is_active=True
            )
            db.session.add(strategy_config)
            db.session.commit()
            
            strategy_configs = [strategy_config]
            logger.info(f"Created default strategy configuration: {strategy_config.strategy_type}")
        
        # Get the current user's API settings
        user = User.query.first()
        if not user:
            return jsonify({"success": False, "message": "User not found"})
            
        # Initialize API client with user's credentials
        api_client = DeltaExchangeAPI(
            api_key=user.api_key or Config.DELTA_EXCHANGE_API_KEY,
            api_secret=user.api_secret or Config.DELTA_EXCHANGE_API_SECRET
        )
        
        # Initialize strategies from configurations
        for config in strategy_configs:
            strategy_key = f"{config.strategy_type}_{config.symbol}_{config.timeframe}"
            
            if strategy_key in ACTIVE_STRATEGIES:
                # Strategy already exists, skip
                continue
                
            strategy_instance = None
            
            if config.strategy_type == 'trend_following':
                strategy_instance = TrendFollowingStrategy(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    parameters=config.parameters,
                    api_client=api_client,
                    account_balance=account_balance
                )
            elif config.strategy_type == 'mean_reversion':
                strategy_instance = MeanReversionStrategy(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    parameters=config.parameters,
                    api_client=api_client,
                    account_balance=account_balance
                )
            elif config.strategy_type == 'breakout':
                strategy_instance = BreakoutStrategy(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    parameters=config.parameters,
                    api_client=api_client,
                    account_balance=account_balance
                )
            elif config.strategy_type == 'momentum':
                strategy_instance = MomentumStrategy(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    parameters=config.parameters,
                    api_client=api_client,
                    account_balance=account_balance
                )
                
            if strategy_instance:
                # Initialize with historical data
                try:
                    strategy_instance.fetch_historical_data()
                    strategy_instance.calculate_indicators()
                    
                    # Add to active strategies
                    ACTIVE_STRATEGIES[strategy_key] = strategy_instance
                    logger.info(f"Initialized strategy: {strategy_key}")
                except Exception as e:
                    logger.error(f"Error initializing strategy {strategy_key}: {e}")
        
        # Create a new trading session
        trading_session = TradingSession(
            user_id=1,  # Default user ID
            start_time=datetime.utcnow(),
            initial_balance=account_balance
        )
        db.session.add(trading_session)
        db.session.commit()
        
        # Set bot status
        BOT_STATUS = "Running"
        BOT_RUNNING = True
        
        logger.info(f"Trading bot started with {len(ACTIVE_STRATEGIES)} active strategies")
        
        # Return status and active strategies
        return jsonify({
            "success": True,
            "active_strategies": list(ACTIVE_STRATEGIES.keys())
        })
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/pause_bot', methods=['POST'])
def pause_bot():
    """Pause the trading bot"""
    global BOT_STATUS, BOT_RUNNING
    
    if not BOT_RUNNING:
        return jsonify({"success": False, "message": "Bot is not running"})
    
    try:
        # In a real implementation, you would pause the background process here
        BOT_STATUS = "Paused"
        BOT_RUNNING = False
        logger.info("Trading bot paused")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error pausing bot: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global BOT_STATUS, BOT_RUNNING, ACTIVE_STRATEGIES
    
    if not BOT_RUNNING and BOT_STATUS != "Paused":
        return jsonify({"success": False, "message": "Bot is not running or paused"})
    
    try:
        # Close any active positions
        for strategy_key, strategy in ACTIVE_STRATEGIES.items():
            if strategy.position['is_open']:
                try:
                    # Get current price
                    ticker_data = strategy.api_client.get_tickers(strategy.symbol)
                    if ticker_data.get('result'):
                        price = float(ticker_data['result'][0]['mark_price'])
                        
                        # Close position
                        strategy._close_position(price)
                        logger.info(f"Closed position for strategy {strategy_key}")
                except Exception as e:
                    logger.error(f"Error closing position for strategy {strategy_key}: {e}")
        
        # Update the active trading session
        trading_session = TradingSession.query.order_by(TradingSession.start_time.desc()).first()
        if trading_session and not trading_session.end_time:
            # Get the account balance from the first strategy if available
            final_balance = 10000.0
            if ACTIVE_STRATEGIES:
                strategy = list(ACTIVE_STRATEGIES.values())[0]
                if hasattr(strategy, 'risk_manager'):
                    final_balance = strategy.risk_manager.account_balance
            
            trading_session.end_time = datetime.utcnow()
            trading_session.final_balance = final_balance
            trading_session.pnl = final_balance - trading_session.initial_balance
            
            # Calculate win rate
            trades = Trade.query.filter_by(session_id=trading_session.id, status='closed').all()
            if trades:
                winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
                trading_session.win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            
            db.session.commit()
            logger.info(f"Updated trading session {trading_session.id} with final balance {final_balance}")
        
        # Clear active strategies
        ACTIVE_STRATEGIES.clear()
        
        # Set bot status
        BOT_STATUS = "Stopped"
        BOT_RUNNING = False
        
        logger.info("Trading bot stopped")
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_active_trades', methods=['GET'])
def get_active_trades():
    """Get active trades"""
    try:
        # In a real implementation, you would query the database
        # For now, we'll query open trades from the database
        active_trades = Trade.query.filter_by(status='open').all()
        
        return jsonify({
            "success": True, 
            "count": len(active_trades),
            "trades": [trade.id for trade in active_trades]
        })
    except Exception as e:
        logger.error(f"Error getting active trades: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_daily_pnl', methods=['GET'])
def get_daily_pnl():
    """Get today's PnL"""
    try:
        # Get today's datetime range
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())
        
        # Query trades closed today
        today_trades = Trade.query.filter(
            Trade.exit_time.between(today_start, today_end),
            Trade.status == 'closed'
        ).all()
        
        # Calculate PnL
        daily_pnl = sum(trade.pnl or 0 for trade in today_trades)
        
        return jsonify({
            "success": True,
            "pnl": daily_pnl
        })
    except Exception as e:
        logger.error(f"Error getting daily PnL: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_last_signal', methods=['GET'])
def get_last_signal():
    """Get the last trading signal"""
    global LAST_SIGNAL
    
    # If no real signal, create a placeholder
    if not LAST_SIGNAL:
        # Get the most recent trade
        latest_trade = Trade.query.order_by(Trade.entry_time.desc()).first()
        
        if latest_trade:
            LAST_SIGNAL = {
                "symbol": latest_trade.symbol,
                "direction": latest_trade.direction,
                "strategy": latest_trade.strategy,
                "time": latest_trade.entry_time.isoformat()
            }
    
    return jsonify({
        "success": True,
        "signal": LAST_SIGNAL
    })

@api_bp.route('/get_recent_trades', methods=['GET'])
def get_recent_trades():
    """Get recent trades for the home page"""
    try:
        # Get 5 most recent closed trades
        recent_trades = Trade.query.filter_by(status='closed').order_by(Trade.exit_time.desc()).limit(5).all()
        
        trades_data = []
        for trade in recent_trades:
            trades_data.append({
                "id": trade.id,
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "pnl": trade.pnl,
                "pnl_percentage": trade.pnl_percentage
            })
        
        return jsonify({
            "success": True,
            "trades": trades_data
        })
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/optimize_strategies', methods=['POST'])
def optimize_strategies():
    """Run strategy optimization"""
    try:
        # Initialize backtester and API client
        api_client = DeltaExchangeAPI()
        backtester = Backtester(api_client=api_client)
        
        # Get strategies from the database
        strategies_config = StrategyConfiguration.query.filter_by(is_active=True).all()
        
        if not strategies_config:
            return jsonify({
                "success": False,
                "message": "No active strategies found"
            })
        
        # Group strategies by symbol and timeframe
        strategy_groups = {}
        for config in strategies_config:
            key = f"{config.symbol}_{config.timeframe}"
            if key not in strategy_groups:
                strategy_groups[key] = []
            strategy_groups[key].append(config)
        
        optimization_results = {}
        
        # Run optimization for each group
        for key, configs in strategy_groups.items():
            symbol, timeframe = key.split('_')
            
            # Fetch historical data
            historical_data = backtester.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=Config.BACKTEST_START_DATE
            )
            
            if historical_data is None:
                logger.warning(f"No historical data available for {symbol} {timeframe}")
                continue
            
            # Initialize strategies
            strategies = []
            for config in configs:
                strategy_instance = None
                
                if config.strategy_type == 'trend_following':
                    strategy_instance = TrendFollowingStrategy(
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        parameters=config.parameters,
                        api_client=api_client
                    )
                elif config.strategy_type == 'mean_reversion':
                    strategy_instance = MeanReversionStrategy(
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        parameters=config.parameters,
                        api_client=api_client
                    )
                elif config.strategy_type == 'breakout':
                    strategy_instance = BreakoutStrategy(
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        parameters=config.parameters,
                        api_client=api_client
                    )
                elif config.strategy_type == 'momentum':
                    strategy_instance = MomentumStrategy(
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        parameters=config.parameters,
                        api_client=api_client
                    )
                
                if strategy_instance:
                    strategies.append(strategy_instance)
            
            if not strategies:
                logger.warning(f"No strategies initialized for {symbol} {timeframe}")
                continue
            
            # Create PPO model for optimization
            ppo_model = PPOModel()
            
            # Run optimization
            optimized_weights = ppo_model.optimize_strategy_weights(
                strategies=strategies,
                historical_data=historical_data,
                episodes=10  # Reduced for faster processing
            )
            
            # Update weights in the database
            for config in configs:
                if config.strategy_type in optimized_weights:
                    config.weight = optimized_weights[config.strategy_type]
            
            # Save changes
            db.session.commit()
            
            # Store results
            optimization_results[key] = optimized_weights
        
        return jsonify({
            "success": True,
            "results": optimization_results
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error optimizing strategies: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_summary_metrics', methods=['GET'])
def get_summary_metrics():
    """Get summary metrics for the dashboard including risk management data"""
    try:
        # Get the most recent trading session
        trading_session = TradingSession.query.order_by(TradingSession.start_time.desc()).first()
        
        # Initialize with default values
        response = {
            "success": True,
            "balance": 10000.0,
            "balance_change": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "recovery_time": 0,
            "risk_metrics": {
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "drawdown": 0.0,
                "trade_count_today": 0,
                "trading_paused": False,
                "pause_reason": None
            }
        }
        
        # Get risk metrics from active strategies
        global ACTIVE_STRATEGIES
        if ACTIVE_STRATEGIES and len(ACTIVE_STRATEGIES) > 0:
            # Get the first active strategy
            strategy = list(ACTIVE_STRATEGIES.values())[0]
            if hasattr(strategy, 'get_risk_metrics'):
                try:
                    risk_metrics = strategy.get_risk_metrics()
                    if risk_metrics:
                        response["risk_metrics"] = risk_metrics
                        response["balance"] = risk_metrics.get("account_balance", 10000.0)
                except Exception as e:
                    logger.error(f"Error getting risk metrics: {e}")
        
        if trading_session:
            # Get all closed trades for this session
            trades = Trade.query.filter_by(
                session_id=trading_session.id, 
                status='closed'
            ).all()
            
            # Calculate metrics
            total_trades = len(trades)
            response["total_trades"] = total_trades
            
            if total_trades > 0:
                winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
                win_rate = len(winning_trades) / total_trades * 100
                
                total_profit = sum(t.pnl for t in winning_trades)
                losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
                total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
                profit_factor = total_profit / total_loss if total_loss > 0 else 0
                
                avg_trade = sum(t.pnl for t in trades if t.pnl) / total_trades
                
                response["win_rate"] = win_rate
                response["profit_factor"] = profit_factor
                response["avg_trade"] = avg_trade
            
            # Get balance and change if not from risk manager
            if "account_balance" not in response["risk_metrics"]:
                balance = trading_session.final_balance or trading_session.initial_balance
                balance_change = ((balance - trading_session.initial_balance) / trading_session.initial_balance) * 100
                response["balance"] = balance
                response["balance_change"] = balance_change
            else:
                # Calculate balance change from initial balance
                balance_change = ((response["balance"] - 10000.0) / 10000.0) * 100
                response["balance_change"] = balance_change
                
            response["max_drawdown"] = max(
                trading_session.max_drawdown or 0.0,
                response["risk_metrics"].get("drawdown", 0.0)
            )
        
        # Add risk status to the response
        if response["risk_metrics"].get("trading_paused"):
            response["risk_status"] = "paused"
            response["pause_reason"] = response["risk_metrics"].get("pause_reason", "Unknown")
        elif response["risk_metrics"].get("drawdown", 0.0) > 10.0:
            response["risk_status"] = "warning"
        else:
            response["risk_status"] = "normal"
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting summary metrics: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_performance_data', methods=['GET'])
def get_performance_data():
    """Get portfolio performance data for charts"""
    try:
        period = request.args.get('period', '1m')
        
        # Define time range based on period
        now = datetime.utcnow()
        if period == '1d':
            start_time = now - timedelta(days=1)
        elif period == '1w':
            start_time = now - timedelta(weeks=1)
        elif period == '1m':
            start_time = now - timedelta(days=30)
        elif period == '3m':
            start_time = now - timedelta(days=90)
        else:  # 'all' or invalid period
            # Get earliest trade
            earliest_trade = Trade.query.order_by(Trade.entry_time.asc()).first()
            start_time = earliest_trade.entry_time if earliest_trade else now - timedelta(days=365)
        
        # Get trading session
        trading_session = TradingSession.query.order_by(TradingSession.start_time.desc()).first()
        
        if not trading_session:
            # Return empty data
            return jsonify({
                "success": True,
                "dates": [],
                "values": []
            })
        
        # Get trades for the session in the time range
        trades = Trade.query.filter(
            Trade.session_id == trading_session.id,
            Trade.entry_time >= start_time
        ).order_by(Trade.entry_time.asc()).all()
        
        # Generate portfolio values over time
        dates = []
        values = []
        
        # Start with initial balance
        current_value = trading_session.initial_balance
        values.append(current_value)
        dates.append(start_time.strftime("%Y-%m-%d %H:%M"))
        
        # Add values at each trade
        for trade in trades:
            if trade.status == 'closed' and trade.exit_time:
                # Add PnL at exit time
                current_value += trade.pnl if trade.pnl else 0
                values.append(current_value)
                dates.append(trade.exit_time.strftime("%Y-%m-%d %H:%M"))
        
        # Add current value
        if values[-1] != current_value:
            values.append(current_value)
            dates.append(now.strftime("%Y-%m-%d %H:%M"))
        
        return jsonify({
            "success": True,
            "dates": dates,
            "values": values
        })
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_strategy_comparison', methods=['GET'])
def get_strategy_comparison():
    """Get strategy comparison data for charts"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        timeframe = request.args.get('timeframe', '1h')
        
        # Initialize backtester
        api_client = DeltaExchangeAPI()
        backtester = Backtester(api_client=api_client)
        
        # Get historical data
        historical_data = backtester.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=Config.BACKTEST_START_DATE
        )
        
        if historical_data is None:
            return jsonify({
                "success": False,
                "message": "No historical data available"
            })
        
        # Initialize strategies
        strategies = [
            TrendFollowingStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client),
            MeanReversionStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client),
            BreakoutStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client),
            MomentumStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client)
        ]
        
        # Run backtest for each strategy
        strategy_names = []
        returns = []
        win_rates = []
        
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__.replace('Strategy', '')
            strategy_names.append(strategy_name)
            
            results = backtester.backtest_strategy(strategy, historical_data.copy())
            
            if results:
                returns.append(results['total_return'])
                win_rates.append(results['win_rate'])
            else:
                returns.append(0)
                win_rates.append(0)
        
        return jsonify({
            "success": True,
            "strategies": strategy_names,
            "returns": returns,
            "win_rates": win_rates
        })
    except Exception as e:
        logger.error(f"Error getting strategy comparison: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_strategy_allocation', methods=['GET'])
def get_strategy_allocation():
    """Get strategy allocation data for charts"""
    try:
        # Get strategies and weights from database
        strategies = StrategyConfiguration.query.filter_by(is_active=True).all()
        
        if not strategies:
            # Return default allocation
            return jsonify({
                "success": True,
                "strategies": ["Trend Following", "Mean Reversion", "Breakout", "Momentum"],
                "weights": [0.25, 0.25, 0.25, 0.25]
            })
        
        # Prepare data for chart
        strategy_names = []
        weights = []
        
        # Group by strategy type and calculate average weight
        strategy_weights = {}
        strategy_display_names = {
            'trend_following': 'Trend Following',
            'mean_reversion': 'Mean Reversion',
            'breakout': 'Breakout',
            'momentum': 'Momentum'
        }
        
        for strategy in strategies:
            if strategy.strategy_type not in strategy_weights:
                strategy_weights[strategy.strategy_type] = []
            
            strategy_weights[strategy.strategy_type].append(strategy.weight)
        
        # Calculate average weight for each strategy
        for strategy_type, type_weights in strategy_weights.items():
            display_name = strategy_display_names.get(strategy_type, strategy_type)
            strategy_names.append(display_name)
            weights.append(sum(type_weights) / len(type_weights))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return jsonify({
            "success": True,
            "strategies": strategy_names,
            "weights": weights
        })
    except Exception as e:
        logger.error(f"Error getting strategy allocation: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_trade_history', methods=['GET'])
def get_trade_history():
    """Get trade history with pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # Get trades with pagination
        trades_query = Trade.query.order_by(Trade.entry_time.desc())
        
        # Count total trades
        total_trades = trades_query.count()
        
        # Calculate total pages
        total_pages = (total_trades + per_page - 1) // per_page
        
        # Get trades for current page
        trades = trades_query.limit(per_page).offset((page - 1) * per_page).all()
        
        # Prepare trade data
        trades_data = []
        for trade in trades:
            trades_data.append({
                "id": trade.id,
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "pnl": trade.pnl,
                "pnl_percentage": trade.pnl_percentage,
                "status": trade.status
            })
        
        return jsonify({
            "success": True,
            "trades": trades_data,
            "current_page": page,
            "total_pages": total_pages,
            "total_trades": total_trades
        })
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/export_trade_history', methods=['GET'])
def export_trade_history():
    """Export all trade history"""
    try:
        # Get all trades
        trades = Trade.query.order_by(Trade.entry_time.desc()).all()
        
        # Prepare trade data
        trades_data = []
        for trade in trades:
            trades_data.append({
                "id": trade.id,
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "pnl": trade.pnl,
                "pnl_percentage": trade.pnl_percentage,
                "status": trade.status
            })
        
        return jsonify({
            "success": True,
            "trades": trades_data
        })
    except Exception as e:
        logger.error(f"Error exporting trade history: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest with the provided parameters"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            })
        
        # Extract parameters
        strategy_type = data.get('strategy')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_capital = float(data.get('initial_capital', 10000))
        optimize_parameters = data.get('optimize_parameters', False)
        parameters = data.get('parameters', {})
        
        # Initialize API client and backtester
        api_client = DeltaExchangeAPI()
        backtester = Backtester(api_client=api_client)
        
        # Fetch historical data
        historical_data = backtester.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date
        )
        
        if historical_data is None:
            return jsonify({
                "success": False,
                "message": "Failed to fetch historical data"
            })
        
        # Run backtest based on strategy type
        if strategy_type == 'all':
            # Run multi-strategy backtest
            strategies = [
                TrendFollowingStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client),
                MeanReversionStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client),
                BreakoutStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client),
                MomentumStrategy(symbol=symbol, timeframe=timeframe, api_client=api_client)
            ]
            
            results = backtester.backtest_multi_strategy(
                strategies=strategies,
                historical_data=historical_data,
                initial_capital=initial_capital
            )
        else:
            # Run single strategy backtest
            strategy = None
            
            if strategy_type == 'trend_following':
                strategy = TrendFollowingStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=api_client
                )
            elif strategy_type == 'mean_reversion':
                strategy = MeanReversionStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=api_client
                )
            elif strategy_type == 'breakout':
                strategy = BreakoutStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=api_client
                )
            elif strategy_type == 'momentum':
                strategy = MomentumStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters=parameters,
                    api_client=api_client
                )
            
            if strategy:
                if optimize_parameters:
                    # Define parameter grid
                    param_grid = {}
                    
                    if strategy_type == 'trend_following':
                        param_grid = {
                            'short_window': [10, 15, 20, 25, 30],
                            'long_window': [40, 50, 60, 70],
                            'take_profit': [2.0, 3.0, 4.0],
                            'stop_loss': [1.5, 2.0, 2.5]
                        }
                    elif strategy_type == 'mean_reversion':
                        param_grid = {
                            'rsi_period': [10, 14, 18],
                            'oversold_threshold': [25, 30, 35],
                            'overbought_threshold': [65, 70, 75],
                            'take_profit': [1.5, 2.0, 2.5],
                            'stop_loss': [1.5, 2.0, 2.5]
                        }
                    elif strategy_type == 'breakout':
                        param_grid = {
                            'period': [15, 20, 25],
                            'atr_period': [10, 14, 18],
                            'atr_multiplier': [1.5, 2.0, 2.5],
                            'take_profit': [3.0, 4.0, 5.0],
                            'stop_loss': [1.5, 2.0, 2.5]
                        }
                    elif strategy_type == 'momentum':
                        param_grid = {
                            'period': [10, 14, 18],
                            'threshold': [0.3, 0.5, 0.7],
                            'take_profit': [2.0, 3.0, 4.0],
                            'stop_loss': [1.5, 2.0, 2.5]
                        }
                    
                    # Run parameter optimization
                    optimize_results = backtester.optimize_parameters(
                        strategy_class=strategy.__class__,
                        symbol=symbol,
                        timeframe=timeframe,
                        param_grid=param_grid,
                        historical_data=historical_data
                    )
                    
                    if optimize_results and optimize_results.get('results'):
                        results = optimize_results['results']
                    else:
                        return jsonify({
                            "success": False,
                            "message": "Parameter optimization failed"
                        })
                else:
                    # Run backtest with provided parameters
                    results = backtester.backtest_strategy(
                        strategy=strategy,
                        historical_data=historical_data,
                        initial_capital=initial_capital
                    )
            else:
                return jsonify({
                    "success": False,
                    "message": "Invalid strategy type"
                })
        
        if results:
            # Add symbol and timeframe to results
            results['symbol'] = symbol
            results['timeframe'] = timeframe
            results['strategy'] = strategy_type
            
            return jsonify({
                "success": True,
                "results": results
            })
        else:
            return jsonify({
                "success": False,
                "message": "Backtest failed to produce results"
            })
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_user_settings', methods=['GET'])
def get_user_settings():
    """Get user API and trading settings"""
    try:
        # In a real implementation, get settings for the current user
        # For now, return placeholder data
        user = User.query.first()
        
        if not user:
            return jsonify({
                "success": True,
                "api_key": "",
                "api_secret": "",
                "trading_settings": {
                    "risk_per_trade": 1.0,
                    "max_trades": 3,
                    "trade_timeout": 48
                },
                "auto_trade": True,
                "trading_symbols": ["BTC-USDT", "ETH-USDT"],
                "trading_timeframes": ["15m", "1h"]
            })
        
        # Get settings from database
        return jsonify({
            "success": True,
            "api_key": user.api_key or "",
            "api_secret": user.api_secret and True or "",  # Just indicate if exists
            "trading_settings": {
                "risk_per_trade": 1.0,  # Placeholder
                "max_trades": 3,        # Placeholder
                "trade_timeout": 48     # Placeholder
            },
            "auto_trade": True,  # Placeholder
            "trading_symbols": ["BTC-USDT", "ETH-USDT"],  # Placeholder
            "trading_timeframes": ["15m", "1h"]  # Placeholder
        })
    except Exception as e:
        logger.error(f"Error getting user settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/save_api_settings', methods=['POST'])
def save_api_settings():
    """Save API settings"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            })
        
        api_key = data.get('api_key', '')
        api_secret = data.get('api_secret', '')
        
        # In a real implementation, save for the current user
        user = User.query.first()
        
        if not user:
            return jsonify({
                "success": False,
                "message": "User not found"
            })
        
        # Update API settings
        user.api_key = api_key
        if api_secret:  # Only update if provided
            user.api_secret = api_secret
        
        db.session.commit()
        
        return jsonify({
            "success": True
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving API settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/save_trading_settings', methods=['POST'])
def save_trading_settings():
    """Save trading settings"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            })
        
        # In a real implementation, we would save these to the database
        # For now, just log and return success
        logger.info(f"Saving trading settings: {data}")
        
        return jsonify({
            "success": True
        })
    except Exception as e:
        logger.error(f"Error saving trading settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/save_strategy_settings', methods=['POST'])
def save_strategy_settings():
    """Save strategy settings"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            })
        
        strategy_type = data.get('strategy')
        is_active = data.get('is_active', True)
        parameters = data.get('parameters', {})
        
        # Find matching strategy configurations
        strategies = StrategyConfiguration.query.filter_by(strategy_type=strategy_type).all()
        
        if not strategies:
            # Create new strategy configuration
            # In a real implementation, we would get the user ID
            user = User.query.first()
            
            if not user:
                return jsonify({
                    "success": False,
                    "message": "User not found"
                })
            
            # Create default configuration
            strategy = StrategyConfiguration(
                user_id=user.id,
                strategy_type=strategy_type,
                symbol="BTC-USDT",  # Default
                timeframe="1h",     # Default
                parameters=parameters,
                is_active=is_active
            )
            
            db.session.add(strategy)
        else:
            # Update existing configurations
            for strategy in strategies:
                strategy.is_active = is_active
                strategy.parameters = parameters
        
        db.session.commit()
        
        return jsonify({
            "success": True
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving strategy settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/save_schedule_settings', methods=['POST'])
def save_schedule_settings():
    """Save trading schedule settings"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            })
        
        # In a real implementation, we would save these to the database
        # For now, just log and return success
        logger.info(f"Saving schedule settings: {data}")
        
        return jsonify({
            "success": True
        })
    except Exception as e:
        logger.error(f"Error saving schedule settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_strategy_settings', methods=['GET'])
def get_strategy_settings():
    """Get strategy settings"""
    try:
        # Get strategies from database
        strategies = StrategyConfiguration.query.all()
        
        if not strategies:
            # Return default settings
            return jsonify({
                "success": True,
                "strategies": {
                    "trend_following": {
                        "is_active": True,
                        "parameters": Config.TREND_FOLLOWING_DEFAULTS
                    },
                    "mean_reversion": {
                        "is_active": True,
                        "parameters": Config.MEAN_REVERSION_DEFAULTS
                    },
                    "breakout": {
                        "is_active": True,
                        "parameters": Config.BREAKOUT_DEFAULTS
                    },
                    "momentum": {
                        "is_active": True,
                        "parameters": Config.MOMENTUM_DEFAULTS
                    }
                }
            })
        
        # Group by strategy type
        strategy_settings = {}
        
        for strategy in strategies:
            if strategy.strategy_type not in strategy_settings:
                strategy_settings[strategy.strategy_type] = {
                    "is_active": strategy.is_active,
                    "parameters": strategy.parameters
                }
        
        # Add any missing strategies with defaults
        if "trend_following" not in strategy_settings:
            strategy_settings["trend_following"] = {
                "is_active": True,
                "parameters": Config.TREND_FOLLOWING_DEFAULTS
            }
            
        if "mean_reversion" not in strategy_settings:
            strategy_settings["mean_reversion"] = {
                "is_active": True,
                "parameters": Config.MEAN_REVERSION_DEFAULTS
            }
            
        if "breakout" not in strategy_settings:
            strategy_settings["breakout"] = {
                "is_active": True,
                "parameters": Config.BREAKOUT_DEFAULTS
            }
            
        if "momentum" not in strategy_settings:
            strategy_settings["momentum"] = {
                "is_active": True,
                "parameters": Config.MOMENTUM_DEFAULTS
            }
        
        return jsonify({
            "success": True,
            "strategies": strategy_settings
        })
    except Exception as e:
        logger.error(f"Error getting strategy settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_supervisor_recommendations', methods=['GET'])
def get_supervisor_recommendations():
    """Get recommendations from the Supervisor Agent"""
    try:
        from main import get_supervisor_instance
        
        # Get supervisor instance with proper app context
        supervisor = get_supervisor_instance()
        
        if not supervisor:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is not initialized"
            })
        
        # Get parameters from request
        max_count = request.args.get('max_count', default=10, type=int)
        min_confidence = request.args.get('min_confidence', default=None, type=str)
        
        # Get recommendations from the supervisor
        recommendations = supervisor.get_recommendations(
            max_count=max_count,
            min_confidence=min_confidence
        )
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting supervisor recommendations: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/apply_recommendation', methods=['POST'])
def apply_recommendation():
    """Apply a specific recommendation from the Supervisor Agent"""
    try:
        from main import get_supervisor_instance
        
        # Get supervisor instance with proper app context
        supervisor = get_supervisor_instance()
        
        if not supervisor:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is not initialized"
            })
        
        data = request.json
        recommendation_id = data.get('recommendation_id')
        
        if recommendation_id is None:
            return jsonify({
                "success": False,
                "message": "Recommendation ID is required"
            })
        
        # Apply the recommendation
        result = supervisor.apply_recommendation(recommendation_id)
        
        if result:
            return jsonify({
                "success": True,
                "message": "Recommendation applied successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to apply recommendation"
            })
    except Exception as e:
        logger.error(f"Error applying recommendation: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/start_supervisor', methods=['POST'])
def start_supervisor():
    """Start the Supervisor Agent if it's not already running"""
    try:
        from main import get_supervisor_instance
        
        # Get supervisor instance with proper app context
        supervisor = get_supervisor_instance()
        
        if not supervisor:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is not initialized"
            })
        
        if supervisor.running:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is already running"
            })
        
        # Start the supervisor
        result = supervisor.start()
        
        if result:
            return jsonify({
                "success": True,
                "message": "Supervisor Agent started successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to start Supervisor Agent"
            })
    except Exception as e:
        logger.error(f"Error starting supervisor agent: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/stop_supervisor', methods=['POST'])
def stop_supervisor():
    """Stop the Supervisor Agent if it's running"""
    try:
        from main import get_supervisor_instance
        
        # Get supervisor instance with proper app context
        supervisor = get_supervisor_instance()
        
        if not supervisor:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is not initialized"
            })
        
        if not supervisor.running:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is not running"
            })
        
        # Stop the supervisor
        result = supervisor.stop()
        
        if result:
            return jsonify({
                "success": True,
                "message": "Supervisor Agent stopped successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to stop Supervisor Agent"
            })
    except Exception as e:
        logger.error(f"Error stopping supervisor agent: {e}")
        return jsonify({"success": False, "message": str(e)})

@api_bp.route('/get_supervisor_status', methods=['GET'])
def get_supervisor_status():
    """Get the current status of the Supervisor Agent"""
    try:
        from main import get_supervisor_instance
        
        # Get supervisor instance with proper app context
        supervisor = get_supervisor_instance()
        
        if not supervisor:
            return jsonify({
                "success": False,
                "message": "Supervisor Agent is not initialized"
            })
        
        # Get status
        status = {
            "running": supervisor.running,
            "last_optimization": supervisor.last_optimization.isoformat() if supervisor.last_optimization else None,
            "recommendation_count": len(supervisor.recommendations),
            "rl_hyperparameters": supervisor.rl_hyperparams
        }
        
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        logger.error(f"Error getting supervisor status: {e}")
        return jsonify({"success": False, "message": str(e)})
