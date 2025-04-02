import logging
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_required, current_user
from datetime import datetime, timedelta

from models import User, TradingSession, Trade, StrategyConfiguration
from app import db

logger = logging.getLogger(__name__)

# Create main blueprint
main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')

@main_bp.route('/backtest')
def backtest():
    """Render the backtest page"""
    return render_template('backtest.html')

@main_bp.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html')

@main_bp.route('/supervisor')
def supervisor():
    """Render the supervisor page"""
    return render_template('supervisor.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # For now, we'll use a simple authentication
        # In a production environment, you would use proper authentication
        user = User.query.filter_by(username=username).first()
        
        if user and password == 'password':  # This is a placeholder; use proper password verification in production
            session['user_id'] = user.id
            flash('Login successful', 'success')
            return redirect(url_for('main_bp.dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@main_bp.route('/logout')
def logout():
    """Handle user logout"""
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('main_bp.index'))

@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('main_bp.register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('main_bp.register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=password  # In production, hash the password
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('main_bp.login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user: {e}")
            flash('An error occurred during registration', 'danger')
    
    return render_template('register.html')

@main_bp.route('/create_demo_data', methods=['GET'])
def create_demo_data():
    """Create demo data for testing (development only)"""
    try:
        # Check if we already have data
        if User.query.count() > 0:
            flash('Demo data already exists', 'info')
            return redirect(url_for('main_bp.index'))
        
        # Create demo user
        demo_user = User(
            username='demo',
            email='demo@example.com',
            password_hash='demo',  # In production, hash the password
            api_key='demo_key',
            api_secret='demo_secret'
        )
        db.session.add(demo_user)
        db.session.flush()  # Get the user ID
        
        # Create strategy configurations
        strategies = [
            {
                'strategy_type': 'trend_following',
                'symbol': 'BTC-USDT',
                'timeframe': '1h',
                'parameters': {
                    'short_window': 20,
                    'long_window': 50,
                    'take_profit': 3.0,
                    'stop_loss': 2.0
                },
                'weight': 0.25
            },
            {
                'strategy_type': 'mean_reversion',
                'symbol': 'BTC-USDT',
                'timeframe': '1h',
                'parameters': {
                    'rsi_period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70,
                    'take_profit': 2.0,
                    'stop_loss': 2.0
                },
                'weight': 0.25
            },
            {
                'strategy_type': 'breakout',
                'symbol': 'BTC-USDT',
                'timeframe': '1h',
                'parameters': {
                    'period': 20,
                    'atr_period': 14,
                    'atr_multiplier': 2.0,
                    'take_profit': 4.0,
                    'stop_loss': 2.0
                },
                'weight': 0.25
            },
            {
                'strategy_type': 'momentum',
                'symbol': 'BTC-USDT',
                'timeframe': '1h',
                'parameters': {
                    'period': 14,
                    'threshold': 0.5,
                    'take_profit': 3.0,
                    'stop_loss': 2.0
                },
                'weight': 0.25
            }
        ]
        
        for strategy_data in strategies:
            strategy = StrategyConfiguration(
                user_id=demo_user.id,
                strategy_type=strategy_data['strategy_type'],
                symbol=strategy_data['symbol'],
                timeframe=strategy_data['timeframe'],
                parameters=strategy_data['parameters'],
                weight=strategy_data['weight']
            )
            db.session.add(strategy)
        
        # Create a trading session
        now = datetime.utcnow()
        session_start = now - timedelta(days=30)
        
        trading_session = TradingSession(
            user_id=demo_user.id,
            start_time=session_start,
            initial_balance=10000.0,
            final_balance=10500.0,
            pnl=500.0,
            win_rate=60.0,
            max_drawdown=5.0,
            sharpe_ratio=1.2
        )
        db.session.add(trading_session)
        db.session.flush()  # Get the session ID
        
        # Create some demo trades
        demo_trades = [
            {
                'symbol': 'BTC-USDT',
                'strategy': 'trend_following',
                'direction': 'long',
                'entry_price': 25000.0,
                'exit_price': 26000.0,
                'quantity': 0.1,
                'entry_time': now - timedelta(days=25),
                'exit_time': now - timedelta(days=23),
                'pnl': 100.0,
                'pnl_percentage': 4.0,
                'status': 'closed'
            },
            {
                'symbol': 'ETH-USDT',
                'strategy': 'mean_reversion',
                'direction': 'short',
                'entry_price': 1800.0,
                'exit_price': 1750.0,
                'quantity': 1.0,
                'entry_time': now - timedelta(days=20),
                'exit_time': now - timedelta(days=19),
                'pnl': 50.0,
                'pnl_percentage': 2.78,
                'status': 'closed'
            },
            {
                'symbol': 'BTC-USDT',
                'strategy': 'breakout',
                'direction': 'long',
                'entry_price': 26500.0,
                'exit_price': 26200.0,
                'quantity': 0.15,
                'entry_time': now - timedelta(days=15),
                'exit_time': now - timedelta(days=14),
                'pnl': -45.0,
                'pnl_percentage': -1.13,
                'status': 'closed'
            },
            {
                'symbol': 'SOL-USDT',
                'strategy': 'momentum',
                'direction': 'long',
                'entry_price': 20.0,
                'exit_price': 22.0,
                'quantity': 50.0,
                'entry_time': now - timedelta(days=10),
                'exit_time': now - timedelta(days=8),
                'pnl': 100.0,
                'pnl_percentage': 10.0,
                'status': 'closed'
            },
            {
                'symbol': 'BTC-USDT',
                'strategy': 'trend_following',
                'direction': 'long',
                'entry_price': 27000.0,
                'quantity': 0.05,
                'entry_time': now - timedelta(days=1),
                'status': 'open'
            }
        ]
        
        for trade_data in demo_trades:
            trade = Trade(
                session_id=trading_session.id,
                symbol=trade_data['symbol'],
                strategy=trade_data['strategy'],
                direction=trade_data['direction'],
                entry_price=trade_data['entry_price'],
                quantity=trade_data['quantity'],
                entry_time=trade_data['entry_time'],
                status=trade_data['status']
            )
            
            if 'exit_price' in trade_data:
                trade.exit_price = trade_data['exit_price']
            
            if 'exit_time' in trade_data:
                trade.exit_time = trade_data['exit_time']
            
            if 'pnl' in trade_data:
                trade.pnl = trade_data['pnl']
            
            if 'pnl_percentage' in trade_data:
                trade.pnl_percentage = trade_data['pnl_percentage']
            
            db.session.add(trade)
        
        # Commit all changes
        db.session.commit()
        
        flash('Demo data created successfully', 'success')
        return redirect(url_for('main_bp.index'))
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating demo data: {e}")
        flash('An error occurred while creating demo data', 'danger')
        return redirect(url_for('main_bp.index'))

@main_bp.context_processor
def inject_global_data():
    """Inject global data into all templates"""
    def get_current_year():
        return datetime.utcnow().year
    
    return dict(current_year=get_current_year())
