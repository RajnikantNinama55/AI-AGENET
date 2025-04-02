from datetime import datetime
from app import db
from flask_login import UserMixin

class User(UserMixin, db.Model):
    """User model for authentication and settings storage"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    api_key = db.Column(db.String(256))
    api_secret = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy_configurations = db.relationship('StrategyConfiguration', backref='user', lazy=True)
    trading_sessions = db.relationship('TradingSession', backref='user', lazy=True)

class StrategyConfiguration(db.Model):
    """Configuration for trading strategies"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    strategy_type = db.Column(db.String(50), nullable=False)  # 'trend_following', 'mean_reversion', 'breakout', 'momentum'
    symbol = db.Column(db.String(20), nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)  # '1m', '5m', '15m', '1h', '4h', '1d'
    parameters = db.Column(db.JSON, nullable=False)  # Strategy-specific parameters
    weight = db.Column(db.Float, default=1.0)  # Weight for strategy combination
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradingSession(db.Model):
    """Trading session with performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    initial_balance = db.Column(db.Float, nullable=False)
    final_balance = db.Column(db.Float)
    pnl = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    is_backtest = db.Column(db.Boolean, default=False)
    
    # Relationships
    trades = db.relationship('Trade', backref='session', lazy=True)

class Trade(db.Model):
    """Individual trade record"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('trading_session.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    order_id = db.Column(db.String(50))
    strategy = db.Column(db.String(50), nullable=False)
    direction = db.Column(db.String(10), nullable=False)  # 'long' or 'short'
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    quantity = db.Column(db.Float, nullable=False)
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime)
    pnl = db.Column(db.Float)
    pnl_percentage = db.Column(db.Float)
    fees = db.Column(db.Float)
    status = db.Column(db.String(20), default='open')  # 'open', 'closed', 'cancelled'
    notes = db.Column(db.Text)
