import os

class Config:
    """Configuration settings for the trading bot"""
    # System paths
    DATA_DIRECTORY = os.path.join(os.getcwd(), "data")
    
    # Delta Exchange API settings
    DELTA_EXCHANGE_BASE_URL = "https://api.delta.exchange/v2"
    # API credentials should come from the user's settings, these are just for testing
    DELTA_EXCHANGE_API_KEY = os.environ.get("DELTA_EXCHANGE_API_KEY", "")
    DELTA_EXCHANGE_API_SECRET = os.environ.get("DELTA_EXCHANGE_API_SECRET", "")
    
    # Trading settings
    DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"]
    DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    # Strategy default parameters
    TREND_FOLLOWING_DEFAULTS = {
        "short_window": 20,
        "long_window": 50,
        "take_profit": 3.0,  # percentage
        "stop_loss": 2.0     # percentage
    }
    
    MEAN_REVERSION_DEFAULTS = {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "take_profit": 2.0,  # percentage
        "stop_loss": 2.0     # percentage
    }
    
    BREAKOUT_DEFAULTS = {
        "period": 20,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "take_profit": 4.0,  # percentage
        "stop_loss": 2.0     # percentage
    }
    
    MOMENTUM_DEFAULTS = {
        "period": 14,
        "threshold": 0.5,
        "take_profit": 3.0,  # percentage
        "stop_loss": 2.0     # percentage
    }
    
    # RL model settings
    RL_MODEL_PATH = "models/ppo_model"
    OBSERVATION_WINDOW = 30
    REWARD_SCALING = 1.0
    TRAINING_EPISODES = 1000
    
    # Backtesting settings
    BACKTEST_START_DATE = "2023-01-01"
    BACKTEST_END_DATE = None  # Will default to current date
    
    # UI settings
    COLORS = {
        "primary": "#0A2647",
        "secondary": "#144272",
        "success": "#25D366",
        "danger": "#FF4444",
        "background": "#000000",
        "text": "#FFFFFF",
        "chart": "#2196F3"
    }
