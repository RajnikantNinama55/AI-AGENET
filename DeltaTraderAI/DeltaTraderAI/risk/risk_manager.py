import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk Manager for trading strategies
    
    Handles risk management features including:
    - Position sizing based on account balance
    - Stop loss management
    - Daily loss limits
    - Maximum drawdown protection
    - Trade frequency limits
    - Volatility-based risk adjustment
    """
    
    def __init__(self, account_balance=10000.0):
        """
        Initialize the risk manager
        
        Args:
            account_balance (float): Initial account balance
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        
        # Risk parameters
        self.max_position_size_pct = 0.02  # 2% of account per trade
        self.min_position_size_pct = 0.01  # 1% of account per trade
        self.stop_loss_pct = 0.02  # 2% below entry price
        self.daily_loss_limit_pct = 0.05  # 5% daily loss limit
        self.max_drawdown_pct = 0.15  # 15% maximum drawdown
        self.max_trades_per_day = 10  # Maximum trades per day
        self.volatility_lookback = 20  # Lookback period for volatility calculation
        
        # Tracking
        self.trades = []
        self.daily_pnl = 0.0
        self.drawdown = 0.0
        self.peak_balance = account_balance
        self.trade_count_today = 0
        self.last_trade_time = None
        self.trading_paused = False
        self.pause_reason = None
        
        logger.info(f"Risk Manager initialized with balance: ${account_balance:.2f}")
    
    def update_account_balance(self, balance):
        """
        Update the current account balance
        
        Args:
            balance (float): New account balance
        """
        self.account_balance = balance
        
        # Update peak balance if current balance is higher
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate current drawdown
        self.drawdown = (self.peak_balance - balance) / self.peak_balance
        
        # Check if max drawdown is exceeded
        if self.drawdown > self.max_drawdown_pct:
            self.trading_paused = True
            self.pause_reason = f"Maximum drawdown of {self.max_drawdown_pct*100}% exceeded"
            logger.warning(f"Trading paused: {self.pause_reason}")
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price=None, volatility=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol (str): Trading symbol
            entry_price (float): Entry price
            stop_loss_price (float, optional): Stop loss price, if None will use default percentage
            volatility (float, optional): Market volatility, if provided will adjust risk
            
        Returns:
            float: Position size in base currency
        """
        if self.trading_paused:
            logger.warning(f"Cannot calculate position size: trading is paused ({self.pause_reason})")
            return 0.0
        
        # Calculate stop loss if not provided
        if stop_loss_price is None:
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        
        # Calculate risk per trade (% of account)
        risk_pct = self.max_position_size_pct
        
        # Adjust risk based on volatility if provided
        if volatility is not None:
            # Higher volatility means lower position size
            volatility_factor = 1.0 - min(0.5, volatility / 100)  # Cap at 50% reduction
            risk_pct = risk_pct * volatility_factor
            
            # Ensure minimum position size
            risk_pct = max(risk_pct, self.min_position_size_pct)
        
        # Calculate risk amount in dollars
        risk_amount = self.account_balance * risk_pct
        
        # Calculate position size based on stop loss
        position_size_in_quote = risk_amount
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        # Convert to base currency units
        position_size_in_base = position_size_in_quote / entry_price
        
        # Adjust position size based on risk per unit
        if risk_per_unit > 0:
            max_position_size = risk_amount / risk_per_unit
            position_size_in_base = min(position_size_in_base, max_position_size)
        
        logger.info(f"Calculated position size: {position_size_in_base:.4f} @ {entry_price} (risk: ${risk_amount:.2f})")
        
        return position_size_in_base
    
    def check_daily_loss_limit(self):
        """
        Check if daily loss limit has been exceeded
        
        Returns:
            bool: True if trading should continue, False if it should pause
        """
        # Calculate daily loss percentage
        daily_loss_pct = abs(self.daily_pnl) / self.initial_balance if self.daily_pnl < 0 else 0
        
        # Check if daily loss limit is exceeded
        if daily_loss_pct >= self.daily_loss_limit_pct:
            self.trading_paused = True
            self.pause_reason = f"Daily loss limit of {self.daily_loss_limit_pct*100}% exceeded"
            logger.warning(f"Trading paused: {self.pause_reason}")
            return False
        
        return True
    
    def check_trade_frequency(self):
        """
        Check if trade frequency limits are exceeded
        
        Returns:
            bool: True if trading should continue, False if it should pause
        """
        # Reset trade count if it's a new day
        today = datetime.now().date()
        
        if self.last_trade_time is not None:
            last_trade_date = self.last_trade_time.date()
            if last_trade_date < today:
                self.trade_count_today = 0
                self.daily_pnl = 0.0
                
                # If trading was paused due to daily limits, unpause
                if self.trading_paused and self.pause_reason and "daily" in self.pause_reason.lower():
                    self.trading_paused = False
                    self.pause_reason = None
                    logger.info("Trading unpaused - new day started")
        
        # Check if max trades per day is reached
        if self.trade_count_today >= self.max_trades_per_day:
            self.trading_paused = True
            self.pause_reason = f"Maximum trades per day ({self.max_trades_per_day}) reached"
            logger.warning(f"Trading paused: {self.pause_reason}")
            return False
        
        return True
    
    def record_trade(self, trade):
        """
        Record a completed trade and update metrics
        
        Args:
            trade (dict): Trade details including entry_price, exit_price, size, pnl, etc.
            
        Returns:
            bool: True if successfully recorded
        """
        self.trades.append(trade)
        self.last_trade_time = datetime.now()
        self.trade_count_today += 1
        
        # Update daily PnL
        self.daily_pnl += trade.get('pnl', 0)
        
        # Update account balance
        self.update_account_balance(self.account_balance + trade.get('pnl', 0))
        
        # Check daily loss limit
        self.check_daily_loss_limit()
        
        # Check trade frequency
        self.check_trade_frequency()
        
        logger.info(f"Trade recorded: PnL ${trade.get('pnl', 0):.2f}, Daily PnL: ${self.daily_pnl:.2f}")
        
        return True
    
    def calculate_market_volatility(self, price_data, window=None):
        """
        Calculate market volatility
        
        Args:
            price_data (DataFrame or Series): Historical price data
            window (int, optional): Lookback window, defaults to self.volatility_lookback
            
        Returns:
            float: Volatility as percentage
        """
        if window is None:
            window = self.volatility_lookback
        
        if isinstance(price_data, pd.DataFrame) and 'close' in price_data:
            price_series = price_data['close']
        else:
            price_series = price_data
        
        if len(price_series) < window:
            logger.warning(f"Insufficient data for volatility calculation (need {window}, got {len(price_series)})")
            return None
        
        # Calculate daily returns
        returns = price_series.pct_change().dropna()
        
        # Calculate standard deviation of returns
        volatility = returns.tail(window).std() * np.sqrt(252)  # Annualized
        
        return volatility * 100  # Return as percentage
    
    def adjust_stop_loss(self, position, current_price, adjust_pct=0.5):
        """
        Adjust stop loss for an existing position (trailing stop)
        
        Args:
            position (dict): Current position details
            current_price (float): Current market price
            adjust_pct (float): Percentage of profit to protect (0.5 = 50%)
            
        Returns:
            float: New stop loss price or None if no adjustment needed
        """
        direction = position.get('direction')
        entry_price = position.get('entry_price')
        current_stop = position.get('stop_loss')
        
        if not all([direction, entry_price, current_stop]):
            logger.warning("Cannot adjust stop loss: missing position details")
            return None
        
        new_stop = None
        
        if direction == 'long' and current_price > entry_price:
            # For long positions, calculate profit and adjust stop
            profit = current_price - entry_price
            potential_stop = entry_price + (profit * adjust_pct)
            
            # Only move stop up, never down
            if potential_stop > current_stop:
                new_stop = potential_stop
                logger.info(f"Adjusted stop loss: {current_stop:.2f} -> {new_stop:.2f}")
                
        elif direction == 'short' and current_price < entry_price:
            # For short positions, calculate profit and adjust stop
            profit = entry_price - current_price
            potential_stop = entry_price - (profit * adjust_pct)
            
            # Only move stop down, never up
            if potential_stop < current_stop:
                new_stop = potential_stop
                logger.info(f"Adjusted stop loss: {current_stop:.2f} -> {new_stop:.2f}")
        
        return new_stop
    
    def can_open_position(self, symbol, price, direction=None):
        """
        Check if a new position can be opened based on risk rules
        
        Args:
            symbol (str): Trading symbol
            price (float): Current price
            direction (str, optional): 'long' or 'short'
            
        Returns:
            bool: True if position can be opened
        """
        # Check if trading is paused
        if self.trading_paused:
            logger.warning(f"Cannot open position: trading is paused ({self.pause_reason})")
            return False
        
        # Check daily loss limit
        if not self.check_daily_loss_limit():
            return False
        
        # Check trade frequency limit
        if not self.check_trade_frequency():
            return False
        
        # Check if max drawdown is exceeded
        if self.drawdown > self.max_drawdown_pct:
            self.trading_paused = True
            self.pause_reason = f"Maximum drawdown of {self.max_drawdown_pct*100}% exceeded"
            logger.warning(f"Cannot open position: {self.pause_reason}")
            return False
        
        return True
    
    def get_risk_metrics(self):
        """
        Get current risk metrics
        
        Returns:
            dict: Risk metrics
        """
        return {
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.initial_balance) * 100,
            'drawdown': self.drawdown * 100,  # as percentage
            'peak_balance': self.peak_balance,
            'trade_count_today': self.trade_count_today,
            'trading_paused': self.trading_paused,
            'pause_reason': self.pause_reason
        }
    
    def reset_daily_metrics(self):
        """Reset daily trading metrics"""
        self.daily_pnl = 0.0
        self.trade_count_today = 0
        
        # If trading was paused due to daily limits, unpause
        if self.trading_paused and self.pause_reason and "daily" in self.pause_reason.lower():
            self.trading_paused = False
            self.pause_reason = None
            logger.info("Trading unpaused - daily metrics reset")