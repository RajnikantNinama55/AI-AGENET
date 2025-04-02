import numpy as np
import pandas as pd
import gym
from gym import spaces
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Trading Environment for Reinforcement Learning
    
    This environment simulates a trading scenario for training RL models.
    It processes historical data and provides observations, rewards, and
    handles actions to buy, sell, or hold.
    """
    
    def __init__(self, historical_data, initial_balance=10000, commission=0.001, window_size=30):
        """
        Initialize the trading environment
        
        Args:
            historical_data: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Trading commission (percentage)
            window_size: Size of the observation window
        """
        super(TradingEnvironment, self).__init__()
        
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        
        # Validate data
        if historical_data is None or len(historical_data) == 0:
            raise ValueError("Historical data cannot be empty")
        
        # Required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Environment attributes
        self.data_length = len(historical_data)
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_price = 0
        self.position_size = 0
        self.portfolio_value = initial_balance
        self.cumulative_reward = 0
        self.trade_count = 0
        self.strategy_performance = {}
        
        # Define action and observation spaces
        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: window_size * features (OHLCV normalized + technical indicators)
        # Number of features depends on what we include
        # For now, let's use OHLCV + position info
        n_features = 5 + 3  # OHLCV + position indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, n_features), dtype=np.float32
        )
    
    def reset(self):
        """
        Reset the environment to its initial state
        
        Returns:
            observation: Initial observation
        """
        # Reset environment state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.position_size = 0
        self.portfolio_value = self.initial_balance
        self.cumulative_reward = 0
        self.trade_count = 0
        self.strategy_performance = {}
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)
            
        Returns:
            observation: New observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        if self.current_step >= self.data_length - 1:
            # If we've reached the end of data, force selling and end episode
            return self._get_observation(), 0, True, {'reason': 'data_end'}
        
        # Get current price data
        current_price = self.historical_data.iloc[self.current_step]['close']
        
        # Process action
        reward = 0
        info = {}
        
        if action == 1:  # Buy
            if self.position == 0:  # If no position, open long
                max_size = self.balance / current_price
                self.position_size = max_size * 0.95  # Use 95% of available balance
                self.position_price = current_price
                self.balance -= self.position_size * current_price * (1 + self.commission)
                self.position = 1
                self.trade_count += 1
                info['action'] = 'buy_open'
                
            elif self.position == -1:  # If short position, close it
                close_value = self.position_size * self.position_price
                current_value = self.position_size * current_price
                profit = close_value - current_value - (close_value + current_value) * self.commission
                self.balance += close_value + profit
                self.position = 0
                self.position_price = 0
                self.position_size = 0
                self.trade_count += 1
                info['action'] = 'buy_close'
                info['profit'] = profit
                
                # Apply reward based on profit
                reward = profit / self.initial_balance * 10  # Scale reward
                
        elif action == 2:  # Sell
            if self.position == 0:  # If no position, open short
                max_size = self.balance / current_price
                self.position_size = max_size * 0.95  # Use 95% of available balance
                self.position_price = current_price
                self.balance -= self.position_size * current_price * self.commission  # Only pay commission for opening
                self.position = -1
                self.trade_count += 1
                info['action'] = 'sell_open'
                
            elif self.position == 1:  # If long position, close it
                close_value = self.position_size * current_price
                open_value = self.position_size * self.position_price
                profit = close_value - open_value - (close_value + open_value) * self.commission
                self.balance += close_value
                self.position = 0
                self.position_price = 0
                self.position_size = 0
                self.trade_count += 1
                info['action'] = 'sell_close'
                info['profit'] = profit
                
                # Apply reward based on profit
                reward = profit / self.initial_balance * 10  # Scale reward
                
        else:  # Hold
            # Small negative reward for holding to encourage action
            reward = -0.001
            info['action'] = 'hold'
        
        # Calculate current portfolio value
        portfolio_value = self.balance
        if self.position == 1:  # Long position
            portfolio_value += self.position_size * current_price
        elif self.position == -1:  # Short position
            portfolio_value += self.position_size * (2 * self.position_price - current_price)
        
        # Update portfolio value and calculate change
        pct_change = (portfolio_value - self.portfolio_value) / self.portfolio_value
        self.portfolio_value = portfolio_value
        
        # Add small reward for portfolio growth
        reward += pct_change * 5
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.data_length - 1
        
        # If done, add final portfolio value to info
        if done:
            info['final_value'] = self.portfolio_value
            info['return'] = (self.portfolio_value - self.initial_balance) / self.initial_balance
            info['trades'] = self.trade_count
        
        # Get new observation
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
            ndarray: Observation data
        """
        # Extract window of data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        if start_idx < 0:
            # If we don't have enough data yet, pad with zeros
            padding = np.zeros((-start_idx, 5))
            data_window = self.historical_data.iloc[0:end_idx][['open', 'high', 'low', 'close', 'volume']].values
            market_data = np.vstack((padding, data_window))
        else:
            market_data = self.historical_data.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize market data
        market_data = self._normalize_data(market_data)
        
        # Create position features
        position_data = np.zeros((self.window_size, 3))
        
        # Fill last row with current position info
        position_data[-1, 0] = self.position  # -1, 0, or 1
        position_data[-1, 1] = self.position_price / market_data[-1, 3] if self.position_price > 0 else 0  # Normalized to current price
        position_data[-1, 2] = self.position_size * market_data[-1, 3] / self.portfolio_value if self.position_size > 0 else 0  # Size as fraction of portfolio
        
        # Combine market and position data
        obs = np.hstack((market_data, position_data))
        
        return obs.astype(np.float32)
    
    def _normalize_data(self, data):
        """
        Normalize the OHLCV data
        
        Args:
            data: OHLCV data array
            
        Returns:
            ndarray: Normalized data
        """
        result = data.copy()
        
        # If we have previous close price, normalize OHLC by it
        if self.current_step > 0:
            last_close = self.historical_data.iloc[self.current_step - 1]['close']
            
            # Normalize OHLC by dividing by last close
            result[:, 0] = data[:, 0] / last_close
            result[:, 1] = data[:, 1] / last_close
            result[:, 2] = data[:, 2] / last_close
            result[:, 3] = data[:, 3] / last_close
        
        # Normalize volume using z-score on the window
        if len(data) > 1:
            volume_mean = np.mean(data[:, 4])
            volume_std = np.std(data[:, 4])
            
            if volume_std > 0:
                result[:, 4] = (data[:, 4] - volume_mean) / volume_std
        
        return result
    
    def update_strategy_performance(self, strategy_name, performance):
        """
        Update the performance metrics for a strategy
        
        Args:
            strategy_name: Name of the strategy
            performance: Performance value
        """
        self.strategy_performance[strategy_name] = performance
