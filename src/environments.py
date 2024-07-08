import numpy as np

BUY = 1
SELL = 2

class StockTradingEnv:
    def __init__(self, df, initial_balance=1000000, commission_rate=0.00015, sequence_length=5):
        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.sequence_length = sequence_length
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.sequence_length - 1
        self.total_steps = len(self.df) - 1
        return self._get_observation()

    def _get_observation(self):
        obs = []
        for i in range(self.sequence_length):
            step = self.current_step - self.sequence_length + 1 + i
            obs.append([
                self.balance,
                self.shares_held,
                self.df.iloc[step]['Close'],
                self.df.iloc[step]['Open'],
                self.df.iloc[step]['High'],
                self.df.iloc[step]['Low']
            ])
        return np.array(obs)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        self.current_step += 1
        done = self.current_step == self.total_steps

        action_type = action  # 0: Hold, 1: Buy, 2: Sell
        action_amount = 1.0  # 항상 전체 금액/주식을 사용

        if action_type == BUY:  # Buy
            amount_to_invest = self.balance
            shares_to_buy = int(amount_to_invest // current_price)
            cost = shares_to_buy * current_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.shares_held += shares_to_buy

        elif action_type == SELL:  # Sell
            shares_to_sell = self.shares_held
            sale_value = shares_to_sell * current_price
            commission = sale_value * self.commission_rate
            self.balance += sale_value - commission
            self.shares_held -= shares_to_sell

        reward = (self.balance + self.shares_held * current_price - self.initial_balance) / self.initial_balance
        return self._get_observation(), reward, done, {}

    def backtest(self, model):
        self.reset()
        done = False
        portfolio_values = [self.initial_balance]
        while not done:
            state = self._get_observation()
            action = model.act(state)
            _, reward, done, _ = self.step(action)
            portfolio_value = self.balance + self.shares_held * self.df.loc[self.current_step, 'Close']
            portfolio_values.append(portfolio_value)
        return portfolio_values
        
    def calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.01):
        daily_returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                        for i in range(1, len(portfolio_values))]
        excess_returns = [r - risk_free_rate/252 for r in daily_returns]  # Assuming 252 trading days
        if len(excess_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
            return sharpe_ratio
        else:
            return 0  # 또는 다른 적절한 기본값