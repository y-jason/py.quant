"""
Portfolio optimization is the process of selecting the best portfolio (combination of assets) that maximizes expected returns and minimizes risk. It involves analyzing historical data, calculating risk and return metrics, and using optimization algorithms to identify the optimal portfolio.

This script uses the provided input of tickers and start/end dates to calculate the optimal position weights, expected return, and volatility. The efficient frontier is plotted using pandas and numpy, with data sourced from Alpha Vantage API. To use the script, you need to obtain a free API key from https://www.alphavantage.co/support/#api-key.

Optimal weights: [1.06057523e-07 8.07545661e-08 8.67556946e-08 8.77734304e-08 9.99999639e-01]
Expected portfolio return: 0.009341859930051025
Portfolio volatility: 0.036942427279929135
""" 

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

url = 'https://www.alphavantage.co/query'
api_key = 'API_KEY'
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
start_date = '2023-01-01'
end_date = '2023-04-01'

df_list = []
for ticker in tickers:
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'outputsize': 'full',
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    data = pd.DataFrame.from_dict(response.json()['Time Series (Daily)'], orient='index')
    data = data.sort_index().loc[start_date:end_date, '5. adjusted close'].astype(float).rename(ticker)
    df_list.append(data)

df = pd.concat(df_list, axis=1).dropna()

# Calculate expected returns and covariance matrix
returns = df.pct_change().mean()
returns_arr = returns.to_numpy()
cov_matrix = df.pct_change().cov()

# Define optimization problem
n = len(tickers)
w = cp.Variable(n)
objective = cp.Maximize(returns_arr @ w)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()

print('Optimal weights:', w.value)
print('Expected portfolio return:', returns @ w.value)
print('Portfolio volatility:', np.sqrt(w.value @ cov_matrix @ w.value))

# Plot efficient frontier
target_returns = np.linspace(0, 0.4, 100)
volatility = []
for target_return in target_returns:
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [returns_arr @ w >= target_return, cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    volatility.append(np.sqrt(problem.value))
plt.plot(volatility, target_returns)
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.show()