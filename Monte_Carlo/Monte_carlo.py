import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

years = 15
enddate = dt.datetime.now()
startdate = enddate- dt.timedelta(days = 365*years)

tickers = ['SPY', 'BND', ' QQQ','GLD', 'VTI']

adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start = startdate, end = enddate)
    adj_close_df[ticker] = data['Adj Close']

log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()

def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)

def std_dev(weights, cov_matrix):
    variance = weights.T @ cov_matrix @weights
    return np.sqrt(variance)

cov_matrix = log_returns.cov()

portfolio_value = 1000000
weights = np.array([1/len(tickers)]*len(tickers))
portfolio_expected_return = expected_return(weights, log_returns)
portfolio_std_dev = std_dev(weights, cov_matrix)

def random_z_score():
    return np.random.normal(0,1)

days = 5

def scencario_gain_loss (portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score*np.sqrt(days)


simulation = 10000

scenario_return = []

for i in range(simulation):
    z_score = random_z_score()
    scenario_return.append(scencario_gain_loss(portfolio_value,portfolio_std_dev, z_score, days))


confidence_interval = 0.95
VaR = np.percentile(scenario_return, 100 * (1 - confidence_interval))

plt.hist(scenario_return, bins =50, density=True)
plt.xlabel('Scenario Gain/Loss($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss over{days} Days')
plt.axvline(-VaR, color = 'r', linestyle= 'dashed', linewidth = 2, label = f'VaR at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()
