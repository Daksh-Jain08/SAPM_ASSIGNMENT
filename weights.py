import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Read the Excel file
file_path = 'Sapm_Assignment.xlsx'
sheet_name = 'Daily_Returns'

# Load daily returns data
returns_df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4, index_col=0)

# Select the 5 stocks for the portfolio
stocks = ['GODREJAGRO_R', 'BAJAJ_R', 'ITC_R', 'HDFC_R', 'TCS_R']
returns = returns_df[stocks].dropna()

# Calculate covariance matrix
cov_matrix = returns.cov()

# Minimum Variance Portfolio
n = len(stocks)
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def minimize_variance(cov_matrix, n):
    def objective(weights):
        return portfolio_variance(weights, cov_matrix)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(objective, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

min_var_weights = minimize_variance(cov_matrix.values, n)

# Sharpe Mimic Portfolio
# Calculate Nifty Sharpe
nifty_returns = returns_df['NIFTY50_R'].dropna()
annual_risk_free = 0.06
nifty_mean = nifty_returns.mean() * 252
nifty_std = nifty_returns.std() * np.sqrt(252)
nifty_sharpe = (nifty_mean - annual_risk_free) / nifty_std

# Function to calculate portfolio Sharpe
def portfolio_sharpe(weights, returns, risk_free_rate):
    mean_return = np.sum(returns.mean() * weights) * 252
    std_return = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

# Objective: minimize squared difference between portfolio Sharpe and Nifty Sharpe
def objective_sharpe(weights, returns, nifty_sharpe, risk_free_rate):
    sharpe = portfolio_sharpe(weights, returns, risk_free_rate)
    return (sharpe - nifty_sharpe) ** 2

# Minimize the difference
risk_free_rate = annual_risk_free

def minimize_sharpe_diff(returns, nifty_sharpe, risk_free_rate, n):
    def objective(weights):
        return objective_sharpe(weights, returns, nifty_sharpe, risk_free_rate)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(objective, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

sharpe_mimic_weights = minimize_sharpe_diff(returns, nifty_sharpe, risk_free_rate, n)

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Stock': stocks,
    'Min Variance Weight': min_var_weights,
    'Sharpe Mimic Weight': sharpe_mimic_weights
})

# Save the results to a new Excel file
results_df.to_excel('Portfolio_Weights_Results.xlsx', index=False)

print("Portfolio weights calculated and saved to 'Portfolio_Weights_Results.xlsx'")
