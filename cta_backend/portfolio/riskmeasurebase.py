import pandas as pd
import numpy as np
from typing import List,Dict,Callable
from scipy.optimize import minimize
from abc import ABC, abstractmethod

class RiskBudgettingPortfolio (ABC):
    def __init__(): 
        pass
    
    @abstractmethod
    def RiskMetric(cov:np.ndarray, w: np.ndarray) -> float:
        pass
    
    
    @abstractmethod
    def RiskContributions(cov:np.ndarray, w: np.ndarray) -> np.ndarray:
        pass
    
   
    def risk_contributions(self, weights:np.array, cov_matrix:np.ndarray):
        port_vol = self.fMetric(weights, cov_matrix)
        # Marginal risk contributions
        mrc = cov_matrix @ weights
        # Total risk contribution for each asset
        rc = weights * mrc / port_vol
        return rc

    def objective(self, weights, cov_matrix, risk_budgets):
        # Portfolio volatility
        port_vol = self.fMetric(weights, cov_matrix)
        # Risk contributions
        rc = self.fRC(weights, cov_matrix)
        # Normalize risk contributions
        rc_percent = rc / port_vol
        # Objective: Minimize the difference between actual and target risk contributions
        return np.sum((rc_percent - risk_budgets) ** 2)

    def compute_risk_budgeted_weights(self,cov_matrix, risk_budgets):
        # Number of assets
        n_assets = len(risk_budgets)
        
        # Initial guess for weights (equal-weighted)
        initial_weights = np.ones(n_assets) / n_assets

        # Constraints: Weights sum to 1, and no short-selling (weights >= 0)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = [(0, 1) for _ in range(n_assets)]

        # Optimization
        result = minimize(self.objective, initial_weights, args=(cov_matrix, risk_budgets),
                        method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x
        
        
