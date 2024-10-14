from riskmeasurebase import RiskBudgettingPortfolio
import numpy as np

class Volatility(RiskBudgettingPortfolio):
    def RiskMetric(self,cov:np.ndarray, w: np.ndarray) -> float:
        return np.sqrt( w.T @ cov @ w )
    
    
    def RiskContributions(self, cov:np.ndarray, w: np.ndarray) -> np.ndarray:
        mrc = cov @ w
        v = self.RiskMetric(cov,w)
        rc = w * mrc/v