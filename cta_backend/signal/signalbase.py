import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from scipy.signal import butter, filtfilt

class Normalisation(Enum):
    NONE = 1
    RANK = 2
    MINIMAX = 3
    ZSCORE = 4
    ICDF = 5

    """_summary_
    3,2,5,1 --> 3,1,0,2 -> 2,1,3,0
    Returns:
        _type_: _description_
    """

class Signal(ABC):
    
    def __init__(self):
        pass
        
    @abstractmethod
    def compute_signal(self,series:pd.Series)->pd.Series:
        pass
    
   
   
class SignalZScores(Signal):
    def __initi__(self):
        super().__init__()
    
    def compute_signal(self,series:pd.Series)->pd.Series:
        return (series-np.mean(series))/np.std(series)
    


class SignalLinearCompression(Signal):
    def __initi__(self,lower_bound:float, upper_bound:float):
        super().__init__()
        self.a = lower_bound
        self.b = upper_bound
    
    def compute_signal(self,series:pd.Series)->pd.Series:
        min_s = np.min(series)
        max_s = np.max(series)
        return self.a + (series-min_s)(max_s-min_s)*(self.b-self.a)
            
    
class SignalCMA(Signal):
    def __initi__(self,short_window:int, long_window:int):
        super().__init__()
        self.short_window = short_window
        self.long_window= long_window
    
    def compute_signal(self,series:pd.Series)->pd.Series:
        return series.rolling(self.short_window)-series.rolling(self.long_window)
    
    
class SignalCMAButterworth(Signal):
    def __initi__(self,short_window:int, long_window:int, order:int):
        super().__init__()
        self.short_window = short_window
        self.long_window= long_window
        self.order=order
    
    def compute_signal(self,series:pd.Series)->pd.Series:
        fs = 1 # like one point per day, or week etc... here we assume one unit of time and sample frequency of 1 sample per unit
        
        def convert_window_to_cutoff_frequency(nWindow:int, fs:float):
            dt = 1/fs
            return 1.0/(nWindow*dt)
        # Calculate the Nyquist frequency
        nyquist = 0.5 * fs
        
        flow = convert_window_to_cutoff_frequency(self.long_window,fs=1)
        fhi = convert_window_to_cutoff_frequency(self.short_window,fs=1)
        
        #Doing low frequency ...
        flow_norm = flow / nyquist                        
        b, a = butter(self.order, flow_norm, btype='low', analog=False)                
        filtered_low= filtfilt(b, a, series)
        
        #Doing high frequency ...
        fhig_norm = fhi / nyquist                        
        b, a = butter(self.order, fhig_norm, btype='low', analog=False)                
        filtered_hi= filtfilt(b, a, series)
        
        final_hi = pd.Series(filtered_hi, index=series.index)
        final_low = pd.Series(filtered_low, index=series.index)
        
        # Return the filtered result as a pandas Series
        return final_hi - final_low