import pandas as pd
from scipy.signal import butter, filtfilt
from typing import List

class SingalTrendFilter:
    
    @staticmethod
    def convert_window_to_cutoff_frequency(nWindow:int, fs:float):
        dt = 1/fs
        return 1.0/(nWindow*dt)

    @staticmethod
    def butter_lowpass_filter(series: pd.Series, cutoff: float, fs: float, order: int = 4) -> pd.Series:
        # Calculate the Nyquist frequency
        nyquist = 0.5 * fs
        
        # Normalize the cutoff frequency
        normal_cutoff = cutoff / nyquist
        
        # Get the Butterworth filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply the filter to the series
        filtered_series = filtfilt(b, a, series)
        
        # Return the filtered result as a pandas Series
        return pd.Series(filtered_series, index=series.index)