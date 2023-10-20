import numpy as np
import matplotlib.pyplot as plt

class Filter:
    def __init__(self, low_freq, high_freq, fs):
        '''Initialize the filter.
        low_freq: lower bound of the filter.
        high_freq: upper bound of the filter.
        fs: sampling rate of the data. 
        '''
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        
        
    def make_data_stream(self, low_freq, high_freq, fs):
        '''make numpy array of 0's and 1's to represent the filter. Everything in between low_freq and high_freq is 0, everything else is 1.
        low_freq: lower bound of the filter.'''
    



