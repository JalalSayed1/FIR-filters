import numpy as np
from ultils import *
import matplotlib.pyplot as plt


class LMSfilter:

    def __init__(self, _coefficients):
        self.ntaps = len(_coefficients)
        self.coefficients = _coefficients
        self.buffer = np.zeros(self.ntaps)

    def doFilter(self, v):
        '''FIR filter operation.'''
        self.buffer = np.roll(self.buffer, shift=1)
        self.buffer[0] = v
        
        return np.sum(self.buffer * self.coefficients)  # dot product

    def doFilterAdaptive(self, signal, noise, learningRate):
        '''LMS adaptive filter
        signal: signal to be filtered
        noise: noise to be removed 
        learningRate: learning rate of the filter
        '''
        error = signal - self.doFilter(noise)
        # error = self.dofilter(signal) - noise

        # update coefficients
        for i in range(self.ntaps):
            self.coefficients[i] = self.coefficients[i] + learningRate * error * self.buffer[i]
            
        # self.coefficients = self.coefficients + learningRate * error * self.buffer

        canceller = signal - error
        output_signal = signal - canceller
        # return cleaned up ECG like before
        return output_signal

