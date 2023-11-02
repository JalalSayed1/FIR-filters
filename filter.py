import numpy as np
import matplotlib.pyplot as plt
from ultils import *

class Filter:

    def __init__(self, cutoff_freqs, fs):
        '''
        cutoff_freqs: list of cutoff frequencies.
        fs: sampling rate of the data.
        '''
        self.cutoff_freqs = cutoff_freqs
        self.fs = fs
        self.M = 2000 # number of taps/coefficients
        self.buffer = [0] * self.M

    
    def calculate_coefficients(self):
        '''Returns the coefficients of the filter.'''
        
        low_freq, high_freq = self.cutoff_freqs
        fs = self.fs

        # X(k) is the frequency response of the filter:
        X = np.ones(self.M)

        # Multiply with M to get the index of the frequency as Fs is normalised frequency.
        first_index = int(low_freq * self.M / fs)
        second_index= int(high_freq * self.M / fs)

        # Set the values of the frequency response to 0.
        X[first_index:second_index+1] = 0 # for the first notch
        X[self.M - second_index:self.M - first_index+1] = 0 # for the mirrored notch

        # self.plot(X, label='Frequency response')

        # x(n) is the sample domain of the filter:
        x = np.fft.ifft(X)
        # real value only so we don't take the imaginary part of the signal and leave the real part as it is.
        x_real = np.real(x)

        h = np.zeros(self.M)
        # Shift the signal to make it causal and swap -ve and +ve time:
        mid = int(self.M / 2)
        h[0:mid] = x_real[mid:self.M]
        h[mid:self.M] = x_real[0:mid]

        # fig = plt.figure(1)
        # self.plot(h, label=' Before Frequency response', figure=fig)

        # h = h * np.hamming(self.M)

        # self.plot(h, label='After Frequency response', figure=fig)

        # store the impulse response of the filter:
        self.h = h
        
        return h
    

    def plot(self, h=None, figure=None, label=''):
        '''Plots the filter.'''

        if h is None:
            h = self.h

        if figure is None:
            figure = plt.figure()
        else:
            plt.figure(figure.number)

        h_db = 20 * np.log10(np.abs(h))
        # plt.plot(np.linspace(0,500,len(h)), h_db, label=label)
        # plt.plot(h)



# time, pulse = read_file("raw_data/person2_standing.dat")
# fs = int(len(pulse) / time[-1])
# filter_obj = Filter([45, 55], fs)
# # plt.plot(time1, pulse)s
# h = filter_obj.calculate_coefficients()
# # plt.plot(h, label='Frequency response')

# plt.legend()
# plt.show()