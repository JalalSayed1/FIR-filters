import numpy as np
import matplotlib.pyplot as plt

class FIR_filter:
    def __init__(self, low_freq, high_freq, fs):
        # self.coeffs = coeffs
        # self.buffer = [0] * len(coeffs)
        
        
    def read_file(self, filename):
        '''Returns the time and pulse data from the file.'''
        data = np.loadtxt(filename)
        time, pulse1, pulse2, pulse3 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        return time, pulse2
    
    def calculate_sampling_rate(self, data_len, end_time):
        '''Returns the sampling rate of the data (Hz).
        data_len: Number of data points.
        end_time: time length (end time) of the data.
        '''
        return int(data_len / end_time)
    
    def my_fft(self, data, rate):
        '''Returns the frequency spectrum of the data.'''
        fft_data = (np.fft.fft(data)) / len(data)
        freqs = np.linspace(0, rate/2, num=len(data))
        return freqs, fft_data


    def my_ifft(self, fft_data):
        '''Returns the time domain of the frequency spectrum.'''
        return np.fft.ifft(fft_data) * len(fft_data)
    
    
fir_filter = FIR_filter([1, 2, 3, 4])
time1, pulse1 = read_file("raw_data/dead_tamim.dat")
fs1 = calculate_sampling_rate(len(pulse1), time[-1])
# freq_pulse = my_fft(pulse, 1/0.0000001)[1]
plt.plot(time1, pulse1)
plt.legend()
plt.show()