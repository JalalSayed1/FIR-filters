import numpy as np

def read_file(filename):
    '''Returns the time and pulse data from the file.'''
    data = np.loadtxt(filename)
    time, pulse1, pulse2, pulse3 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return time, pulse2

def calculate_sampling_rate(data_len, end_time):
        '''Returns the sampling rate of the data (Hz).
        data_len: Number of data points.
        end_time: time length (end time) of the data.
        '''
        return int(data_len / end_time)
    
def my_fft(data, rate):
    '''Returns the frequency spectrum of the data.'''
    fft_data = (np.fft.fft(data)) / len(data)
    freqs = np.linspace(0, rate/2, num=len(data))
    return freqs, fft_data


def my_ifft(fft_data):
    '''Returns the time domain of the frequency spectrum.'''
    return np.fft.ifft(fft_data) * len(fft_data)


def freq_to_index(freq, length, rate):
    '''Returns the index of the frequency in the frequency spectrum.'''
    return int(freq * length / rate)