import numpy as np
from ultils import *
import matplotlib.pyplot as plt


def calculate_coefficients(fs, cutoff_freqs):
    '''Returns the coefficients of the filter.'''
    
    # calc resolution:
    # resolution = int(fs / M)

    # number of taps/coefficients
    M = 200

    low_freq, high_freq = cutoff_freqs
    if high_freq == 0:
        high_freq = fs
        

    # X(k) is the frequency response of the filter:
    X = np.ones(M)

    # Multiply with M to get the index of the frequency as Fs is normalised frequency.
    first_index = int(low_freq * M / fs)
    second_index= int(high_freq * M / fs)

    # Set the values of the frequency response to 0.
    X[first_index:second_index+1] = 0 # for the first notch
    X[M - second_index:M - first_index+1] = 0 # for the mirrored notch


    #remove everything before 1Hz:
    # low_index = 0
    # high_index = int(1 * M / fs)
    # print(high_index)
    # X[low_index:high_index] = 0

    plt.plot(X, label='Frequency response')

    # x(n) is the sample domain of the filter:
    x = np.fft.ifft(X)
    # real value only so we don't take the imaginary part of the signal and leave the real part as it is.
    x_real = np.real(x)

    h = np.zeros(M)
    # Shift the signal to make it causal and swap -ve and +ve time:
    mid = int(M / 2)
    h[0:mid] = x_real[mid:M]
    h[mid:M] = x_real[0:mid]

    # fig = plt.figure(1)
    # self.plot(h, label=' Before Frequency response', figure=fig)

    # h = h * np.hamming(M)

    # self.plot(h, label='After Frequency response', figure=fig)

    # store the impulse response of the filter:
    # self.h = h

    #remove everything before 1Hz:
    # h[0:int(1 * M / fs)] = 0
    
    return h

class FIRfilter:

    def __init__(self, _coefficients):
        self.ntaps = len(_coefficients)
        self.coefficients = _coefficients
        self.buffer = np.zeros(self.ntaps)

    def dofilter(self, v):
        self.buffer = np.roll(self.buffer, shift=1)
        self.buffer[0] = v
        # print(self.buffer)

        # print(self.coefficients)
        # print(self.buffer)
        return np.sum(np.multiply(self.buffer, self.coefficients))
    

time, pulses = read_file("raw_data/person2_standing.dat")
fs = calculate_sampling_rate(len(pulses), time[-1])

# h_hp = calculate_coefficients(fs, [1, 0]) # 0 means no high freq specified, therefore, highpass
h = calculate_coefficients(fs, [45, 55])
# h = h_hp + h_bp
plt.plot(h)
fir_filter = FIRfilter(h)

filtered_pulse = []
for pulse in pulses:
    filtered_pulse.append(fir_filter.dofilter(pulse))

plt.plot(time, pulses, label='Raw pulse')
plt.plot(time, filtered_pulse, label='Filtered pulse')

plt.legend()
plt.show()

