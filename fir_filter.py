import numpy as np
from ultils import *
import matplotlib.pyplot as plt


def calculate_coefficients(fs, bs_cutoff_freqs, hp_cutoff_freq):
    '''Returns the coefficients of the filter.'''

    # calc resolution:
    # resolution = int(fs / M)

    # number of taps/coefficients
    M = int(fs/hp_cutoff_freq)
    print(f"Number of taps: {M}")

    low_freq, high_freq = bs_cutoff_freqs
    if high_freq == 0:
        high_freq = fs

    # X(k) is the frequency response of the filter:
    X = np.ones(M)

    # Multiply with M to get the index of the frequency as Fs is normalised frequency.
    first_index = int(low_freq * M / fs)
    second_index = int(high_freq * M / fs)

    hp_high_index = int(hp_cutoff_freq * M / fs)
    print(f"High pass index: {hp_high_index}")

    # Set the values of the frequency response to 0.
    X[first_index:second_index+1] = 0  # for the first notch
    X[M - second_index:M - first_index+1] = 0  # for the mirrored notch

    # remove everything before 1Hz:
    X[0:hp_high_index] = 0
    X[M - hp_high_index:M] = 0  # mirror

    # calculate y axis in db:
    # X = 20 * np.log10(X)

    # plt.plot(X, label='Frequency response')
    # plt.title(f"Frequency response of the filter with {low_freq}Hz and {high_freq}Hz notch")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude (dB)")

    # x(n) is the sample domain of the filter:
    x = np.fft.ifft(X)
    # real value only so we don't take the imaginary part of the signal and leave the real part as it is.
    x_real = np.real(x)

    h = np.zeros(M)
    # Shift the signal to make it causal and swap -ve and +ve time:
    mid = int(M / 2)
    h[0:mid] = x_real[mid:M]
    h[mid:M] = x_real[0:mid]

    h = h * np.hamming(M)

    # plt.plot(h, label='Filter coefficients after hamming window')

    # self.plot(h, label='After Frequency response', figure=fig)

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
        return np.sum(self.buffer * self.coefficients)  # dot product



# time, pulses = read_file("raw_data/person2_standing.dat")
# fs = calculate_sampling_rate(len(pulses), time[-1])

# # h_hp = calculate_coefficients(fs, [1, 0]) # 0 means no high freq specified, therefore, highpass
# h = calculate_coefficients(fs, [45, 55])
# # h = h_hp + h_bp
# plt.plot(h)
# fir_filter = FIRfilter(h)

# filtered_pulse = []
# # for pulse in pulses:
# for pulse in [1,2,3,4,5]:
#     filtered_pulse.append(fir_filter.dofilter(pulse))

# plt.plot(time, pulses, label='Raw pulse')
# plt.plot(time, filtered_pulse, label='Filtered pulse')

# plt.legend()
# plt.show()
