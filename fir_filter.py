import numpy as np
from ultils import *
import matplotlib.pyplot as plt


def calculate_coefficients(fs, bs_cutoff_freqs, hp_cutoff_freq):
    '''Returns the coefficients of the filter.'''

    # calc resolution:
    # resolution = int(fs / M)

    # number of taps/coefficients
    M = int(fs/hp_cutoff_freq)

    low_freq, high_freq = bs_cutoff_freqs
    if high_freq == 0:
        high_freq = fs

    # X(k) is the frequency response of the filter:
    X = np.ones(M)

    # Multiply with M to get the index of the frequency as Fs is normalised frequency.
    first_index = int(low_freq * M / fs)
    second_index = int(high_freq * M / fs)

    hp_high_index = int(hp_cutoff_freq * M / fs)

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


def filtering_with_FIR(fs, pulses):
    # filter coefficients are a bandstop filter with cutoff frequencies of 45Hz and 55Hz and a high pass filter with cutoff frequency of 1Hz:
    h = calculate_coefficients(fs, [45, 55], 0.5)
    print(len(h))

    fir_filter = FIRfilter(h)

    filtered_pulse = []

    for i, pulse in enumerate(pulses):
        filtered_pulse.append(fir_filter.dofilter(pulse))
    return filtered_pulse


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


if __name__ == "__main__":

    time, pulse1, pulse2, pulse3 = read_file("raw_data/person1_sleeping.dat")
    pulses = pulse1
    fs = calculate_sampling_rate(len(pulses), time[-1])

    filtered_pulse = filtering_with_FIR(fs, pulses)

    plt.plot(pulses, label='Raw pulse')
    plt.plot(filtered_pulse, label='Filtered pulse')
    plt.title(f'Filtered pulse')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # plt.plot(h, label='Filter coefficients')
    # plt.title('Filter coefficients')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')

    # plt.plot(np.linspace(0, fs, len(pulses)), np.abs(np.fft.fft(pulses)), label='FFT of pulse')
    # plt.plot(np.linspace(0, fs, len(pulses)), np.abs(np.fft.fft?(filtered_pulse)), label='FFT after filtering')

    plt.legend()
    plt.show()
