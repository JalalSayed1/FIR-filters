import numpy as np
from ultils import *
import matplotlib.pyplot as plt
from fir_filtering import FIRfilter, calculate_coefficients, filtering_with_FIR


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
            self.coefficients[i] = self.coefficients[i] + \
                learningRate * error * self.buffer[i]

        # self.coefficients = self.coefficients + learningRate * error * self.buffer

        canceller = signal - error
        output_signal = signal - canceller
        # return cleaned up ECG like before
        return output_signal


if __name__ == "__main__":

    time, pulse1, pulse2, pulse3 = read_file("raw_data/ecg_lying.dat")
    pulses = pulse1
    fs = calculate_sampling_rate(len(pulses), time[-1])

    ntaps = 100
    # initial coefficients are all zeros:
    lms_filter = LMSfilter(np.zeros(ntaps))

    filtered_pulse_LMS = []

    noise = 50  # Hz
    DC = 0.5  # Hz
    learning_rate = 0.001
    # learning_rates = [0.001, 0.005, 0.01, 0.09]

    # for learning_rate in learning_rates:
    #     plt.figure()
    for i, pulse in enumerate(pulses):
        # filtered_pulse.append(fir_filter.dofilter(pulse))
        ref_noise = np.sin(2*np.pi*noise/fs*i)
        ref_DC = np.sin(2*np.pi*DC/fs*i)

        removed_noise = lms_filter.doFilterAdaptive(
            pulse, ref_noise, learning_rate)
        removed_DC = lms_filter.doFilterAdaptive(
            removed_noise, ref_DC, learning_rate)

        filtered_pulse_LMS.append(removed_DC)

    Filtered_pulse_FIR = filtering_with_FIR(fs, pulses)

    # plt.plot(pulses, label='Raw pulse')
    plt.plot(filtered_pulse_LMS, label='Filtered pulse of LMS', color='orange')
    plt.plot(Filtered_pulse_FIR, label='Filtered pulse of FIR', color='green')
    plt.title(f'Filtered pulse with learning rate = {learning_rate}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.legend()
        # save plot as svg:
        # plt.savefig(f'./images/LMS_filtering_{learning_rate}.svg')

        # reset the list:
        # filtered_pulse_LMS = []


    plt.show()
