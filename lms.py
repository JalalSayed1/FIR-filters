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



time, pulse1, pulse2, pulse3 = read_file("raw_data/person1_sleeping.dat")
pulses = pulse1
fs = calculate_sampling_rate(len(pulses), time[-1])

ntaps = 100
# initial coefficients are all zeros:
lms_filter = LMSfilter(np.zeros(ntaps))

filtered_pulse = []

noise = 50  # Hz
DC = 0.5 # Hz
learning_rate = 0.009

for i, pulse in enumerate(pulses):
    # filtered_pulse.append(fir_filter.dofilter(pulse))
    ref_noise = np.sin(2*np.pi*noise/fs*i)
    ref_DC = np.sin(2*np.pi*DC/fs*i)

    removed_noise = lms_filter.doFilterAdaptive(pulse, ref_noise, learning_rate)
    removed_DC = lms_filter.doFilterAdaptive(removed_noise, ref_DC, learning_rate)

    filtered_pulse.append(removed_DC)

# plt.plot(pulses, label='Raw pulse')
plt.plot(filtered_pulse, label='Filtered pulse')
plt.title(f'Filtered pulse with learning rate = {learning_rate}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.legend()
plt.show()