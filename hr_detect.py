from fir_filtering import FIRfilter, calculate_coefficients, filtering_with_FIR
import numpy as np
import matplotlib.pyplot as plt
from ultils import *
import scipy.signal as signal

def find_peaks(signal, threshold):
    '''Returns the peaks of the signal.
    signal: input signal
    threshold: threshold of the peaks
    '''
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append(i)
    return peaks, signal[peaks]

def detect_R_peaks(signal, fs):
    '''detect R peaks in ECG signal using a matched FIR filter.
    signal: ECG signal
    fs: sampling rate
    '''

    filtered_pulse = filtering_with_FIR(fs, signal)
    
    # find peaks in filtered signal:
    peaks, _ = find_peaks(filtered_pulse, threshold=0)

    # find the maximum of the peaks:
    max_peaks = []
    for peak in peaks:
        max_peaks.append(filtered_pulse[peak])

    # find the threshold:
    threshold = np.mean(max_peaks)

    # find the peaks that are above the threshold:
    R_peaks = []
    for peak in peaks:
        if filtered_pulse[peak] > threshold:
            R_peaks.append(peak)

    return R_peaks

def test(template, pulses):
    det = signal.lfilter(template, 1, pulses)

    return det**2


def matched_filter(pulses, fs):

    filtered_pulse = filtering_with_FIR(fs, pulses)

    # plt.plot(filtered_pulse, label='Filtered pulse')

    template = filtered_pulse[1200:1800]
    # template = filtered_pulse[4200:5000]

    # plt.plot(template, label='Template')


    template_reversed = template[::-1]


    detector = FIRfilter(template_reversed)

    detections = []

    for i, pulse in enumerate(pulses):
        detections.append(detector.dofilter(pulse))

    detections = [detection**2 for detection in detections]

    # plt.plot(detections, label='Detections')

    # check if detection above threshold:
    threshold = np.mean(detections)
    plt.plot([threshold]*len(detections), label='Threshold')

    for i, detection in enumerate(detections):
        if detection < threshold:
            detections[i] = 0

    plt.plot(detections, label='Detections')
    # apply heuristic to remove false positives:
    for i in range(1, len(detections)-1):
        if detections[i] != 0 and detections[i-1] == 0 and detections[i+1] == 0:
            detections[i] = 0

    # plt.plot(detections, label='Detections')
    


    # plt.plot(template, label='Template')
    plt.plot(detections, label='Detections after')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Matched filter')
    




if __name__ == "__main__":

    # time, pulse1, pulse2, pulse3 = read_file("raw_data/ecg_standing.dat")
    time, pulse1, pulse2, pulse3 = read_file("raw_data/ecg_standing.dat")
    pulses = pulse2
    fs = calculate_sampling_rate(len(pulses), time[-1])

    matched_filter(pulses, fs)

    # plt.plot(pulses, label='Raw pulse')
    # plt.title(f'Detected R peaks')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')

    plt.legend()
    plt.show()



