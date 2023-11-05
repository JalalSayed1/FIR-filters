from fir_filtering import FIRfilter, calculate_coefficients, filtering_with_FIR
import numpy as np
import matplotlib.pyplot as plt
from ultils import *
import scipy.signal as signal

def matched_filter(pulses, fs):

    filtered_pulse = filtering_with_FIR(fs, pulses)

    # plt.plot(filtered_pulse, label='Filtered pulse')

    template = filtered_pulse[1200:1800]

    # plt.plot(template, label='Template')

    # invert the template:
    template_reversed = template[::-1]


    detector = FIRfilter(template_reversed)

    detections = []

    for i, pulse in enumerate(filtered_pulse):
        detections.append(detector.dofilter(pulse))

    # square detections:
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



