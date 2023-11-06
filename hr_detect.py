from fir_filtering import FIRfilter, calculate_coefficients, filtering_with_FIR
import numpy as np
import matplotlib.pyplot as plt
from ultils import *

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
    # plt.plot([threshold]*len(detections), label='Threshold')

    for i, detection in enumerate(detections):
        if detection < threshold:
            detections[i] = 0

    # plt.plot(detections, label='Detections')
    # apply heuristic to remove false positives:
    for i in range(1, len(detections)-1):
        if detections[i] != 0 and detections[i-1] == 0 and detections[i+1] == 0:
            detections[i] = 0

    # plt.plot(detections, label='Detections')
    


    # plt.plot(template, label='Template')
    # plt.plot(detections, label='Detections after')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.title('Matched filter')

    return detections
    
def calculate_hear_rate(detections, fs):
    # calculate heart rate:
    # find the time between two consecutive R peaks:
    R_peaks = []
    for i, detection in enumerate(detections):
        if detection != 0 and detection > detections[i-1] and detection > detections[i+1]:
            R_peaks.append(i)

    time_between_R_peaks = []
    for i in range(len(R_peaks)-1):
        time_between_R_peaks.append(
            (R_peaks[i+1] - R_peaks[i]) / fs)


    peak_time = np.array(R_peaks) / fs

    # heart_rates = []
    # for time in time_between_R_peaks:
    #     heart_rates.append(60 / time)
    # heart_rate = 60 / np.array(time_between_R_peaks)
    # print(f'{heart_rates}')

    # plt.figure()
    # plot momentory hear rate against time:
    # plt.plot(np.linspace(0, len(heart_rates), len(heart_rates)), heart_rates)
    return peak_time, R_peaks

if __name__ == "__main__":

    # time, pulse1, pulse2, pulse3 = read_file("raw_data/ecg_standing.dat")
    time, pulse1, pulse2, pulse3 = read_file("raw_data/ecg_standing.dat")
    pulses = pulse2
    fs = calculate_sampling_rate(len(pulses), time[-1])

    detections = matched_filter(pulses, fs)

    peak_time, R_peaks = calculate_hear_rate(detections, fs)

    # plot momentory hear rate against time:
    # plt.figure()
    plt.plot(peak_time, [pulses[i] for i in R_peaks], label='R peaks')

    # plt.plot(pulses, label='Raw pulse')
    plt.title(f'Detected R peaks')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.legend()
    plt.show()



