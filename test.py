from fir_filter import *
from ultils import *

# pulses = (pulse1, pulse2, pulse3):
time, pulse1, pulse2, pulse3 = read_file("raw_data/person1_sleeping.dat") # pulse1 works
# time, pulse1, pulse2, pulse3 = read_file("raw_data/person2_standing.dat")
pulses = pulse1
fs = calculate_sampling_rate(len(pulses), time[-1])

# h = [0.1, 0.4, 0.5, 0.4, 0.1]
h1 = calculate_coefficients(fs, [45, 55], 1, np.hamming)
h2 = calculate_coefficients(fs, [45, 55], 1, np.hanning)
h3 = calculate_coefficients(fs, [45, 55], 1, np.blackman)

fir_filter1 = FIRfilter(h1)
fir_filter2 = FIRfilter(h2)
fir_filter3 = FIRfilter(h3)

filtered_pulse1 = []
filtered_pulse2 = []
filtered_pulse3 = []

# for pulse in [1,2,3,4,5]:
for pulse in pulses:
        filtered_pulse1.append(fir_filter1.dofilter(pulse))
        filtered_pulse2.append(fir_filter2.dofilter(pulse))
        filtered_pulse3.append(fir_filter3.dofilter(pulse))
        

# plt.plot(pulses, label='Raw pulse')
plt.plot(filtered_pulse3, label='Filtered pulse with blackman window')
plt.plot(filtered_pulse1, label='Filtered pulse with hamming window')
plt.plot(filtered_pulse2, label='Filtered pulse with hanning window')
plt.title('Filtered pulse')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# plt.plot(h, label='Filter coefficients')
# plt.title('Filter coefficients')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')


# plt.plot(np.linspace(0, fs,len(pulses)), np.abs(np.fft.fft(pulses)), label='FFT of pulse')
# plt.plot(np.linspace(0, fs,len(pulses)), np.abs(np.fft.fft(filtered_pulse)), label='FFT after filtering')
# plt.title('FFT of pulse')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')

plt.legend()
plt.show()

# save svg
# plt.savefig('filtered_data_with_hp.svg', format='svg', dpi=1200)