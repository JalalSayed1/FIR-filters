from fir_filter import *
from lms import *
from ultils import *

# pulses = (pulse1, pulse2, pulse3):
time, pulse1, pulse2, pulse3 = read_file(
    "raw_data/person1_sleeping.dat")  # pulse1 works
# time, pulse1, pulse2, pulse3 = read_file("raw_data/person2_standing.dat")
pulses = pulse1
fs = calculate_sampling_rate(len(pulses), time[-1])

# h = [0.1, 0.4, 0.5, 0.4, 0.1]
# h = calculate_coefficients(fs, [45, 55], 1)

# fir_filter = FIRfilter(h)
ntaps = 100
# initial coefficients are all zeros:
lms_filter = LMSfilter(np.zeros(ntaps))

filtered_pulse = []

noise = 50  # Hz
DC = 0.5 # Hz
learning_rate = 0.009

# for pulse in [1,2,3,4,5]:
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

# plt.plot(h, label='Filter coefficients')
# plt.title('Filter coefficients')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')


# plt.plot(np.linspace(0, fs, len(pulses)), np.abs(np.fft.fft(pulses)), label='FFT of pulse')
# plt.plot(np.linspace(0, fs, len(pulses)), np.abs(np.fft.fft?(filtered_pulse)), label='FFT after filtering')

# plt.title('FFT of pulse')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')

plt.legend()
plt.show()

# save svg
# plt.savefig('filtered_data_with_hp.svg', format='svg', dpi=1200)
