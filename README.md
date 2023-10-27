# FIR-filters
The task of this assignment is to filter an ECG with FIR filters and to detect the R peaks. In contrast to the FFT assignment we write filter code which can be used for realtime processing. This means that the FIR filter needs to be implemented with the help of delay lines and the impulse response is truncated.

## Notes

1. we take the `np.read(x)` bc x is in the time domain and the imaginary part (from `np.fft.ifft`) doesn't mean anything.


## for next lab

1. make FIR filter class that inherits from `Filter` class.
2. use the ring buffer to store and delete values.
