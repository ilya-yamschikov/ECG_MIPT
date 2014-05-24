import math
import logging
import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


REASONABLE_MAIN_FQ = np.array([0.5, 3.0])


def filterSignal(y, fq, samplingFrequency, filterType='lowpass'):
    b,a = butter(4, fq / (samplingFrequency / 2.), btype=filterType)
    return filtfilt(b, a, y)

def normalize(x):
    return x / np.mean(np.absolute(x))

def shift_on_zero(x):
    return x - np.mean(x)

def get_fft_idx_by_frequency(fq, sampling_frequency, signal_len):
    return int(2. * fq / sampling_frequency * (signal_len / 2))

def getMainFrequency(x, sampling_frequency):
    fft = np.absolute(np.fft.rfft(x)) / (0.5 * len(x))
    reasonable_interval = map(lambda f: get_fft_idx_by_frequency(f, sampling_frequency, len(x)), REASONABLE_MAIN_FQ)
    fft = fft[reasonable_interval[0]:reasonable_interval[1]]
    fq_idx = reasonable_interval[0] + np.argmax(fft)
    return (sampling_frequency / 2.) * (fq_idx / float(len(x)/2))

def aline(x, sampling_frequency):
    x = shift_on_zero(x)
    freq = getMainFrequency(x, sampling_frequency)
    logging.debug('Main component of signal: %f' % freq)
    trend_component = filterSignal(x, freq, sampling_frequency / .5, filterType='lowpass')
    return x - trend_component

def RMS(x):
    return math.sqrt(x.dot(x) / len(x))