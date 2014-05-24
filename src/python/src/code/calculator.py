import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

logging.basicConfig(level=logging.DEBUG)

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

def _is_local_max(x, idx, rng=3):
    if idx < 0 or idx >= len(x):
        raise ValueError('index %d out of bounds of x - len(x) = %d', (idx, len(x)))
    if len(x) < 2:
        raise ValueError('x too short: len(x) = ' % len(x))
    interval = x[max(0, idx - rng):min(len(x)-1, idx + rng + 1)]
    return np.all(interval <= x[idx])

def _try_lower_fq(fft, main_freq_idx):
    half_freq_candidates = range((main_freq_idx / 2 - 2), (main_freq_idx / 2 + 3))
    for idx in half_freq_candidates:
        if _is_local_max(fft, idx) and fft[idx] > .5 * fft[main_freq_idx]:
            return idx
    return main_freq_idx

def getMainFrequency(x, sampling_frequency):
    fft = np.absolute(np.fft.rfft(x)) / (0.5 * len(x))
    reasonable_interval = map(lambda f: get_fft_idx_by_frequency(f, sampling_frequency, len(x)), REASONABLE_MAIN_FQ)
    # plt.plot(fft, 'g-')
    # plt.show()
    fft_cropped = fft[reasonable_interval[0]:reasonable_interval[1]]
    # plt.plot(fft, 'g-')
    # plt.show()
    fq_idx = reasonable_interval[0] + np.argmax(fft_cropped)
    while True:
        new_fq_idx = _try_lower_fq(fft, fq_idx)
        if new_fq_idx == fq_idx or new_fq_idx < reasonable_interval[0]:
            break
        logging.debug('Updated main frequency candidate: %d -> %d' % (fq_idx, new_fq_idx))
        fq_idx = new_fq_idx
    main_freq = (sampling_frequency / 2.) * (fq_idx / float(len(x)/2))
    logging.debug('Main component of signal: %f' % main_freq)
    return main_freq

def aline(x, sampling_frequency):
    x = shift_on_zero(x)
    freq = getMainFrequency(x, sampling_frequency)
    trend_component = filterSignal(x, freq * .5, sampling_frequency, filterType='lowpass')
    plt.plot(x, 'g-', trend_component, 'r-')
    plt.show()
    return x - trend_component

def RMS(x):
    return math.sqrt(x.dot(x) / len(x))