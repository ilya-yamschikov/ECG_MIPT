import numpy as np
import logging

from src.code.features import BasicFeature
import src.code.calculator as clc


class SpectralDensity(BasicFeature):
    type = 'NUMERIC'

    def __init__(self):
        pass

    def run(self, ecg, begin=200.0, end=500.0, normalized=True):
        y = ecg.getHighFreq()
        fq = ecg.getDataFrequency()
        assert end <= fq / 2.
        if normalized:
            y = clc.normalize(y)
        f = fq / 2. * np.linspace(0.0, 1.0, len(y)/2 + 1)
        fft = np.absolute(np.fft.rfft(y)) / (0.5 * len(y))
        assert len(f) == len(fft), 'len f = %d, len fft = %d' % (len(f), len(fft))
        croppedFft = np.asarray([y for x,y in zip(f,fft) if begin < x and x < end])
        res = clc.RMS(croppedFft)
        logging.info('Spectral density [%.2f, %.2f] calculated on %s data: %f' % (begin, end, 'normalized' if normalized else 'not normalized', res))
        return res